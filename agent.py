
import json
import os
import time
from datetime import datetime

from openai import BadRequestError, InternalServerError, OpenAI, RateLimitError

from prompts import GREETING, SYSTEM_PROMPT
from tools import (MAX_ANSWERS_PER_TOPIC, TOOL_FUNCTIONS, TOOL_SCHEMAS, ExamError,
                   ExamSession, topic_materials, update_result_history)

PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"],
    },
}

MAX_STEPS = 10


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _remove_duplicate(text: str) -> str:
    t = text.strip()
    for i in range(len(t) // 2 - 20, len(t) // 2 + 21):
        if 0 < i < len(t):
            first, second = t[:i].strip(), t[i:].strip()
            if first and first == second:
                return first
    return t


class ExamAgent:
    def __init__(self, api_key: str, provider: str = "Groq", model: str | None = None):
        cfg = PROVIDERS[provider]
        api_key = api_key.strip() or os.getenv(cfg["env_key"], "").strip()  # ключ з поля або з .env
        if not api_key:
            raise ExamError(f"Введіть API-ключ або вкажіть {cfg['env_key']} у файлі .env")
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"], max_retries=1, timeout=60)
        self.model = model or cfg["models"][0]
        self.session = ExamSession()

        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": GREETING},
        ]

        self.history: list[dict] = []
        self._log("system", GREETING)


    def _log(self, role: str, content: str):
        self.history.append({"role": role, "content": content, "datetime": _now()})

    def _run_tool(self, name: str, args: dict) -> str:
        func = TOOL_FUNCTIONS.get(name)
        if name == "end_exam":
            # email, оцінку та історію підставляє код — модель не може їх підмінити
            args = {"email": self.session.email,
                    "score": self.session.average_score if self.session.scores else 0,
                    "history": list(self.history)}
        ok = False
        try:
            if func is None:
                raise ExamError(f"Невідомий інструмент: {name}")
            result = func(self.session, **args)
            ok = True
            if name == "get_next_topic" and isinstance(result, str):
                # функція повертає лише назву теми (як у завданні), а модель одразу отримує ще й шпаргалку
                result = {"topic": result, **topic_materials(result)}
        except ExamError as e:  # помилку повертаємо моделі, щоб вона могла відреагувати
            result = {"error": str(e)}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        args_for_log = {k: v for k, v in args.items() if k != "history"}
        self._log("tool_call", f"{name}({json.dumps(args_for_log, ensure_ascii=False)}) -> "
                               f"{json.dumps(result, ensure_ascii=False)}")

        # Після старту іспиту та після оцінки теми система САМА викликає get_next_topic
        # і додає наступну тему до результату: модель не може «забути» взяти тему.
        if ok and name in ("start_exam", "record_topic_score"):
            next_topic = json.loads(self._run_tool("get_next_topic", {}))
            result = {"result": result, "next_topic": next_topic}
        return json.dumps(result, ensure_ascii=False)

    def _call_model(self, force_tool: str | None = None):
        """Виклик моделі. При перевищенні ліміту (429) чекає і пробує ще раз."""
        params = {"model": self.model, "messages": self.messages,
                  "tools": TOOL_SCHEMAS, "temperature": 0.5}
        if force_tool:
            params["tools"] = [t for t in TOOL_SCHEMAS if t["function"]["name"] == force_tool]
            params["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        if "gpt-oss" in self.model:
            params["reasoning_effort"] = "low"

        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**params)
            except RateLimitError as e:
                if attempt == 2:
                    raise
                try:
                    wait = float(e.response.headers.get("retry-after", 15)) + 1
                except ValueError:
                    wait = 15
                print(f"Ліміт запитів (429), чекаю {wait:.0f} с...")
                time.sleep(wait)
            except InternalServerError:
                if attempt == 2:
                    raise
                wait = 5 * (attempt + 1)
                print(f"Сервер перевантажений (5xx), чекаю {wait} с...")
                time.sleep(wait)
            except BadRequestError as e:
                if "tool_choice" not in params:
                    raise
                print(f"Примусовий виклик {force_tool} не вдався, повторюю без примусу: {e}")
                params["tools"] = TOOL_SCHEMAS
                params.pop("tool_choice")

    def _required_tool(self) -> str | None:
        s = self.session
        if not s.started or s.finished:
            return None
        if s.current_topic and s.current_topic not in s.scores:
            return None
        if len(s.scores) == len(s.topics):
            return "end_exam"
        return "get_next_topic"

    @staticmethod
    def _assistant_message(msg) -> dict:
        calls = []
        for c in msg.tool_calls:
            call = {"id": c.id, "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
            extra = (c.model_extra or {}).get("extra_content")
            if extra:
                call["extra_content"] = extra
            calls.append(call)
        return {"role": "assistant", "content": msg.content or "", "tool_calls": calls}

    def chat(self, user_text: str) -> str:
        if self.session.finished:
            return "Іспит уже завершено. Щоб почати новий, натисніть «Новий іспит»."

        n_messages, n_history, n_answers = len(self.messages), len(self.history), self.session.answers_on_topic
        try:
            return self._chat(user_text)
        except Exception:
            # Якщо API впав ще до виклику будь-якого інструмента — відкочуємо повідомлення студента,
            # щоб при повторному надсиланні воно не задублювалося.
            if len(self.messages) == n_messages + 1:
                del self.messages[n_messages:]
                del self.history[n_history:]
                self.session.answers_on_topic = n_answers
            raise

    def _chat(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self._log("user", user_text)
        if self.session.current_topic and self.session.current_topic not in self.session.scores:
            self.session.answers_on_topic += 1

        force_tool = None
        if (self.session.current_topic and self.session.current_topic not in self.session.scores
                and self.session.answers_on_topic >= MAX_ANSWERS_PER_TOPIC):
            force_tool = "record_topic_score"

        for _ in range(MAX_STEPS):
            response = self._call_model(force_tool or self._required_tool())
            force_tool = None
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = _remove_duplicate(msg.content or "") or "Перепрошую, повторіть, будь ласка, відповідь."
                s = self.session
                if s.started and not s.finished and s.topics and len(s.scores) == len(s.topics):
                    # Модель написала підсумок, але не викликала end_exam — система зберігає результат сама
                    self._run_tool("end_exam", {})
                self.messages.append({"role": "assistant", "content": text})
                self._log("system", text)
                if self.session.finished:  # фінальне повідомлення теж має бути в збереженій історії
                    update_result_history(self.session, self.history)
                return text

            self.messages.append(self._assistant_message(msg))
            if msg.content:  # інколи модель пише текст разом із викликом інструмента
                self._log("system", msg.content)

            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(call.function.name, args)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        return "Вибачте, сталася технічна помилка. Спробуйте переформулювати відповідь."

    @property
    def status(self) -> str:
        s = self.session
        if not s.started:
            return "**Статус:** очікую ім'я та email студента."
        if s.finished:
            return f"**Іспит завершено · оцінка {s.final_score}/10**"
        n = len(s.scores) + (1 if s.current_topic and s.current_topic not in s.scores else 0)
        topic = s.current_topic or "перехід до наступної теми"
        return (f"**Тема {max(n, 1)}/{len(s.topics)}:** {topic}  \n"
                f"Відповідей на тему: {s.answers_on_topic}/{MAX_ANSWERS_PER_TOPIC}")

    @property
    def tool_calls_log(self) -> str:
        """Лог викликів інструментів для панелі налагодження в UI."""
        lines = [h["content"] for h in self.history if h["role"] == "tool_call"]
        return "\n\n".join(lines) if lines else "Інструменти ще не викликалися."