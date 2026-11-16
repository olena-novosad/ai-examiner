import gradio as gr
from dotenv import load_dotenv
from openai import (APIConnectionError, APIStatusError, AuthenticationError,
                    NotFoundError, RateLimitError)

from agent import PROVIDERS, ExamAgent
from prompts import GREETING, SKIP_MESSAGE
from tools import ExamError
import os

DESCRIPTION = """
# AI Examiner — інтелектуальний екзаменатор з NLP

AI-агент для автоматизованого проведення іспиту з обробки природної мови (NLP).

Система випадково обирає 3 теми, проводить іспит у форматі діалогу,
аналізує відповіді студента та за потреби ставить уточнювальні запитання.
Після завершення іспиту агент виставляє оцінку за шкалою від 0 до 10,
формує відгук і зберігає результати.

**Як розпочати:** оберіть провайдера та модель, введіть API-ключ, натисніть «Новий іспит»
та вкажіть своє ім'я й email.

**Тестовий студент:** `Test Student` — `test@test.com`.
"""


INITIAL_CHAT = [{"role": "assistant", "content": GREETING}]
NOT_STARTED = "**Статус:** натисніть «Новий іспит»."


def _api_error_text(e: Exception) -> str:
    if isinstance(e, AuthenticationError):
        return "Невірний API-ключ."
    if isinstance(e, NotFoundError):
        return "Модель не знайдено (404). Перевірте назву моделі."
    if isinstance(e, RateLimitError):
        return "Перевищено ліміт запитів (429). Зачекайте хвилину або оберіть іншу модель."
    if isinstance(e, APIConnectionError):
        return "Немає з'єднання з API. Перевірте інтернет."
    if isinstance(e, APIStatusError):
        print("API error body:", e.response.text)  # повний текст — у термінал
        return f"Помилка API {e.status_code}: {e.response.text[:300]}"
    return f"{type(e).__name__}: {e}"


def new_exam(api_key, provider, model):
    try:
        agent = ExamAgent(api_key, provider, model.strip() or None)
    except ExamError as e:
        raise gr.Error(str(e))
    return (agent, list(INITIAL_CHAT), agent.tool_calls_log, agent.status, None,
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))


def respond(user_text, chat, agent):
    if not user_text.strip():
        return user_text, chat, agent, gr.update(), gr.update(), gr.update()
    if agent is None:
        raise gr.Error("Натисніть «Новий іспит», щоб почати.")

    try:
        answer = agent.chat(user_text)
    except Exception as e:
        gr.Warning(_api_error_text(e))
        # текст відповіді лишаємо в полі, щоб його можна було надіслати ще раз
        return user_text, chat, agent, agent.tool_calls_log, agent.status, gr.update()

    chat = chat + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    result_file = str(agent.session.result_file) if agent.session.finished else None
    return "", chat, agent, agent.tool_calls_log, agent.status, result_file


def skip_topic(chat, agent):
    return respond(SKIP_MESSAGE, chat, agent)

def on_provider_change(provider):
    models = PROVIDERS[provider]["models"]
    return gr.update(choices=models, value=models[0])

with gr.Blocks(title="AI Examiner") as demo:
    gr.Markdown(DESCRIPTION)
    agent_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            provider = gr.Dropdown(list(PROVIDERS), value="Groq", label="Провайдер")
            model = gr.Dropdown(PROVIDERS["Groq"]["models"], value=PROVIDERS["Groq"]["models"][0],
                                label="Модель", allow_custom_value=True)
            api_key = gr.Textbox(label="API-ключ", type="password",
                                 placeholder="gsk_... / AIza... / sk-... ")
            start_btn = gr.Button("Новий іспит", variant="primary")
            status = gr.Markdown(NOT_STARTED)
            result_file = gr.File(label="Результат іспиту (JSON)", interactive=False)
            tool_log = gr.Textbox(visible=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(value=INITIAL_CHAT, height=520, label="Іспит")
            msg = gr.Textbox(placeholder="Ваша відповідь... ",
                             show_label=False, lines=2, interactive=False)
            with gr.Row():
                send_btn = gr.Button("Надіслати", variant="primary", interactive=False)
                skip_btn = gr.Button("Не знаю / пропустити тему", interactive=False)

    outputs = [msg, chatbot, agent_state, tool_log, status, result_file]
    provider.change(on_provider_change, provider, model)
    start_btn.click(new_exam, [api_key, provider, model],
                    [agent_state, chatbot, tool_log, status, result_file, msg, send_btn, skip_btn])
    msg.submit(respond, [msg, chatbot, agent_state], outputs)
    send_btn.click(respond, [msg, chatbot, agent_state], outputs)
    skip_btn.click(skip_topic, [chatbot, agent_state], outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
    )
