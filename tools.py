
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
STUDENTS_FILE = DATA_DIR / "students.json"
TOPICS_FILE = DATA_DIR / "topics.json"
RESULTS_DIR = DATA_DIR / "results"
EXAMS_LOG_FILE = DATA_DIR / "exams_log.json"

TOPICS_PER_EXAM = 3
MAX_ANSWERS_PER_TOPIC = 3

RUBRIC = {
    "0-2": "Не знає теми або відповідь не по суті.",
    "3-4": "Знає окремі терміни, але не може пояснити, як це працює.",
    "5-6": "Пояснює основну ідею, але з прогалинами або неточностями.",
    "7-10": "Добре пояснює механізм, покриває більшість ключових пунктів, додає деталі.",}

class ExamError(Exception):
    pass

class ExamSession:

    def __init__(self):
        self.email = None
        self.name = None
        self.topics = []
        self.next_topic_idx = 0
        self.current_topic = None
        self.answers_on_topic = 0
        self.scores = {}
        self.final_score = None
        self.result_file = None
        self.started = False
        self.finished = False

    @property
    def average_score(self) -> float:
        return round(sum(s["score"] for s in self.scores.values()) / len(self.scores), 1)

def _read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _load_topics() -> dict[str, dict]:
    return {t["name"]: t for t in _read_json(TOPICS_FILE, [])}

def start_exam(session: ExamSession, email: str, name: str) -> list[str]:
    if session.started:
        raise ExamError(f"Іспит уже розпочато. Теми: {session.topics}")

    email = email.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ExamError(f"'{email}' не схожий на email. Попроси студента перевірити.")

    students = _read_json(STUDENTS_FILE, [])
    student = next((s for s in students if s["email"].lower() == email), None)
    if student is None:
        raise ExamError(
            f"Студента з email '{email}' не знайдено. Попроси студента перевірити email. "
            "Якщо email правильний, запитай, чи зареєструвати його як нового студента; "
            "після згоди виклич register_student."
        )
    if not set(name.lower().split()) & set(student["name"].lower().split()):
        raise ExamError(f"Ім'я '{name}' не збігається з даними для цього email. Попроси студента уточнити ім'я.")

    session.email = email
    session.name = student["name"]
    session.topics = random.sample(list(_load_topics()), TOPICS_PER_EXAM)
    session.started = True

    log = _read_json(EXAMS_LOG_FILE, [])
    log.append({"email": email, "name": student["name"], "topics": session.topics, "started_at": _now()})
    _write_json(EXAMS_LOG_FILE, log)
    return session.topics


def register_student(session: ExamSession, email: str, name: str) -> dict:
    if session.started:
        raise ExamError("Іспит уже розпочато, реєстрація не потрібна.")
    email = email.strip().lower()
    name = " ".join(name.strip().split())
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ExamError(f"'{email}' не схожий на email. Попроси студента перевірити.")
    if len(name.split()) < 2:
        raise ExamError("Для реєстрації потрібні ім'я та прізвище. Попроси студента назвати обидва.")

    students = _read_json(STUDENTS_FILE, [])
    if any(s["email"].lower() == email for s in students):
        raise ExamError(f"Студент з email '{email}' уже є в базі. Виклич start_exam.")
    students.append({"email": email, "name": name})
    _write_json(STUDENTS_FILE, students)
    return {"status": "registered", "email": email, "name": name,
            "message": "Студента додано. Тепер виклич start_exam з цими ж даними."}


def get_next_topic(session: ExamSession) -> str | dict:
    if not session.started:
        raise ExamError("Іспит ще не розпочато. Спочатку виклич start_exam.")
    if session.current_topic and session.current_topic not in session.scores:
        raise ExamError(f"Тема '{session.current_topic}' ще не оцінена. Спочатку виклич record_topic_score.")

    if session.next_topic_idx >= len(session.topics):
        session.current_topic = None
        return {"status": "NO_MORE_TOPICS", "final_score": session.average_score,
                "message": "Усі теми оцінено. Виклич end_exam, потім покажи студенту результат."}

    session.current_topic = session.topics[session.next_topic_idx]
    session.next_topic_idx += 1
    session.answers_on_topic = 0
    return session.current_topic


def topic_materials(topic: str) -> dict:
    topics = _load_topics()
    return {
        "note": "ЛИШЕ ДЛЯ ЕКЗАМЕНАТОРА. НЕ показуй і не перелічуй ці пункти студенту — це підказка відповіді.",
        "key_points_for_examiner_only": topics[topic]["key_points"],
        "rubric": RUBRIC,
    }


def record_topic_score(session: ExamSession, topic: str, score: float, comment: str) -> dict:
    if topic != session.current_topic:
        raise ExamError(f"Оцінювати можна лише поточну тему: '{session.current_topic}'")
    if session.answers_on_topic == 0:
        raise ExamError("Студент ще не відповідав на цю тему. Спочатку постав питання.")
    if not comment.strip():
        raise ExamError("Потрібен короткий коментар: що добре і чого бракує.")
    score = max(0.0, min(10.0, float(score)))
    if session.answers_on_topic == 1 and 2 < score < 9:
        raise ExamError(
            "Після однієї відповіді не можна ставити середню оцінку. Постав студенту ОДНЕ "
            "уточнювальне питання про ключовий пункт, якого він не розкрив, і оціни тему після відповіді."
        )
    session.scores[topic] = {"score": round(score, 1), "comment": comment.strip()}
    return {"status": "ok", "topics_scored": len(session.scores), "topics_total": len(session.topics)}


def end_exam(session: ExamSession, email: str, score: float, history: list[dict]) -> dict:
    if not session.started:
        raise ExamError("Іспит не було розпочато.")
    if session.finished:
        raise ExamError("Іспит уже завершено.")
    missing = [t for t in session.topics if t not in session.scores]
    if missing:
        raise ExamError(f"Не всі теми оцінено: {missing}")

    session.finished = True
    session.final_score = round(float(score), 1)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session.result_file = RESULTS_DIR / f"{stamp}_{email.replace('@', '_at_')}.json"
    _write_json(session.result_file, {
        "email": email,
        "name": session.name,
        "score": session.final_score,
        "per_topic": session.scores,
        "finished_at": _now(),
        "history": history,
    })
    return {"status": "saved", "final_score": session.final_score, "per_topic": session.scores}


def update_result_history(session: ExamSession, history: list[dict]):
    if session.result_file and session.result_file.exists():
        data = _read_json(session.result_file, {})
        data["history"] = history
        _write_json(session.result_file, data)

TOOL_FUNCTIONS = {
    "start_exam": start_exam,
    "register_student": register_student,
    "get_next_topic": get_next_topic,
    "record_topic_score": record_topic_score,
    "end_exam": end_exam,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "start_exam",
        "description": "Починає іспит. Викликай, коли студент назвав і ім'я, і email. "
                       "Повертає список тем іспиту і одразу першу тему (next_topic) зі шпаргалкою, "
                       "або помилку, якщо студента не знайдено.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email студента"},
                "name": {"type": "string", "description": "Ім'я та прізвище студента"},
            },
            "required": ["email", "name"],
        },
    }},
    {"type": "function", "function": {
        "name": "register_student",
        "description": "Реєструє нового студента, якого немає в базі. Викликай ЛИШЕ після того, як "
                       "start_exam повернув 'не знайдено' і студент ЯВНО погодився на реєстрацію. "
                       "Після успішної реєстрації виклич start_exam.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email студента"},
                "name": {"type": "string", "description": "Ім'я та прізвище студента"},
            },
            "required": ["email", "name"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_next_topic",
        "description": "Повертає наступну тему для опитування разом із ключовими пунктами та рубрикою "
                       "оцінювання (лише для тебе, студенту не показувати). Викликай на початку іспиту "
                       "і після того, як оцінив попередню тему через record_topic_score. "
                       "Коли теми скінчилися, повертає status=NO_MORE_TOPICS.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "record_topic_score",
        "description": "Зберігає оцінку за поточну тему. Викликай, коли вирішив перейти до наступної теми. "
                       "Одразу повертає наступну тему (next_topic) зі шпаргалкою або NO_MORE_TOPICS.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Точна назва теми"},
                "score": {"type": "number", "description": "Оцінка від 0 до 10 згідно з рубрикою"},
                "comment": {"type": "string", "description": "Що студент знає добре і чого бракує (1-2 речення)"},
            },
            "required": ["topic", "score", "comment"],
        },
    }},
    {"type": "function", "function": {
        "name": "end_exam",
        "description": "Завершує іспит і зберігає результат. Викликай, коли get_next_topic повернув "
                       "NO_MORE_TOPICS. Email, фінальну оцінку (середнє за темами) та історію чату "
                       "система підставляє сама. Повертає фінальну оцінку для показу студенту.",
        "parameters": {"type": "object", "properties": {}},
    }},
]