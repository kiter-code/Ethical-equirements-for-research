import logging

from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)

o = Orchestrator()

phrases = [
    "Что такое машинное обучение?",
    "Почему нейронные сети так популярны?",
    "Перепиши ответ более официальным языком",
    "Перефразируй этот текст",
    "Проверь, верно ли это утверждение",
    "Это действительно правда?",
    "Объясни подробнее",
    "Уточни свой ответ",
    "asdf qwerty",
]

session_id = "test"

try:
    for phrase in phrases:
        intent = o.detect_intent(phrase)
        role = o.route(session_id, intent)

        print(
            f"Фраза: {phrase}\n"
            f"Интент: {intent}\n"
            f"Роль: {role}\n"
            f"{'-' * 50}"
        )
finally:
    o.close()