"""Проверка определения интентов и маршрутизации конечного автомата."""

import logging

from orchestrator import Orchestrator


logging.basicConfig(level=logging.INFO)


CASES: dict[str, list[str]] = {
    "question": [
        "Что говорится об авторстве публикации?",
        "Как работает этический комитет?",
        "Почему необходимо информированное согласие?",
        "asdf qwerty",
        "обычная фраза без команды",
        "",
        "   ",
    ],
    "verify": [
        "Проверь, верно ли это утверждение",
        "Это действительно правда?",
        "Ты уверен? Перепроверь по источникам",
    ],
    "clarify": [
        "Уточни, каких данных не хватает",
        "Объясни подробнее, что нужно указать",
        "Разъясни условия вопроса",
    ],
}


def bare_orchestrator() -> Orchestrator:
    """Создать автомат без запуска MCP-сервера и LLM-клиента."""
    orchestrator = object.__new__(Orchestrator)
    orchestrator._state = {}
    return orchestrator


def check_cases() -> None:
    orchestrator = bare_orchestrator()
    assert set(Orchestrator.INTENT_TO_ROLE) == {"question", "verify", "clarify"}

    for expected_intent, phrases in CASES.items():
        expected_role = Orchestrator.INTENT_TO_ROLE[expected_intent]
        for number, phrase in enumerate(phrases):
            session_id = f"{expected_intent}-{number}"
            intent = orchestrator.detect_intent(phrase)
            role = orchestrator.route(session_id, intent)

            assert intent in Orchestrator.INTENT_TO_ROLE, f"Неопределённый интент: {intent!r}"
            assert intent == expected_intent, (
                f"{phrase!r}: ожидался интент {expected_intent!r}, получен {intent!r}"
            )
            assert role == expected_role, (
                f"{phrase!r}: ожидалась роль {expected_role!r}, получена {role!r}"
            )
            print(f"{phrase:<48} интент={intent:<8} роль={role}")


def check_dialogue() -> None:
    orchestrator = bare_orchestrator()
    session_id = "dialogue"
    dialogue = [
        ("Что сказано об информированном согласии?", "analyst"),
        ("Проверь, действительно ли это правда", "verifier"),
        ("Уточни, каких условий не хватает", "clarifier"),
        ("Как эти условия влияют на решение?", "analyst"),
    ]

    for phrase, expected_role in dialogue:
        intent = orchestrator.detect_intent(phrase)
        role = orchestrator.route(session_id, intent)
        assert role == expected_role, (
            f"{phrase!r}: ожидалась роль {expected_role!r}, получена {role!r}"
        )


if __name__ == "__main__":
    check_cases()
    check_dialogue()
    print("Автомат и определение интентов: OK")
