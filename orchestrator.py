"""
orchestrator.py — ядро системы: конечный автомат, переключающий роли агентов.

ПРАКТИКА 3. Что делает студент:
  1. описать роли в ROLES — минимум две, с разными зонами ответственности
  2. заполнить INTENT_TO_ROLE и TRANSITIONS — граф переходов автомата
  3. реализовать detect_intent() — определение интента пользователя

Готово и трогать не нужно: route(), handle(), подключение MCP и LLM.

Критерии приёмки:
  - ролей не меньше двух, у каждой описана своя зона ответственности
  - граф переходов задан структурой данных, а не цепочкой if по всему файлу
  - смена интента в диалоге переключает роль, это видно в логе
  - неизвестный интент не роняет процесс, а уходит в состояние по умолчанию
  - контекст при переходе не теряется: вторая роль видит найденное первой
  - вызов инструмента идёт через MCP-клиент, а не прямым импортом функции поиска

НЕ МЕНЯТЬ: сигнатуру handle(session_id, text) -> str.
"""

import logging
import re

import guardrails
import memory
import prompts
from contracts import Chunk
from llm_client import build_client
from mcp_client import MCPTools

log = logging.getLogger(__name__)

BLOCKED_REPLY = "Запрос отклонён слоем безопасности. Переформулируйте вопрос по существу темы."


class Orchestrator:
    """Оркестратор ролей на конечном автомате."""

    # --- Роли: у каждой своя зона ответственности ---
    ROLES: dict[str, dict] = {
        "analyst": {
            "description": (
                "Ищет применимые нормы в базе знаний и отвечает со ссылками на источники; "
                "не подтверждает спорные утверждения без отдельной проверки. "
                "Меняет исход, когда найденная норма позволяет дать доказательный ответ."
            ),
            "temperature": 0.2,
            "filters": {},
        },
        "verifier": {
            "description": (
                "Сверяет утверждение с найденными фрагментами и отказывает в подтверждении, "
                "если доказательств недостаточно; не формулирует новый ответ вместо аналитика. "
                "Меняет исход, подтверждая либо опровергая спорное утверждение."
            ),
            "temperature": 0.1,
            "filters": {},
        },
        "clarifier": {
            "description": (
                "Выявляет недостающие условия и задаёт один уточняющий вопрос; "
                "не делает окончательных выводов и не подменяет аналитика. "
                "Меняет исход, останавливая преждевременный ответ до получения условий."
            ),
            "temperature": 0.2,
            "filters": {},
        },
    }

    INITIAL_ROLE = "analyst"
    FALLBACK_ROLE = "analyst"

    # --- Граф переходов конечного автомата ---
    TRANSITIONS: dict[str, set[str]] = {
        "analyst": {"analyst", "verifier", "clarifier"},
        "verifier": {"analyst", "verifier", "clarifier"},
        # Повторное уточнение без нового ответа запрещено: сначала управление
        # должно вернуться аналитику или верификатору.
        "clarifier": {"analyst", "verifier"},
    }

    # --- Соответствие интентов ролям ---
    INTENT_TO_ROLE: dict[str, str] = {
        "question": "analyst",
        "verify": "verifier",
        "clarify": "clarifier",
    }

    def __init__(self):
        self.tools = MCPTools()
        self.llm = build_client()
        memory.create_schema()
        self._state: dict[str, str] = {}      # session_id -> текущая роль
        self._context: dict[str, list[Chunk]] = {}  # session_id -> что нашла прошлая роль
        self.usage: list[dict] = []           # расход токенов по вызовам, на нём считается TCO (П8)

    # --- Реализация студента: определение интента ---

    def detect_intent(self, text: str) -> str:
        """Определить интент реплики. Возвращает ключ из INTENT_TO_ROLE.

        Используется простая эвристика на основе ключевых слов.
        LLM не вызывается: маршрутизация остаётся предсказуемой.

        Интент по умолчанию — "question", он направляется к аналитику.
        """

        text_lower = text.lower().strip()

        # 1. Проверка фактов и утверждений
        verify_patterns = [
            r"\bпроверить\b",
            r"\bпроверь\b",
            r"\bпроверьте\b",
            r"\bперепроверь\b",
            r"\bперепроверьте\b",
            r"\bправда ли\b",
            r"\bдействительно правда\b",
            r"\bверно ли\b",
            r"\bневерно\b",
            r"\bправильно ли\b",
            r"\bкорректно ли\b",
            r"\bты уверен\b",
            r"\bвы уверены\b",
            r"\bподтверди\b",
            r"\bверифицируй\b",
        ]

        if any(re.search(pattern, text_lower) for pattern in verify_patterns):
            return "verify"

        # 2. Уточнение вопроса
        clarify_patterns = [
            r"\bуточнить\b",
            r"\bуточни\b",
            r"\bуточните\b",
            r"\bкак уточнить\b",
            r"\bточнее\b",
            r"\bподробнее\b",
            r"\bразъясни\b",
            r"\bразъясните\b",
        ]

        if any(re.search(pattern, text_lower) for pattern in clarify_patterns):
            return "clarify"

        # 3. Обычный вопрос
        if "?" in text_lower:
            return "question"

        # Важно: проверяем наличие слова, а не просто непустой кортеж.
        question_words = (
            "что",
            "как",
            "почему",
            "когда",
            "где",
            "кто",
        )

        if any(
            re.search(rf"\b{word}\b", text_lower)
            for word in question_words
        ):
            return "question"

        # Нераспознанная или пустая реплика получает определённый интент по умолчанию.
        return "question"

    # --- Готовая механика: не менять ---

    def route(self, session_id: str, intent: str) -> str:
        """Перевести автомат в новое состояние и вернуть активную роль."""
        current = self._state.get(session_id, self.INITIAL_ROLE)
        target = self.INTENT_TO_ROLE.get(intent, self.FALLBACK_ROLE)

        allowed = self.TRANSITIONS.get(current, {self.FALLBACK_ROLE})
        rolled_back = target not in allowed
        if rolled_back:
            log.warning("Переход %s -> %s запрещён, откат в %s",
                        current, target, self.FALLBACK_ROLE)
            target = self.FALLBACK_ROLE

        if target != current:
            # Про откат сказано прямо: иначе строка «verifier -> analyst (интент verify)»
            # читается как «интент verify обслуживается ролью analyst», то есть наоборот.
            log.info("Сессия %s: роль %s -> %s (интент %s%s)",
                     session_id, current, target, intent, ", откат" if rolled_back else "")

        self._state[session_id] = target
        return target

    def handle(self, session_id: str, text: str) -> str:
        """Единственная точка входа. Оба канала и тесты идут сюда.

        Конвейер: guardrail -> память -> интент -> роль -> поиск через MCP ->
        сборка промпта -> LLM -> память -> ответ.
        """
        verdict = guardrails.check(text)
        if not verdict.allowed:
            guardrails.log_attempt(session_id, text, verdict)
            log.warning("Сессия %s: запрос заблокирован (%s)", session_id, verdict.reason)
            return BLOCKED_REPLY

        previous_role = self._state.get(session_id, self.INITIAL_ROLE)
        intent = self.detect_intent(text)
        role = self.route(session_id, intent)
        role_config = self.ROLES[role]

        filters = role_config.get("filters") or {}
        if filters:
            chunks = self.tools.search_filtered(text, filters)
        else:
            chunks = self.tools.search(text)

        # Контекст переезжает между ролями: при переключении новая роль видит
        # и свои фрагменты, и то, что нашла предыдущая, — иначе разговор рвётся.
        # Признак переноса — смена роли, а не пустая выдача: поиск по реплике
        # «а в моём случае?» возвращает не ноль фрагментов, а пять нерелевантных.
        if role != previous_role:
            known = {(c.source, c.page, c.text[:80]) for c in chunks}
            chunks = chunks + [c for c in self._context.get(session_id, [])
                               if (c.source, c.page, c.text[:80]) not in known]
        self._context[session_id] = chunks

        history = memory.window(session_id)

        # Реплика пользователя сохраняется после чтения окна, а не до него.
        # Иначе текущий вопрос уходил бы в модель дважды — в истории и в блоке
        # с фрагментами, — а полезная глубина окна была бы N−1, а не N.
        memory.save(session_id, "user", text)

        messages = prompts.build_messages(role, text, chunks, history)

        # Считается одним вызовом по склеенному тексту, а не по сообщению на вызов:
        # у облачного клиента count_tokens() — обращение к API, и поштучный подсчёт
        # добавлял бы к каждому вопросу столько сетевых запросов, сколько сообщений.
        prompt_size = self.llm.count_tokens("\n".join(m["content"] for m in messages))
        log.info("Сессия %s: роль %s, чанков %d, промпт ~%d токенов",
                 session_id, role, len(chunks), prompt_size)

        answer = self.llm.generate(messages, temperature=role_config["temperature"])

        # Вход и выход считаются раздельно: они тарифицируются по разным ценам (П8)
        self.usage.append({
            "session_id": session_id,
            "role": role,
            "prompt_tokens": prompt_size,
            "completion_tokens": self.llm.count_tokens(answer),
        })

        memory.save(session_id, "assistant", answer)
        return answer

    def reset(self, session_id: str) -> None:
        """Сбросить сессию: состояние автомата, контекст и историю."""
        self._state.pop(session_id, None)
        self._context.pop(session_id, None)
        memory.clear(session_id)

    def close(self) -> None:
        self.tools.close()
