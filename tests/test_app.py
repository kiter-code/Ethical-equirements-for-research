"""Проверка интерфейса Streamlit с изолированным тестовым ядром."""
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class FakeOrchestrator:
    reset_sessions = []

    def __init__(self):
        self.sessions = {}

    def handle(self, session_id, text):
        self.sessions.setdefault(session_id, []).append(text)
        return "Тестовый ответ: " + text

    def reset(self, session_id):
        self.sessions.pop(session_id, None)
        self.reset_sessions.append(session_id)


class AppTests(unittest.TestCase):
    def test_chat_isolation_and_reset(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        with patch("orchestrator.Orchestrator", FakeOrchestrator):
            first = AppTest.from_file(str(app)).run()
            second = AppTest.from_file(str(app)).run()
            self.assertFalse(first.exception)
            self.assertFalse(second.exception)
            session1 = first.session_state["session_id"]
            self.assertNotEqual(session1, second.session_state["session_id"])
            first.chat_input[0].set_value("Вопрос первой сессии").run()
            self.assertEqual(len(first.session_state["history"]), 2)
            self.assertEqual(second.session_state["history"], [])
            first.chat_input[0].set_value("Следующий вопрос").run()
            self.assertEqual(len(first.session_state["history"]), 4)
            first.button[0].click().run()
            self.assertEqual(first.session_state["history"], [])
            self.assertNotEqual(first.session_state["session_id"], session1)
            self.assertIn(session1, FakeOrchestrator.reset_sessions)


if __name__ == "__main__":
    unittest.main()
