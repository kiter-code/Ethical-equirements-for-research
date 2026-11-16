"""Проверки памяти на отдельной временной базе, не затрагивающие диалоги пользователя."""
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import config
import memory
from contracts import Chunk
from orchestrator import Orchestrator


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "memory.db"
        self.db_patch = patch.object(config, "DB_PATH", self.db)
        self.db_patch.start()
        memory.create_schema()
        memory.create_schema()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def test_window_order_isolation_and_clear(self):
        for i in range(1, 11):
            memory.save("first", "user" if i % 2 else "assistant", f"сообщение {i}")
        memory.save("second", "user", "чужая реплика")
        messages = memory.window("first", 4)
        self.assertEqual([m.content for m in messages], [f"сообщение {i}" for i in range(7, 11)])
        self.assertEqual([m.role for m in messages], ["user", "assistant", "user", "assistant"])
        datetime.fromisoformat(messages[0].timestamp)
        self.assertEqual(memory.window("first", 0), [])
        self.assertEqual(memory.window("first", -1), [])
        self.assertEqual(len(memory.window("first")), config.MEMORY_WINDOW)
        memory.clear("first")
        self.assertEqual(memory.window("first"), [])
        self.assertEqual([m.content for m in memory.window("second")], ["чужая реплика"])

    def test_persistence_across_processes(self):
        writer = (
            "import sys,config,memory; from pathlib import Path; "
            "config.DB_PATH=Path(sys.argv[1]); memory.create_schema(); "
            "memory.save('session', 'user', 'сохранено')"
        )
        reader = (
            "import sys,config,memory; from pathlib import Path; "
            "config.DB_PATH=Path(sys.argv[1]); "
            "assert [m.content for m in memory.window('session')]==['сохранено']"
        )
        subprocess.run([sys.executable, "-c", writer, str(self.db)], check=True)
        subprocess.run([sys.executable, "-c", reader, str(self.db)], check=True)

    def test_dialogue_history_reaches_client(self):
        class StubTools:
            def search(self, text):
                return [Chunk(text="Комитет проводит этическую экспертизу.",
                              source="example.pdf", page=1, category="ethics")]

            def close(self):
                pass

        class RecordingClient:
            def __init__(self):
                self.calls = []

            def count_tokens(self, text):
                return len(text)

            def generate(self, messages, temperature=0.3):
                self.calls.append(messages)
                return f"Тестовый ответ {len(self.calls)}"

        client = RecordingClient()
        with patch("orchestrator.MCPTools", StubTools), patch("orchestrator.build_client", return_value=client):
            o = Orchestrator()
            try:
                questions = ["Что делает комитет?", "Как он работает?", "А кого это касается?"]
                for question in questions:
                    o.handle("dialogue", question)
            finally:
                o.close()
        third = client.calls[2]
        self.assertEqual([m["role"] for m in third],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertIn(questions[0], third[1]["content"])
        self.assertEqual(third[2]["content"], "Тестовый ответ 1")
        self.assertIn(questions[1], third[3]["content"])
        self.assertEqual(third[4]["content"], "Тестовый ответ 2")
        self.assertTrue(third[5]["content"].endswith(questions[2]))


if __name__ == "__main__":
    unittest.main()
