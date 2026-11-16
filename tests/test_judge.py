"""Проверка судьи и подсчёта метрик без вызовов модели и изменения БД."""
import json
import unittest
from unittest.mock import patch

from contracts import Chunk
import test_suite


class Client:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.temperature = None

    def generate(self, messages, temperature=0.3):
        self.messages, self.temperature = messages, temperature
        return self.response


class JudgeTests(unittest.TestCase):
    def test_structured_response_and_context(self):
        client = Client('```json\n{"faithfulness":0.4,"answer_relevance":0.8,"context_precision":0.6,"comment":"тест"}\n```')
        chunk = Chunk(text="текст источника", source="source.pdf", page=7, category="x")
        result = test_suite.judge(client, "вопрос", "ответ", "эталон", [chunk], "direct")
        self.assertEqual(result["faithfulness"], 0.4)
        self.assertEqual(client.temperature, 0.0)
        self.assertIn("[source.pdf, стр. 7]", client.messages[1]["content"])
        self.assertIn("текст источника", client.messages[1]["content"])

    def test_invalid_and_out_of_scope(self):
        for raw in ('ошибка', '{"faithfulness":2,"answer_relevance":0,"context_precision":0,"comment":""}',
                    '{"faithfulness":NaN,"answer_relevance":0,"context_precision":0,"comment":""}'):
            self.assertTrue(test_suite.judge(Client(raw), "q", "a", "e", [], "direct")["comment"].startswith("разбор:"))
        raw = '{"faithfulness":1,"answer_relevance":1,"context_precision":1,"comment":"отказ"}'
        verdict = test_suite.judge(Client(raw), "q", "a", "e", [], "out_of_scope")
        self.assertEqual(verdict["context_precision"], 0.0)
        self.assertIn("не определена", verdict["comment"])

    def test_run_excludes_failures_and_out_of_scope_precision(self):
        class FakeOrchestrator:
            def __init__(self):
                self._context = {}
            def handle(self, session_id, question):
                self._context[session_id] = [Chunk("основание", "a.pdf", 1, "x")]
                return "ответ"
            def reset(self, session_id):
                self._context.pop(session_id, None)
            def close(self):
                pass

        responses = iter([
            '{"faithfulness":1,"answer_relevance":1,"context_precision":0.5,"comment":"ok"}',
            'не JSON',
            '{"faithfulness":1,"answer_relevance":1,"context_precision":1,"comment":"отказ"}',
        ])
        client = Client("")
        client.generate = lambda messages, temperature=0.3: next(responses)
        cases = [{"id": 1, "question": "q1", "expected": "e", "type": "direct"},
                 {"id": 2, "question": "q2", "expected": "e", "type": "direct"},
                 {"id": 3, "question": "q3", "expected": "e", "type": "out_of_scope"}]
        with patch.object(test_suite, "Orchestrator", FakeOrchestrator), patch.object(test_suite, "build_client", return_value=client), patch.object(test_suite.DATASET.__class__, "read_text", return_value=json.dumps(cases)):
            report = test_suite.run()
        self.assertEqual(report["failures"], [2])
        self.assertEqual(report["means"]["in_scope"]["context_precision"], 0.5)
        self.assertIsNone(report["means"]["out_of_scope"]["context_precision"])


if __name__ == "__main__":
    unittest.main()
