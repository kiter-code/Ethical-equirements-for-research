"""Необязательный локальный мост для оценки через LM Studio, без замены OllamaClient.

Запуск: python scripts/evaluate_lm_studio.py --output reports/practice-06-results.json
Требует запущенный сервер LM Studio с qwen/qwen3.5-9b на localhost:1234.
Для реальной проверки именно Ollama используйте python test_suite.py напрямую.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import requests


ROOT = Path(__file__).resolve().parents[1]


class Bridge(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            if self.path != "/api/chat":
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = request["messages"]
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            conversation = "\n\n".join(
                f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"
            )
            response = requests.post("http://localhost:1234/api/v1/chat", json={
                "model": "qwen/qwen3.5-9b", "system_prompt": system, "input": conversation,
                "temperature": request.get("options", {}).get("temperature", 0.3),
                "reasoning": "off", "max_output_tokens": 512, "store": False,
            }, timeout=165)
            response.raise_for_status()
            answer = "".join(x["content"] for x in response.json()["output"] if x["type"] == "message")
            if not answer.strip():
                raise ValueError("модель вернула пустой ответ")
            body = json.dumps({"message": {"role": "assistant", "content": answer}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            print(f"Мост LM Studio: {exc}", file=sys.stderr, flush=True)
            self.send_error(502, "LM Studio request failed")

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/practice-06-results.json")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 11439), Bridge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = os.environ.copy()
    env.update({"OLLAMA_HOST": "http://127.0.0.1:11439", "LLM_BACKEND": "ollama"})
    try:
        result = subprocess.run([sys.executable, "-u", str(ROOT / "test_suite.py"),
                                 "--output", str(args.output)], cwd=ROOT, env=env)
        return result.returncode
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
