import unittest

from ingest import clean_text, chunk_text, detect_category


class CleanTextTests(unittest.TestCase):
    def test_normalizes_pdf_artifacts(self):
        raw = "Исследова-\nние\u00a0  научно - исследовательское\nстр. 12\nпродолжается"
        self.assertEqual(
            clean_text(raw),
            "Исследование научно-исследовательское продолжается",
        )


class ChunkTextTests(unittest.TestCase):
    def test_chunks_have_requested_overlap(self):
        chunks = chunk_text("abcdefghij", size=6, overlap=2)
        self.assertEqual(chunks, ["abcdef", "efghij", "ij"])
        self.assertEqual(chunks[0][-2:], chunks[1][:2])

    def test_empty_text(self):
        self.assertEqual(chunk_text("", size=6, overlap=2), [])

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("text", size=4, overlap=4)


class DetectCategoryTests(unittest.TestCase):
    def test_detects_category_by_keywords(self):
        self.assertEqual(
            detect_category("Рукопись содержит плагиат.", "other.pdf"),
            "publication_ethics",
        )

    def test_uses_source_for_fallback(self):
        self.assertEqual(
            detect_category("Нейтральный фрагмент.", "полож_об_эт_комитете.pdf"),
            "ethics_committee",
        )

    def test_always_returns_nonempty_default(self):
        self.assertEqual(detect_category("Нейтральный фрагмент.", "other.pdf"), "general_ethics")


if __name__ == "__main__":
    unittest.main()
