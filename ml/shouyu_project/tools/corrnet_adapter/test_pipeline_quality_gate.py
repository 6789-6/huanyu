from __future__ import annotations

import unittest
from unittest.mock import patch

import pipeline


class StubTranslator:
    def __init__(self, text: str, hit_rate: float = 1.0):
        self.text = text
        self.hit_rate = hit_rate

    def translate(self, gloss: str) -> dict:
        return {
            "gloss": gloss,
            "text": self.text,
            "tokenCount": len([token for token in gloss.split("/") if token]),
            "unknownCount": 0,
            "vocabHitRate": self.hit_rate,
        }


class PipelineQualityGateTest(unittest.TestCase):
    def test_short_gloss_uses_readable_text_and_preserves_raw_g2t(self):
        with patch.object(
            pipeline,
            "G2TTranslator",
            lambda: StubTranslator("\u8fd9\u623f\u5b50\u5f88\u591a\u623f\u5b50\u3002"),
        ):
            result = pipeline.translate_gloss("1/\u623f\u5b50")

        self.assertEqual(result["text"], "\u623f\u5b50\u3002")
        self.assertEqual(result["rawG2TText"], "\u8fd9\u623f\u5b50\u5f88\u591a\u623f\u5b50\u3002")
        self.assertIs(result["fallback"], True)
        self.assertEqual(result["fallbackReason"], "short_gloss")

    def test_clean_sentence_keeps_g2t_translation(self):
        with patch.object(
            pipeline,
            "G2TTranslator",
            lambda: StubTranslator("\u6211\u9700\u8981\u5e2e\u52a9\u3002"),
        ):
            result = pipeline.translate_gloss("\u6211/\u9700\u8981/\u5e2e\u52a9/\u3002")

        self.assertEqual(result["text"], "\u6211\u9700\u8981\u5e2e\u52a9\u3002")
        self.assertIs(result["fallback"], False)
        self.assertEqual(result["fallbackReason"], "")


if __name__ == "__main__":
    unittest.main()
