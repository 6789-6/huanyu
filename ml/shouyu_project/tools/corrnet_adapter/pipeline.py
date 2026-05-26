from __future__ import annotations

import argparse
import time
from pathlib import Path

from corrnet_smoke import run_corrnet
from g2t_infer import G2TTranslator

LOW_HIT_RATE = 0.60
SHORT_GLOSS_TOKEN_LIMIT = 2
PUNCT = {"\u3002", "\uff0c", "\uff01", "\uff1f", ",", ".", "!", "?"}
SENTENCE_END = {"\u3002", "\uff01", "\uff1f", ".", "!", "?"}


def gloss_tokens(gloss: str) -> list[str]:
    return [token.strip() for token in gloss.split("/") if token.strip()]


def content_tokens(gloss: str) -> list[str]:
    return [
        token
        for token in gloss_tokens(gloss)
        if token not in PUNCT and not token.isdigit()
    ]


def gloss_to_readable_text(gloss: str) -> str:
    tokens = gloss_tokens(gloss)
    if not tokens:
        return ""
    text = "".join(
        "" if token.isdigit() else token
        for token in tokens
        if token not in PUNCT or token in SENTENCE_END
    )
    if text and text[-1] not in SENTENCE_END:
        text += "\u3002"
    return text


def fallback_reason(gloss: str, result: dict) -> str:
    text = result.get("text", "").strip()
    tokens = content_tokens(gloss)
    if not text:
        return "empty_translation"
    if result.get("vocabHitRate", 0.0) < LOW_HIT_RATE:
        return "low_vocab_hit_rate"
    if len(tokens) <= SHORT_GLOSS_TOKEN_LIMIT:
        return "short_gloss"
    if any(len(token) >= 2 and text.count(token) > 1 for token in set(tokens)):
        return "repeated_source_term"
    readable_len = len(gloss_to_readable_text(gloss).rstrip("\u3002"))
    translated_len = len(text.rstrip("\u3002\uff01\uff1f.!?"))
    if readable_len and translated_len > max(12, readable_len * 3):
        return "over_expanded"
    return ""


def translate_gloss(gloss: str) -> dict:
    if not gloss:
        return {
            "text": "",
            "vocabHitRate": 0.0,
            "fallback": False,
            "rawG2TText": "",
            "fallbackReason": "",
        }
    result = G2TTranslator().translate(gloss)
    reason = fallback_reason(gloss, result)
    if reason:
        return {
            **result,
            "text": gloss_to_readable_text(gloss),
            "rawG2TText": result.get("text", ""),
            "fallback": True,
            "fallbackReason": reason,
        }
    return {
        **result,
        "rawG2TText": result.get("text", ""),
        "fallback": False,
        "fallbackReason": "",
    }


def translate_video(video_path: str | Path, device: int = 0) -> dict:
    started = time.perf_counter()
    corrnet = run_corrnet(video_path, device=device)
    if corrnet["returncode"] != 0:
        return {
            "status": "failed",
            "gloss": "",
            "result": "",
            "confidence": 0.0,
            "latencyMs": int((time.perf_counter() - started) * 1000),
            "model": "CorrNet + G2T",
            "fallback": False,
            "error": corrnet["stderr"] or corrnet["stdout"],
        }
    gloss = corrnet["gloss"]
    g2t = translate_gloss(gloss)
    return {
        "status": "completed",
        "gloss": gloss,
        "result": g2t["text"],
        "confidence": round(float(g2t.get("vocabHitRate", 0.0)), 4),
        "latencyMs": int((time.perf_counter() - started) * 1000),
        "model": "CorrNet + G2T",
        "fallback": bool(g2t.get("fallback", False)),
        "rawText": g2t.get("rawG2TText", ""),
        "fallbackReason": g2t.get("fallbackReason", ""),
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--gloss")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.gloss:
        print(translate_gloss(args.gloss))
        return 0
    if not args.video:
        parser.error("--video or --gloss is required")
    print(translate_video(args.video, args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
