from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_gloss import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from model import G2TModel


class G2TTranslator:
    def __init__(
        self,
        checkpoint_path: str | Path = ROOT / "output" / "best_g2t.pt",
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        self.ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.gloss_vocab = self.ckpt["gloss_vocab"]
        self.char_vocab = self.ckpt["char_vocab"]
        self.idx_to_char = self.ckpt.get("idx_to_char") or {v: k for k, v in self.char_vocab.items()}
        self.model = G2TModel(
            src_vocab_size=len(self.gloss_vocab),
            tgt_vocab_size=len(self.char_vocab),
            d_model=256,
            n_heads=4,
            n_encoder_layers=3,
            n_decoder_layers=3,
            d_ff=1024,
            max_len=100,
            dropout=0.0,
            pad_idx=self.gloss_vocab[PAD_TOKEN],
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()

    def encode_gloss(self, gloss: str, max_gloss_len: int = 32) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        tokens = [t.strip() for t in gloss.replace(" ", "/").split("/") if t.strip()]
        unk = 0
        ids = [self.gloss_vocab[BOS_TOKEN]]
        for token in tokens:
            if token not in self.gloss_vocab:
                unk += 1
            ids.append(self.gloss_vocab.get(token, self.gloss_vocab[UNK_TOKEN]))
        ids.append(self.gloss_vocab[EOS_TOKEN])
        if len(ids) > max_gloss_len:
            ids = ids[:max_gloss_len]
            length = max_gloss_len
        else:
            length = len(ids)
            ids.extend([self.gloss_vocab[PAD_TOKEN]] * (max_gloss_len - len(ids)))
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.arange(max_gloss_len, device=self.device).unsqueeze(0) < length
        stats = {"tokens": len(tokens), "unknown": unk}
        return src, mask, stats

    @torch.no_grad()
    def translate(self, gloss: str, max_len: int = 80) -> dict:
        src, mask, stats = self.encode_gloss(gloss)
        generated = self.model.generate(
            src,
            mask,
            max_len=max_len,
            bos_idx=self.char_vocab[BOS_TOKEN],
            eos_idx=self.char_vocab[EOS_TOKEN],
        )[0].tolist()
        skip = {self.char_vocab[PAD_TOKEN], self.char_vocab[BOS_TOKEN], self.char_vocab[EOS_TOKEN]}
        text = "".join(self.idx_to_char.get(i, "") for i in generated if i not in skip)
        hit_rate = 1.0 if stats["tokens"] == 0 else (stats["tokens"] - stats["unknown"]) / stats["tokens"]
        return {
            "gloss": gloss,
            "text": text,
            "tokenCount": stats["tokens"],
            "unknownCount": stats["unknown"],
            "vocabHitRate": hit_rate,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloss", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "output" / "best_g2t.pt"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = G2TTranslator(args.checkpoint, args.device).translate(args.gloss)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
