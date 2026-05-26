from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORRNET_DICT = ROOT / "external" / "corrnet" / "CorrNet" / "preprocess" / "CSL-Daily" / "gloss_dict.npy"
G2T_CKPT = ROOT / "output" / "best_g2t.pt"


def load_corrnet_tokens(path: Path) -> set[str]:
    data = np.load(path, allow_pickle=True).item()
    return {str(v) for v in data.keys()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrnet-dict", default=str(CORRNET_DICT))
    parser.add_argument("--g2t", default=str(G2T_CKPT))
    args = parser.parse_args()
    corrnet = load_corrnet_tokens(Path(args.corrnet_dict))
    ckpt = torch.load(args.g2t, map_location="cpu")
    g2t = set(ckpt["gloss_vocab"].keys())
    special = {"<pad>", "<bos>", "<eos>", "<unk>"}
    corrnet -= special
    g2t -= special
    overlap = corrnet & g2t
    hit_rate = 0.0 if not corrnet else len(overlap) / len(corrnet)
    missing = sorted(corrnet - g2t)[:50]
    print(f"corrnet_tokens={len(corrnet)}")
    print(f"g2t_tokens={len(g2t)}")
    print(f"overlap={len(overlap)}")
    print(f"hit_rate={hit_rate:.4f}")
    print("missing_sample=" + "/".join(missing))
    return 0 if hit_rate >= 0.60 else 2


if __name__ == "__main__":
    raise SystemExit(main())
