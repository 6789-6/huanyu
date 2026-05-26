from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_decode_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "external" / "corrnet" / "CorrNet" / "utils" / "decode.py"
    spec = importlib.util.spec_from_file_location("corrnet_decode_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_decode_falls_back_to_max_when_ctcdecode_is_unavailable():
    module = load_decode_module()
    decoder = module.Decode({"A": [1], "B": [2]}, num_classes=3, search_mode="beam", blank_id=0)
    logits = torch.tensor([[[0.0, 5.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])
    result = decoder.decode(logits, torch.LongTensor([4]), batch_first=True)
    assert result == [[("A", 0), ("B", 1)]]
