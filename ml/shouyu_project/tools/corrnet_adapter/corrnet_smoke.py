from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

SHOUYU_ROOT = Path(__file__).resolve().parents[2]
CORRNET_ROOT = SHOUYU_ROOT / "external" / "corrnet" / "CorrNet"
DEFAULT_WEIGHT = CORRNET_ROOT / "pretrained" / "dev_30.60_CSL-Daily.pt"


def parse_recognized_sents(stdout: str) -> list[str]:
    match = re.search(r"output glosses\s*:\s*(.+)", stdout)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError):
        return []
    if not parsed:
        return []
    first = parsed[0]
    tokens = []
    for item in first:
        if isinstance(item, (list, tuple)) and item:
            tokens.append(str(item[0]))
        else:
            tokens.append(str(item))
    return tokens


def run_corrnet(
    video_path: str | Path,
    model_path: str | Path = DEFAULT_WEIGHT,
    device: int = 0,
) -> dict:
    video_path = Path(video_path).resolve()
    model_path = Path(model_path).resolve()
    cmd = [
        sys.executable,
        "test_one_video.py",
        "--model_path",
        str(model_path),
        "--video_path",
        str(video_path),
        "--language",
        "csl",
        "--device",
        str(device),
    ]
    proc = subprocess.run(cmd, cwd=CORRNET_ROOT, text=True, capture_output=True)
    tokens = parse_recognized_sents(proc.stdout)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": f"cmd_video_path={video_path}\ncmd_model_path={model_path}\n{proc.stderr}",
        "tokens": tokens,
        "gloss": "/".join(tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=str(DEFAULT_WEIGHT))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    result = run_corrnet(args.video, args.model, args.device)
    print("returncode:", result["returncode"])
    print("gloss:", result["gloss"])
    if result["stdout"]:
        print("stdout:", result["stdout"])
    if result["stderr"]:
        print("stderr:", result["stderr"])
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
