"""Real-time CSLR inference: camera → MediaPipe → model → streaming gloss output.

Usage:
    python realtime_inference.py                          # use best_online_by_wer.pt
    python realtime_inference.py --checkpoint <path>     # custom checkpoint
    python realtime_inference.py --camera 1              # use camera index 1
    python realtime_inference.py --no-boundary           # disable boundary detector
"""
import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# MediaPipe (IMAGE mode — same API as extract_keypoints.py)
os.environ["GLOG_minloglevel"] = "2"

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as mp_base_options
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] mediapipe not installed. Run: pip install mediapipe")

from boundary_detector import BoundaryDetector
from model_online import OnlineCSLR
from dataset_keypoint import BOS_TOKEN, EOS_TOKEN, KP_DIM, PAD_TOKEN, BLANK_TOKEN, UNK_TOKEN

# ─── Constants ─────────────────────────────────────────────────────────────
SHOUYU_ROOT = Path(__file__).resolve().parent
MODEL_PATH = SHOUYU_ROOT / "models" / "holistic_landmarker.task"
OUTPUT_DIR = SHOUYU_ROOT / "output"
WINDOW_SIZE = 64       # frames in sliding window (~4.3 s at 15 fps)
STRIDE = 16            # inference every N frames (~1 Hz)
MIN_CONFIDENCE = 0.5

FRAME_SKIP = 2         # 30 fps → 15 fps (match training)


def ctc_ignored_ids(vocab: dict[str, int]) -> set[int]:
    """CTC ids that should never be emitted as user-visible gloss tokens."""
    return {
        vocab[tok]
        for tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BLANK_TOKEN)
        if tok in vocab
    }


def load_online_weights(model: OnlineCSLR, state: dict):
    """Load online CSLR weights strictly so checkpoint/model drift fails fast."""
    weights = state.get("ema_model_state_dict") or state["model_state_dict"]
    return model.load_state_dict(weights, strict=True)


# ─── Keypoint extraction (per-frame) ──────────────────────────────────────

def extract_one_frame(landmarker, frame_bgr: np.ndarray):
    """Extract a flattened 258-dim keypoint vector from one BGR frame.

    Returns:
        np.ndarray (258,) float32, or None if detection fails.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    pose = np.zeros((33, 4), dtype=np.float32)
    lhand = np.zeros((21, 3), dtype=np.float32)
    rhand = np.zeros((21, 3), dtype=np.float32)

    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks):
            pose[i] = [lm.x, lm.y, lm.z, lm.visibility]
    if result.left_hand_landmarks:
        for i, lm in enumerate(result.left_hand_landmarks):
            lhand[i] = [lm.x, lm.y, lm.z]
    if result.right_hand_landmarks:
        for i, lm in enumerate(result.right_hand_landmarks):
            rhand[i] = [lm.x, lm.y, lm.z]

    return np.concatenate([
        pose.reshape(-1),
        lhand.reshape(-1),
        rhand.reshape(-1),
    ]).astype(np.float32)


# ─── Streaming CSLR engine ────────────────────────────────────────────────

class StreamingCSLR:
    """Sliding-window streaming sign language recogniser.

    Buffer accumulates keypoint frames. Every STRIDE frames the model runs
    CTC greedy decode on the current window.  A simple diff against the
    previous window's output yields newly recognised glosses.
    """

    def __init__(
        self,
        model: OnlineCSLR,
        idx_to_token: dict[int, str],
        blank_idx: int = 4,
        ignored_ids: set[int] | None = None,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
    ):
        self.model = model
        self.idx_to_token = idx_to_token
        self.blank_idx = blank_idx
        self.ignored_ids = set(ignored_ids or {blank_idx})
        self.ignored_ids.add(blank_idx)
        self.window_size = window_size
        self.stride = stride
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        self.buffer: deque = deque(maxlen=window_size)
        self.frame_count = 0
        self.last_decoded: list[str] = []
        self.new_glosses: list[str] = []

    def _needs_inference(self) -> bool:
        return (self.frame_count > 0 and
                len(self.buffer) == self.window_size and
                self.frame_count % self.stride == 0)

    def _ctc_greedy(self, log_probs: torch.Tensor) -> list[str]:
        """Greedy CTC decode: collapse repeats, remove blanks."""
        ids = log_probs.argmax(dim=-1).tolist()
        decoded: list[str] = []
        prev = self.blank_idx
        for tid in ids:
            if tid != prev and tid not in self.ignored_ids:
                decoded.append(self.idx_to_token.get(tid, "<unk>"))
            prev = tid
        return decoded

    @staticmethod
    def _diff(current: list[str], previous: list[str]) -> list[str]:
        """Return suffix of *current* not overlapping with *previous*.

        Finds the longest suffix of `previous` that equals a prefix of
        `current`, then returns the remainder of `current`.
        """
        if not previous:
            return current
        best = 0
        max_k = min(len(previous), len(current))
        for k in range(max_k, 0, -1):
            if previous[-k:] == current[:k]:
                best = k
                break
        return current[best:]

    def add_frame(self, kp: np.ndarray) -> list[str] | None:
        """Feed one keypoint vector.

        Returns a list of newly recognised gloss strings when inference
        fires, otherwise None.
        """
        self.buffer.append(torch.from_numpy(kp))
        self.frame_count += 1

        if not self._needs_inference():
            return None

        seq = torch.stack(list(self.buffer)).unsqueeze(0).to(self.device)  # (1, T, 258)

        with torch.no_grad():
            log_probs = self.model(seq)        # (1, T, V)
            decoded = self._ctc_greedy(log_probs[0])

        new = self._diff(decoded, self.last_decoded)
        self.last_decoded = decoded

        if new:
            self.new_glosses = new
            return new
        return None

    def accumulated_text(self) -> str:
        """Full accumulated gloss sequence so far."""
        # Reconstruct from last_decoded: the complete current hypothesis
        return "/".join(self.last_decoded) if self.last_decoded else ""


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time CSLR inference")
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint path (default: best_online_by_wer.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--no-boundary", action="store_true",
                        help="Disable boundary detector")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE,
                        help="Sliding window size in frames")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="Inference stride in frames")
    args = parser.parse_args()

    if not HAS_MEDIAPIPE:
        print("MediaPipe required.  Install:  pip install mediapipe")
        sys.exit(1)

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = OUTPUT_DIR / "best_online_by_wer.pt"
        if not ckpt.exists():
            ckpt = OUTPUT_DIR / "best_online.pt"
    ckpt = Path(ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu")
    vocab: dict = state["vocab"]
    idx_to_token: dict = state.get("idx_to_token", {v: k for k, v in vocab.items()})

    model = OnlineCSLR(
        kp_dim=KP_DIM, vocab_size=len(vocab),
        blank_idx=vocab.get(BLANK_TOKEN, 4),
    )
    # Load EMA weights if available, else raw
    load_online_weights(model, state)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"Vocab: {len(vocab)} tokens")

    # ── MediaPipe ───────────────────────────────────────────────────────────
    print(f"Loading MediaPipe from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"MediaPipe model not found: {MODEL_PATH}")
        print("Download from: https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker")
        sys.exit(1)

    mp_options = vision.HolisticLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=MIN_CONFIDENCE,
        min_pose_detection_confidence=MIN_CONFIDENCE,
        min_hand_landmarks_confidence=MIN_CONFIDENCE,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(mp_options)

    # ── Camera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera index {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Streaming engine ────────────────────────────────────────────────────
    streamer = StreamingCSLR(
        model, idx_to_token, blank_idx=vocab.get(BLANK_TOKEN, 4),
        ignored_ids=ctc_ignored_ids(vocab),
        window_size=args.window, stride=args.stride,
    )
    boundary = None if args.no_boundary else BoundaryDetector(still_threshold=12,
                                                                velocity_threshold=0.008)
    accumulated: list[str] = []
    frame_idx = 0
    fps_t0 = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("\n[Real-time CSLR running]  Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Frame skip (30 → 15 fps to match training)
        if frame_idx % FRAME_SKIP != 0:
            # Still show the frame
            cv2.imshow("Real-Time CSLR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── MediaPipe keypoints ─────────────────────────────────────────
        kp = extract_one_frame(landmarker, frame)

        # ── Boundary detection (optional) ───────────────────────────────
        at_boundary = boundary.update(kp) if boundary else False

        # ── Model inference ─────────────────────────────────────────────
        new_glosses = streamer.add_frame(kp)

        if new_glosses:
            accumulated.extend(new_glosses)
            # Reset boundary detector after a recognised sign
            if boundary:
                boundary.reset()
            print(f"  ✓ {new_glosses}")

        # ── Render ──────────────────────────────────────────────────────
        # FPS counter
        fps_counter += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            current_fps = fps_counter / (now - fps_t0)
            fps_counter = 0
            fps_t0 = now

        # Draw recognised text
        text = " / ".join(accumulated[-6:]) if accumulated else "(listening...)"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # Draw latest output
        latest = streamer.new_glosses
        if latest:
            cv2.putText(frame, f"NEW: {' / '.join(latest)}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw FPS and boundary indicator
        cv2.putText(frame, f"FPS: {current_fps:.0f}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if boundary and at_boundary:
            cv2.putText(frame, "BOUNDARY", (frame.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Real-Time CSLR", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
