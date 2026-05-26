"""Rule-based sign boundary detector using hand motion velocity.

A boundary is detected when hands are relatively still for a minimum number
of consecutive frames — suggesting a pause between signs/words.
"""
import numpy as np

POSE_DIM = 33 * 4    # 132
HAND_DIM = 21 * 3    # 63
HAND_START = POSE_DIM  # hands start at index 132
HAND_END = POSE_DIM + 2 * HAND_DIM  # hands end at index 258


class BoundaryDetector:
    """Detect sign boundaries from hand velocity.

    Attributes:
        still_threshold: number of consecutive still frames to trigger a boundary.
        velocity_threshold: mean hand-coordinate change below which a frame
            is considered "still" (in normalised coordinate space).
        still_counter: current count of consecutive still frames.
        prev_hand: hand keypoints from the previous frame (or None).
    """

    def __init__(
        self,
        still_threshold: int = 12,
        velocity_threshold: float = 0.008,
    ):
        self.still_threshold = still_threshold
        self.velocity_threshold = velocity_threshold
        self.still_counter = 0
        self.prev_hand: np.ndarray | None = None

    def update(self, kp_frame: np.ndarray) -> bool:
        """Process one keypoint frame, return True if a boundary is detected.

        Args:
            kp_frame: (258,) float32 keypoint vector for the current frame.

        Returns:
            True when hands have been still for still_threshold consecutive frames.
        """
        hand = kp_frame[HAND_START:HAND_END]
        if not self._has_detected_hand(hand):
            self.still_counter = 0
            self.prev_hand = None
            return False

        if self.prev_hand is None:
            self.prev_hand = hand.copy()
            return False

        # Mean absolute hand-coordinate change
        velocity = float(np.abs(hand - self.prev_hand).mean())
        self.prev_hand = hand.copy()

        if velocity < self.velocity_threshold:
            self.still_counter += 1
        else:
            # Fast decay: reset immediately (hands are moving)
            self.still_counter = 0

        return self.still_counter >= self.still_threshold

    def reset(self):
        """Reset state (e.g., after a recognised sign)."""
        self.still_counter = 0
        self.prev_hand = None

    @staticmethod
    def _has_detected_hand(hand: np.ndarray) -> bool:
        """All-zero hand blocks are MediaPipe missing-detection sentinels."""
        return bool(np.any(hand != 0.0))
