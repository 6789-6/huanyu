"""Online-friendly CSLR model: keypoint projection → Transformer → BiLSTM → CTC.

~11M params, ~4GB VRAM, <2ms/frame on GPU. Designed for real-time inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LearnedPositionalEncoding(nn.Module):
    """Learned position embeddings, more flexible than sinusoidal for keypoint data."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class OnlineCSLR(nn.Module):
    """Keypoint-only CSLR model for training and streaming inference.

    Architecture:
      Linear(258→384) → LearnedPE → Transformer×3 → +residual → BiLSTM×2 → CTC
    """

    def __init__(
        self,
        kp_dim: int = 258,
        d_model: int = 384,
        n_heads: int = 6,
        n_tf_layers: int = 3,
        d_ff: int = 1536,
        lstm_hidden: int = 384,
        lstm_layers: int = 2,
        vocab_size: int = 3846,
        blank_idx: int = 4,
        dropout: float = 0.15,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.blank_idx = blank_idx

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(kp_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder (Pre-LN for stability)
        tf_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=n_tf_layers)

        # Layer norm after residual add
        self.tf_norm = nn.LayerNorm(d_model)

        # BiLSTM for local sequence smoothing
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Final projection before CTC
        self.out_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # CTC classification head
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if "weight" in name and "norm" not in name.lower():
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Re-init positional encoding
        nn.init.normal_(self.pos_enc.pe, mean=0.0, std=0.02)

    def forward(self, keypoints: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Full forward pass returning CTC log-probabilities.

        Args:
            keypoints: (B, T, 258) float32 keypoint sequence.
            valid_mask: (B, T) bool, True for valid frames.

        Returns:
            log_probs: (B, T, vocab_size) log-softmax probabilities.
        """
        # Input projection
        x = self.input_proj(keypoints)              # (B, T, 256)
        residual = x

        # Positional encoding
        x = self.pos_enc(x)                         # (B, T, 256)

        # Transformer
        src_mask = None
        if valid_mask is not None:
            src_mask = ~valid_mask
        x = self.transformer(x, src_key_padding_mask=src_mask)  # (B, T, 256)

        # Residual connection: projector output + transformer output
        x = self.tf_norm(x + residual)              # (B, T, 256)

        # BiLSTM. Pack padded batches so invalid tail frames do not leak into
        # valid timesteps through the backward LSTM direction.
        if valid_mask is not None:
            lengths = valid_mask.sum(dim=1).clamp_min(1).detach().cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.bilstm(packed)
            x, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=keypoints.size(1),
            )
        else:
            x, _ = self.bilstm(x)                   # (B, T, 512)

        # Output projection + CTC head
        x = self.out_proj(x)                        # (B, T, 256)
        log_probs = self.ctc_head(x).log_softmax(dim=-1)  # (B, T, V)

        return log_probs

    @torch.no_grad()
    def decode(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
        idx_to_token: dict[int, str],
        ignored_ids: set[int] | None = None,
    ) -> list[list[str]]:
        """Greedy CTC decode: collapse repeats, remove blank & ignored tokens.

        Args:
            log_probs: (B, T, V) log probabilities.
            input_lengths: (B,) valid frame counts per sample.
            idx_to_token: int → token string mapping.
            ignored_ids: token ids to skip (pad, bos, eos, blank).

        Returns:
            List of decoded token lists, one per batch element.
        """
        max_indices = log_probs.argmax(dim=-1)       # (B, T)
        ignored = ignored_ids or {self.blank_idx}
        results: list[list[str]] = []

        for b in range(len(max_indices)):
            raw = max_indices[b, :input_lengths[b]].tolist()
            decoded: list[str] = []
            prev = self.blank_idx
            for tid in raw:
                if tid != prev and tid not in ignored:
                    token = idx_to_token.get(tid, "<unk>")
                    decoded.append(token)
                prev = tid
            results.append(decoded)

        return results
