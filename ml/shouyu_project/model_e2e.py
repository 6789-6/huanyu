"""End-to-end CSLR model: ResNet18 + TSCM temporal shift + TCN + Transformer + BiLSTM + CTC.

Aligned with SOTA (VAC_CSLR / ResNetT / CT-ATHA / OLMD):
  - ResNet18 end-to-end (NOT frozen) — backbone learns sign-specific features
  - TSCM (Temporal Shift Channel Mixing) — zero-parameter temporal modeling
  - TCN → Transformer → BiLSTM → Multi-level CTC heads
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TemporalShift(nn.Module):
    """TSCM-style temporal channel shifting (ResNetT, 2024).
    Zero parameters — shifts 1/8 channels forward, 1/8 backward in time.
    Provides bidirectional temporal context without 3D convs.
    """

    def __init__(self, n_channels: int, shift_ratio: float = 0.125):
        super().__init__()
        self.n_shift = max(1, int(n_channels * shift_ratio))

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        if T < 2:
            return x
        out = x.clone()
        # Forward shift: earlier channels get info from t+1
        out[:, :T - 1, :self.n_shift] = x[:, 1:, :self.n_shift]
        # Backward shift: later channels get info from t-1
        out[:, 1:, self.n_shift:self.n_shift * 2] = x[:, :T - 1, self.n_shift:self.n_shift * 2]
        return out


class ResNet18TSCM(nn.Module):
    """ResNet18 with TSCM between stages for end-to-end CSLR.

    Outputs per-frame features [B, T, 512] for the temporal model.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None,
        )
        # Split into stages
        self.conv1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 64 channels
        self.layer2 = resnet.layer2   # 128 channels
        self.layer3 = resnet.layer3   # 256 channels
        self.layer4 = resnet.layer4   # 512 channels
        self.avgpool = resnet.avgpool  # AdaptiveAvgPool2d((1, 1))

        # TSCM modules between stages (shifts 1/8 of channels)
        self.tscm1 = TemporalShift(64)
        self.tscm2 = TemporalShift(128)
        self.tscm3 = TemporalShift(256)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T = x.shape[:2]
        # Merge B and T for 2D conv processing
        x = x.view(B * T, x.shape[2], x.shape[3], x.shape[4])

        x = self.conv1(x)       # [B*T, 64, H/2, W/2]
        x = self.maxpool(x)     # [B*T, 64, H/4, W/4]

        x = self.layer1(x)      # [B*T, 64, H/4, W/4]
        x = x.view(B, T, *x.shape[1:])
        x = self.tscm1(x)       # temporal shift
        x = x.view(B * T, *x.shape[2:])

        x = self.layer2(x)      # [B*T, 128, H/8, W/8]
        x = x.view(B, T, *x.shape[1:])
        x = self.tscm2(x)
        x = x.view(B * T, *x.shape[2:])

        x = self.layer3(x)      # [B*T, 256, H/16, W/16]
        x = x.view(B, T, *x.shape[1:])
        x = self.tscm3(x)
        x = x.view(B * T, *x.shape[2:])

        x = self.layer4(x)      # [B*T, 512, H/32, W/32]
        x = self.avgpool(x)     # [B*T, 512, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B*T, 512]
        x = x.view(B, T, 512)   # [B, T, 512]

        return x


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=(kernel_size - 1) * dilation // 2,
                              dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return self.dropout(F.relu(self.norm(x))) + residual


class TCN(nn.Module):
    def __init__(self, d_model, num_layers=3, kernel_size=5, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class E2EModel(nn.Module):
    """End-to-end CSLR: ResNet18+TSCM → TCN → Transformer → BiLSTM → Multi-Level CTC."""

    def __init__(self, d_model=512, n_heads=8, n_tf_layers=3, d_ff=2048,
                 vocab_size=1329, dropout=0.3, blank_idx=0,
                 tcn_layers=3, tcn_kernel=5, lstm_layers=2,
                 pretrained_backbone=True):
        super().__init__()
        self.d_model = d_model
        self.blank_idx = blank_idx

        # Visual backbone
        self.backbone = ResNet18TSCM(pretrained=pretrained_backbone)

        self.pos_encoder = PositionalEncoding(d_model, max_len=300, dropout=dropout)

        # Stage 1: TCN
        self.tcn = TCN(d_model, num_layers=tcn_layers, kernel_size=tcn_kernel, dropout=dropout)

        # Stage 2: Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_tf_layers)

        # Stage 3: BiLSTM
        self.bilstm = nn.LSTM(
            d_model, d_model // 2, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )

        # Multi-level CTC heads
        self.ctc_head_tcn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        self.ctc_head_tf = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        self.ctc_head_lstm = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for module in [self.ctc_head_tcn, self.ctc_head_tf, self.ctc_head_lstm,
                       self.pos_encoder, self.tcn, self.transformer, self.bilstm]:
            for p in module.parameters():
                if p.dim() > 1 and p.requires_grad:
                    nn.init.xavier_uniform_(p)

    def forward(self, frames, mask=None):
        """frames: [B, T, C, H, W] float32 0-1, mask: [B, T] bool."""
        x = self.backbone(frames)  # [B, T, 512]
        x = self.pos_encoder(x)

        src_key_padding_mask = ~mask if mask is not None else None

        tcn_out = self.tcn(x)
        tcn_log_probs = self.ctc_head_tcn(tcn_out).log_softmax(dim=-1)

        tf_out = self.transformer(tcn_out, src_key_padding_mask=src_key_padding_mask)
        tf_log_probs = self.ctc_head_tf(tf_out).log_softmax(dim=-1)

        lstm_out, _ = self.bilstm(tf_out)
        lstm_log_probs = self.ctc_head_lstm(lstm_out).log_softmax(dim=-1)

        return {
            "log_probs": [tcn_log_probs, tf_log_probs, lstm_log_probs],
            "head_weights": [0.3, 0.5, 1.0],
        }

    def forward_final(self, frames, mask=None):
        x = self.backbone(frames)
        x = self.pos_encoder(x)
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.tcn(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x, _ = self.bilstm(x)
        return self.ctc_head_lstm(x).log_softmax(dim=-1)

    def decode(self, log_probs, input_lengths, idx_to_token, blank_idx=0, ignored_token_ids=None):
        max_indices = log_probs.argmax(dim=-1)
        results = []
        ignored = set(ignored_token_ids or [])
        for b in range(len(max_indices)):
            raw = max_indices[b, :input_lengths[b]].tolist()
            decoded = []
            prev = blank_idx
            for token_id in raw:
                if token_id != prev and token_id != blank_idx and token_id not in ignored:
                    decoded.append(idx_to_token.get(token_id, "<unk>"))
                prev = token_id
            results.append(decoded)
        return results
