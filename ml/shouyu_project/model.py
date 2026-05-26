"""Transformer model for Sign Language Translation."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class SLTModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 512,
        vocab_size: int = 2000,
        max_len: int = 500,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

        # Input projection: keypoints -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,        # [B, T_src, input_dim]
        tgt: torch.Tensor,        # [B, T_tgt] token ids
        src_mask: torch.Tensor,   # [B, T_src] True = keep
        tgt_mask: torch.Tensor,   # [B, T_tgt] True = keep
    ) -> torch.Tensor:
        # Encode source keypoints
        src_emb = self.input_proj(src)  # [B, T_src, d_model]
        src_emb = self.pos_encoder(src_emb)

        # Create padding mask for encoder: True = IGNORE
        src_key_padding_mask = ~src_mask  # [B, T_src]

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # Decode
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        # Causal mask for decoder self-attention
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )

        tgt_key_padding_mask = ~tgt_mask

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.output_head(out)  # [B, T_tgt, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int = 50,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ) -> torch.Tensor:
        """Greedy decode."""
        self.eval()
        B = src.size(0)
        device = src.device

        src_emb = self.input_proj(src)
        src_emb = self.pos_encoder(src_emb)
        src_key_padding_mask = ~src_mask
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_decoder(tgt_emb)

            tgt_len = generated.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1,
            )

            out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask,
                              memory_key_padding_mask=src_key_padding_mask)
            logits = self.output_head(out[:, -1:])  # [B, 1, vocab]
            next_token = logits.argmax(dim=-1)  # [B, 1]

            # Mark finished sequences
            finished |= (next_token.squeeze(1) == eos_idx)

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated


class G2TModel(nn.Module):
    """Gloss2Text: gloss tokens -> Chinese characters."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 100,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.output_head = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_tokens, tgt_tokens, src_mask, tgt_mask):
        src_emb = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_key_padding_mask = ~src_mask
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        tgt_len = tgt_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_key_padding_mask = ~tgt_mask

        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_head(out)

    @torch.no_grad()
    def generate(self, src_tokens, src_mask, max_len=50, bos_idx=1, eos_idx=2):
        self.eval()
        B = src_tokens.size(0)
        device = src_tokens.device

        src_emb = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_key_padding_mask = ~src_mask
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.tgt_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_decoder(tgt_emb)

            tgt_len = generated.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1,
            )

            out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask,
                              memory_key_padding_mask=src_key_padding_mask)
            logits = self.output_head(out[:, -1:])
            next_token = logits.argmax(dim=-1)

            finished |= (next_token.squeeze(1) == eos_idx)
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated
