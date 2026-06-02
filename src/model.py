import math

import torch
import torch.nn as nn


class JSBDecoder(nn.Module):
    """Small Pre-LN causal Transformer decoder shared by both tasks."""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_len=1024,
        tie_weights=True,
        pad_id=0,
    ):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        x = self.blocks(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.head(self.ln_f(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(model, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "ln" in name.lower() or "emb" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


class WarmupCosine:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps=500, total_steps=10000, peak_lr=3e-4, min_lr=3e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.peak_lr * self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

