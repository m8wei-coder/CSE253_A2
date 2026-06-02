import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import WarmupCosine, make_optimizer


def batch_loss(model, batch, device, pad_id=0):
    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    logits = model(input_ids, attention_mask=attention_mask)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view_as(targets)
    denom = loss_mask.sum().clamp_min(1.0)
    return (loss * loss_mask).sum() / denom


@torch.no_grad()
def evaluate_nll(model, loader, device, pad_id=0):
    model.eval()
    total_loss, total_tokens = 0.0, 0.0
    for batch in loader:
        loss = batch_loss(model, batch, device, pad_id=pad_id)
        ntok = batch["loss_mask"].sum().item()
        total_loss += loss.item() * ntok
        total_tokens += ntok
    nll = total_loss / max(total_tokens, 1.0)
    return nll, math.exp(min(nll, 50))


def train_model(
    model,
    train_loader,
    valid_loader,
    device,
    ckpt_path,
    epochs=80,
    patience=10,
    peak_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=500,
    grad_clip=1.0,
    pad_id=0,
):
    optimizer = make_optimizer(model, lr=peak_lr)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = WarmupCosine(optimizer, warmup_steps, total_steps, peak_lr, min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_nll": [], "valid_nll": [], "valid_ppl": [], "lr": [], "seconds": []}
    best, bad_epochs = float("inf"), 0
    start = time.time()
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running, tokens = 0.0, 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = batch_loss(model, batch, device, pad_id=pad_id)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr = scheduler.step()
            ntok = batch["loss_mask"].sum().item()
            running += loss.item() * ntok
            tokens += ntok
        train_nll = running / max(tokens, 1.0)
        valid_nll, valid_ppl = evaluate_nll(model, valid_loader, device, pad_id=pad_id)
        history["train_nll"].append(train_nll)
        history["valid_nll"].append(valid_nll)
        history["valid_ppl"].append(valid_ppl)
        history["lr"].append(lr)
        history["seconds"].append(time.time() - start)
        print(f"epoch {epoch:03d} train_nll={train_nll:.4f} valid_nll={valid_nll:.4f} ppl={valid_ppl:.2f}")
        if valid_nll < best:
            best = valid_nll
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "history": history, "best_nll": best}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return history

