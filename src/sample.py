import torch
import torch.nn.functional as F


def top_p_filter(logits, top_p=0.9):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits.masked_fill(remove, float("-inf")))
    return filtered


@torch.no_grad()
def sample(model, prefix, max_new_tokens, temperature=0.9, top_p=0.9, eos_id=2, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ctx = ids[:, -model.max_len :]
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-6)
        logits = top_p_filter(logits, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == eos_id:
            break
    return ids[0].tolist()

