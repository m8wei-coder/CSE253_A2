import math

import numpy as np


VOICE_RANGES = np.array([[60, 81], [55, 74], [48, 69], [40, 64]])


def nll(model, dataloader, device):
    """Return validation NLL and perplexity for a masked next-token loader."""
    from .train import evaluate_nll

    return evaluate_nll(model, dataloader, device)


def _valid_pitches(grid):
    arr = np.asarray(grid)
    return arr[arr >= 0]


def pitch_class_hist(grids):
    hist = np.ones(12, dtype=np.float64) * 1e-8
    for grid in grids:
        pitches = _valid_pitches(grid)
        if len(pitches):
            hist += np.bincount(pitches % 12, minlength=12)
    return hist / hist.sum()


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-8
    q = np.asarray(q, dtype=np.float64) + 1e-8
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def overlap_area(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    return float(np.minimum(p, q).sum())


def interval_hist(grids, max_interval=24):
    hist = np.ones(2 * max_interval + 1, dtype=np.float64) * 1e-8
    for grid in grids:
        grid = np.asarray(grid)
        for v in range(grid.shape[1]):
            pitches = grid[:, v]
            mask = (pitches[1:] >= 0) & (pitches[:-1] >= 0)
            diffs = np.clip(pitches[1:][mask] - pitches[:-1][mask], -max_interval, max_interval)
            hist += np.bincount(diffs + max_interval, minlength=len(hist))
    return hist / hist.sum()


def interval_kl(grids_gen, grids_real):
    return kl_divergence(interval_hist(grids_gen), interval_hist(grids_real))


def note_duration_hist(grids, max_len=64):
    hist = np.ones(max_len, dtype=np.float64) * 1e-8
    for grid in grids:
        grid = np.asarray(grid)
        for v in range(grid.shape[1]):
            prev = None
            run = 0
            for p in list(grid[:, v]) + [None]:
                if p == prev:
                    run += 1
                else:
                    if prev is not None and prev >= 0:
                        hist[min(run, max_len) - 1] += 1
                    prev = p
                    run = 1
    return hist / hist.sum()


def note_duration_kl(grids_gen, grids_real):
    return kl_divergence(note_duration_hist(grids_gen), note_duration_hist(grids_real))


def polyphony_hist(grids):
    hist = np.ones(5, dtype=np.float64) * 1e-8
    for grid in grids:
        counts = (np.asarray(grid) >= 0).sum(axis=1)
        hist += np.bincount(np.clip(counts, 0, 4), minlength=5)
    return hist / hist.sum()


def voice_range_adherence(grids):
    ok, total = 0, 0
    for grid in grids:
        grid = np.asarray(grid)
        for v in range(min(4, grid.shape[1])):
            pitches = grid[:, v]
            mask = pitches >= 0
            total += int(mask.sum())
            lo, hi = VOICE_RANGES[v]
            ok += int(((pitches[mask] >= lo) & (pitches[mask] <= hi)).sum())
    return ok / max(total, 1)


def parallel_intervals(grid):
    grid = np.asarray(grid)
    count = 0
    t_len = grid.shape[0]
    for t in range(1, t_len):
        for v1, v2 in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            if grid[t, v1] < 0 or grid[t, v2] < 0 or grid[t - 1, v1] < 0 or grid[t - 1, v2] < 0:
                continue
            if grid[t, v1] == grid[t - 1, v1] or grid[t, v2] == grid[t - 1, v2]:
                continue
            i_prev = abs(grid[t - 1, v1] - grid[t - 1, v2]) % 12
            i_curr = abs(grid[t, v1] - grid[t, v2]) % 12
            if i_prev == i_curr and i_prev in (0, 7):
                count += 1
    return count / max(t_len - 1, 1) * 100


def parallel_intervals_mean(grids):
    vals = [parallel_intervals(g) for g in grids if len(g)]
    return float(np.mean(vals)) if vals else 0.0


def n_gram_repetition(token_seqs, n=4):
    vals = []
    for seq in token_seqs:
        grams = [tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1))]
        vals.append(0.0 if not grams else 1.0 - len(set(grams)) / len(grams))
    return float(np.mean(vals)) if vals else 0.0


def frame_accuracy(pred_grids, gt_grids):
    ok, total = 0, 0
    for pred, gt in zip(pred_grids, gt_grids):
        n = min(len(pred), len(gt))
        if n:
            ok += int((pred[:n, 1:] == gt[:n, 1:]).all(axis=1).sum())
            total += n
    return ok / max(total, 1)


def per_voice_accuracy(pred_grids, gt_grids):
    out = {}
    for name, v in zip(["A", "T", "B"], [1, 2, 3]):
        ok, total = 0, 0
        for pred, gt in zip(pred_grids, gt_grids):
            n = min(len(pred), len(gt))
            ok += int((pred[:n, v] == gt[:n, v]).sum())
            total += n
        out[name] = ok / max(total, 1)
    return out


def pitch_class_accuracy(pred_grids, gt_grids):
    ok, total = 0, 0
    for pred, gt in zip(pred_grids, gt_grids):
        n = min(len(pred), len(gt))
        if n:
            ok += int(((pred[:n, 1:] % 12) == (gt[:n, 1:] % 12)).sum())
            total += 3 * n
    return ok / max(total, 1)


def edit_distance(pred_tokens, gt_tokens):
    m, n = len(pred_tokens), len(gt_tokens)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if pred_tokens[i - 1] == gt_tokens[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / max(n, 1)
