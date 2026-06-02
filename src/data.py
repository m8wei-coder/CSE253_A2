import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<BAR>", "<FERMATA>", "<REST>", "<SEP>"]
PITCH_MIN = 21
PITCH_MAX = 108
REST = -1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab():
    tokens = SPECIAL_TOKENS + [f"P_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token


def save_vocab(path):
    token_to_id, id_to_token = build_vocab()
    with open(path, "w") as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token}, f, indent=2)
    return token_to_id, id_to_token


def pitch_to_id(pitch, token_to_id):
    if pitch == REST or pitch is None:
        return token_to_id["<REST>"]
    pitch = int(np.clip(round(pitch), PITCH_MIN, PITCH_MAX))
    return token_to_id[f"P_{pitch}"]


def id_to_pitch(idx, id_to_token):
    tok = id_to_token[int(idx)] if isinstance(id_to_token, dict) else id_to_token[int(idx)]
    if tok == "<REST>":
        return REST
    if tok.startswith("P_"):
        return int(tok.split("_", 1)[1])
    return None


def load_bach_chorales():
    """Load available music21 Bach scores; parse failures are skipped."""
    from music21 import corpus

    scores = []
    for entry in corpus.getComposer("bach"):
        try:
            score = corpus.parse(entry)
            scores.append({"id": str(entry), "score": score})
        except Exception as exc:
            print(f"warning: skipped {entry}: {exc}")
    return scores


def _part_to_pitch_array(part, step, n_steps):
    """Return active MIDI pitch per 16th-note frame for one voice."""
    out = np.full(n_steps, REST, dtype=np.int16)
    events = sorted(part.flatten().notesAndRests, key=lambda n: float(n.offset))
    for n in events:
        if n.isRest:
            continue
        start = float(n.offset)
        end = start + float(n.quarterLength)
        i0 = int(round(start / step))
        i1 = int(round(end / step))
        i0, i1 = max(0, i0), min(n_steps, i1)
        if i1 <= i0:
            continue
        if n.isChord:
            pitch = int(round(n.pitches[-1].midi))
        else:
            pitch = int(round(n.pitch.midi))
        out[i0:i1] = pitch
    return out


def chorale_to_grid(score, step=0.25):
    """Quantize a four-part score to a (T, 4) SATB pitch grid."""
    parts = list(score.parts)
    if len(parts) < 4:
        return None
    parts = parts[:4]
    end_time = max(float(p.flatten().highestTime) for p in parts)
    n_steps = int(math.ceil(end_time / step))
    grid = np.full((n_steps, 4), REST, dtype=np.int16)
    for v, part in enumerate(parts):
        grid[:, v] = _part_to_pitch_array(part, step, n_steps)
    return grid


def grid_to_task1_tokens(grid, token_to_id, max_len=1024, bars_every=16):
    ids = [token_to_id["<BOS>"]]
    for t, frame in enumerate(grid):
        if t % bars_every == 0:
            ids.append(token_to_id["<BAR>"])
        ids.extend(pitch_to_id(p, token_to_id) for p in frame)
    ids.append(token_to_id["<EOS>"])
    return ids[:max_len]


def grid_to_task2_pair(grid, token_to_id, max_len=1024):
    soprano = [pitch_to_id(p, token_to_id) for p in grid[:, 0]]
    atb = []
    for frame in grid:
        atb.extend(pitch_to_id(p, token_to_id) for p in frame[1:])
    max_s = max(1, min(len(soprano), (max_len - 3) // 4))
    soprano = soprano[:max_s]
    atb = atb[: max(0, max_len - len(soprano) - 3)]
    return soprano, atb


def split_indices(n, seed=42):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    return idx[:n_train], idx[n_train : n_train + n_valid], idx[n_train + n_valid :]


def save_preprocessed(data_dir="data", max_len=1024, seed=42):
    """Build vocab and split artifacts from music21 Bach chorales."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    token_to_id, id_to_token = save_vocab(Path(data_dir) / "vocab.json")
    rows, failures = [], []
    for item in load_bach_chorales():
        grid = chorale_to_grid(item["score"])
        if grid is None or grid.shape[1] != 4:
            failures.append(item["id"])
            continue
        rows.append({"id": item["id"], "score": item["score"], "grid": grid})
    train_idx, valid_idx, test_idx = split_indices(len(rows), seed=seed)
    splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    stats = {
        "n_chorales": len(rows),
        "n_skipped": len(failures),
        "split_method": "seed=42 random 80/10/10 fallback",
        "vocab_size": len(token_to_id),
        "max_len": max_len,
        "task1_lengths": [len(grid_to_task1_tokens(r["grid"], token_to_id, max_len=10**9)) for r in rows],
    }
    stats["truncated_task1"] = int(sum(x > max_len for x in stats["task1_lengths"]))
    with open(Path(data_dir) / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    for split, ids in splits.items():
        task1 = [grid_to_task1_tokens(rows[i]["grid"], token_to_id, max_len=max_len) for i in ids]
        task2 = [grid_to_task2_pair(rows[i]["grid"], token_to_id, max_len=max_len) for i in ids]
        with open(Path(data_dir) / f"jsb_task1_{split}.pkl", "wb") as f:
            pickle.dump(task1, f)
        with open(Path(data_dir) / f"jsb_task2_{split}.pkl", "wb") as f:
            pickle.dump(task2, f)
    return rows, stats, token_to_id, id_to_token


class TokenDataset(Dataset):
    def __init__(
        self,
        examples,
        task="task1",
        pad_id=0,
        bos_id=1,
        sep_id=6,
        eos_id=2,
        max_len=1024,
        augment_transpose=False,
        transpose_range=5,
    ):
        self.examples = examples
        self.task = task
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.sep_id = sep_id
        self.eos_id = eos_id
        self.max_len = max_len
        self.augment_transpose = augment_transpose
        self.transpose_range = transpose_range

    def _transpose_ids(self, ids, shift):
        if shift == 0:
            return list(ids)
        out = []
        first_pitch = len(SPECIAL_TOKENS)
        for idx in ids:
            if first_pitch <= idx < first_pitch + (PITCH_MAX - PITCH_MIN + 1):
                pitch = PITCH_MIN + (idx - first_pitch) + shift
                if PITCH_MIN <= pitch <= PITCH_MAX:
                    out.append(first_pitch + (pitch - PITCH_MIN))
                else:
                    out.append(idx)
            else:
                out.append(idx)
        return out

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        shift = random.randint(-self.transpose_range, self.transpose_range) if self.augment_transpose else 0
        if self.task == "task1":
            seq = self._transpose_ids(ex, shift)[: self.max_len]
            loss_mask = [1] * len(seq)
        else:
            soprano, atb = ex
            soprano = self._transpose_ids(soprano, shift)
            atb = self._transpose_ids(atb, shift)
            seq = [self.bos_id] + soprano + [self.sep_id] + atb + [self.eos_id]
            seq = seq[: self.max_len]
            prefix_len = min(len(soprano) + 2, len(seq))
            loss_mask = [0] * prefix_len + [1] * (len(seq) - prefix_len)
        return {"input_ids": seq[:-1], "targets": seq[1:], "loss_mask": loss_mask[1:]}


def collate_batch(batch, pad_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, targets, loss_mask, attention_mask = [], [], [], []
    for item in batch:
        n = len(item["input_ids"])
        pad = max_len - n
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        targets.append(item["targets"] + [pad_id] * pad)
        loss_mask.append(item["loss_mask"] + [0] * pad)
        attention_mask.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
