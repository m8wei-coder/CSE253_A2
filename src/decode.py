import json
from pathlib import Path

import numpy as np

from .data import REST, id_to_pitch


def task1_tokens_to_grid(tokens, id_to_token):
    pitches = [id_to_pitch(t, id_to_token) for t in tokens]
    pitches = [p for p in pitches if p is not None]
    n = len(pitches) // 4 * 4
    if n == 0:
        return np.empty((0, 4), dtype=np.int16)
    return np.array(pitches[:n], dtype=np.int16).reshape(-1, 4)


def task2_tokens_to_grid(soprano_tokens, generated_tokens, id_to_token):
    soprano = [id_to_pitch(t, id_to_token) for t in soprano_tokens]
    soprano = [p for p in soprano if p is not None]
    atb = [id_to_pitch(t, id_to_token) for t in generated_tokens]
    atb = [p for p in atb if p is not None]
    frames = min(len(soprano), len(atb) // 3)
    grid = np.full((frames, 4), REST, dtype=np.int16)
    grid[:, 0] = np.array(soprano[:frames], dtype=np.int16)
    grid[:, 1:] = np.array(atb[: frames * 3], dtype=np.int16).reshape(frames, 3)
    return grid


def grid_to_midi(grid, out_path, bpm=80):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    names = ["Soprano", "Alto", "Tenor", "Bass"]
    programs = [0, 0, 0, 0]
    frame_seconds = 60.0 / bpm / 4.0
    for voice in range(4):
        inst = pretty_midi.Instrument(program=programs[voice], name=names[voice])
        start = None
        prev = None
        for t, pitch in enumerate(list(grid[:, voice]) + [REST]):
            if pitch == prev:
                continue
            if prev is not None and prev != REST and start is not None:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=80,
                        pitch=int(prev),
                        start=start * frame_seconds,
                        end=t * frame_seconds,
                    )
                )
            start = t
            prev = pitch
        pm.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    return pm


def load_id_to_token(vocab_path):
    with open(vocab_path) as f:
        obj = json.load(f)
    return {int(k): v for k, v in obj["id_to_token"].items()}

