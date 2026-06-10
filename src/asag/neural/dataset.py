"""Cross-encoder text-pair dataset over the Phase 2A encoder view.

Each row becomes a ``(premise, student)`` pair, tokenized by the model-native
tokenizer at train time (no pre-materialized ``input_ids`` — Phase 2A's contract).
``premise`` is the reference answer when present, else the question; when both
exist we prepend the question so the model sees what was asked. Targets are a
plain float vector (regression / ordinal rank / class code) assembled by the
trainer, so this class stays task-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_premise(df: pd.DataFrame) -> list[str]:
    """premise = question + reference when both present, else whichever exists."""
    q = df.get("question_enc", pd.Series([""] * len(df))).fillna("").astype(str)
    r = df.get("reference_answer_enc", pd.Series([""] * len(df))).fillna("").astype(str)
    out = []
    for qi, ri in zip(q, r):
        qi, ri = qi.strip(), ri.strip()
        if ri and qi:
            out.append(f"{qi} [SEP] {ri}")
        else:
            out.append(ri or qi)
    return out


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, targets: np.ndarray, tokenizer, max_len: int):
        self.premise = build_premise(df)
        self.student = df["student_answer_enc"].fillna("").astype(str).tolist()
        self.targets = torch.as_tensor(np.asarray(targets, dtype="float32"))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.student)

    def __getitem__(self, i: int) -> dict:
        enc = self.tok(
            self.premise[i], self.student[i],
            truncation=True, max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["target"] = self.targets[i]
        return item
