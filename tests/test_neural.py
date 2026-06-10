"""Phase 2G — neural-head unit tests.

Pure logic on hand-built logits/labels: the CORAL/CORN losses, the rank-monotone
decoders, and the LabelSpace mapping. The transformer itself is never built here
(that needs a download + minutes of CPU); those paths are exercised by the smoke
run. ``importorskip`` keeps the suite green when torch is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from asag.neural import heads


def test_levels_from_ranks_is_step_monotone():
    y = torch.tensor([0, 1, 3])
    lv = heads.levels_from_ranks(y, num_classes=4)        # K-1 = 3 columns
    assert lv.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 1]]


def test_coral_and_corn_decode_perfect_logits():
    # large +/- logits that decode to ranks 0,1,2,3 (K=4 -> 3 tasks)
    big = 12.0
    logits = torch.tensor([
        [-big, -big, -big],   # rank 0
        [+big, -big, -big],   # rank 1
        [+big, +big, -big],   # rank 2
        [+big, +big, +big],   # rank 3
    ])
    assert heads.coral_predict(logits).tolist() == [0, 1, 2, 3]
    assert heads.corn_predict(logits).tolist() == [0, 1, 2, 3]


def test_corn_predict_respects_conditional_chain():
    # a later task firing while an earlier one is off must NOT inflate the rank
    logits = torch.tensor([[-5.0, 5.0, 5.0]])             # P(y>0) tiny -> rank 0
    assert heads.corn_predict(logits).tolist() == [0]


def test_losses_are_lower_for_correct_logits():
    y = torch.tensor([0, 2, 3])
    good = torch.tensor([[-8.0, -8, -8], [8, 8, -8], [8, 8, 8]])
    bad = -good
    for name in ("coral", "corn"):
        assert heads.ordinal_loss(name, good, y, 4) < heads.ordinal_loss(name, bad, y, 4)


def test_corn_loss_ignores_empty_conditional_set():
    # all-zero labels: every conditional task past 0 is empty -> finite, no nan
    y = torch.tensor([0, 0, 0])
    logits = torch.zeros(3, 3)
    loss = heads.corn_loss(logits, y, 4)
    assert torch.isfinite(loss)


def test_labelspace_ordinal_roundtrip():
    from asag.neural.trainer import LabelSpace
    from asag.models.tasks import get_spec

    df = pd.DataFrame({"score": [0, 2, 3, 2, 0], "label": [""] * 5})
    ls = LabelSpace(get_spec("mindreading"), df)
    assert ls.num_classes == 3                     # scores {0,2,3} -> ranks {0,1,2}
    ranks = ls.targets(df)
    assert ranks.tolist() == [0, 1, 2, 1, 0]
    back = ls.to_metric_space(np.array([0, 1, 2]))
    assert back.tolist() == [0, 2, 3]              # rank -> original score value
