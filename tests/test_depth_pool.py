import numpy as np
import pandas as pd

from data.depth_pool import band_bucket, build_table
from data.routerbench import OutcomeTable


def test_band_bucket_edges():
    assert band_bucket(np.array([0, 1, 2, 3, 6, 7, 12, 13, 19])).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_build_table_orders_options_cheapest_first_and_keeps_band():
    wide = pd.DataFrame({
        "prompt_id": ["p0", "p1", "p2"], "split": "train", "source": ["mmlu", "gsm8k", "mmlu"],
        "correct_big@high": [True, True, False], "cost_big@high": [0.01, 0.02, 0.01], "think_big@high": [100, 200, 0],
        "correct_small@off": [True, False, False], "cost_small@off": [0.001, 0.001, 0.001], "think_small@off": 0,
    })
    options = [{"name": "big@high", "provider": "x", "depth": "high"}, {"name": "small@off", "provider": "y", "depth": "off"}]
    texts = pd.Series({"p0": "a", "p1": "b", "p2": "c"})
    table, cards = build_table(wide, options, texts)
    assert [c.model_id for c in cards] == ["small@off", "big@high"] and [c.tier for c in cards] == [0, 1]
    assert table["correct_0"].tolist() == [True, False, False] and table["band"].tolist() == [0, 1, 1]
    assert table["step_type"].tolist() == [0, 4, 0]
    assert cards[1].capability[-2] == 1.0 and cards[0].capability[-2] == 0.0  # depth level / 4
    t = OutcomeTable.from_frame(table)
    assert t.band.tolist() == [0, 1, 1] and t.n_models == 2
