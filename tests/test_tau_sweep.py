import pandas as pd

from eval.tau_sweep import choose


def curve():
    return pd.DataFrame({"tau": [0.5, 0.9, 0.95, 0.99], "halt_tau": 0.5,
                         "quality": [0.90, 0.951, 0.952, 0.96], "cost_usd": [1.0, 2.0, 3.0, 9.0]})


def test_choose_takes_cheapest_setting_near_the_best_quality_under_the_cost_cap():
    assert choose(curve(), cost_cap=5.0)["tau"] == 0.9  # 0.951 is within 0.002 of 0.952 and cheaper
    assert choose(curve(), cost_cap=5.0, tol=0.0)["tau"] == 0.95
    assert choose(curve())["tau"] == 0.99  # no cap: the best quality


def test_choose_falls_back_to_cheapest_when_nothing_is_affordable():
    assert choose(curve(), cost_cap=0.5)["tau"] == 0.5
