import pandas as pd

from eval.tau_sweep import choose


def test_choose_takes_cheapest_setting_near_the_best_quality():
    curve = pd.DataFrame({"tau": [0.5, 0.9, 0.99], "halt_tau": 0.5,
                          "quality": [0.90, 0.951, 0.952], "cost_usd": [1.0, 2.0, 5.0]})
    assert choose(curve)["tau"] == 0.9
    assert choose(curve, tol=0.0)["tau"] == 0.99
