from eval.diagnostics import memory_trace, plot_residuals, residual_profile
from tests.test_rollout import tiny_model, world_and_episodes


def test_residual_profile_has_six_l_and_two_h_steps(tmp_path):
    world, eps = world_and_episodes(8)
    profile = residual_profile(tiny_model("full").eval(), [ep.skeleton for ep in eps], world, n=8)
    assert [k for k, _ in profile] == ["L", "L", "L", "H", "L", "L", "L", "H"]
    assert all(v >= 0 for _, v in profile)
    plot_residuals(profile, tmp_path / "r.png")
    assert (tmp_path / "r.png").stat().st_size > 0


def test_memory_trace_lists_writes_of_one_episode():
    world, eps = world_and_episodes(8)
    sks = [ep.skeleton for ep in eps]
    df = memory_trace(tiny_model("full").eval(), sks, world, sks[0].episode_id)
    assert len(df) >= 1 and (df.episode_id == sks[0].episode_id).all()
    assert {"step_index", "verifier_ok", "surprise", "rate", "alpha"} <= set(df.columns)
