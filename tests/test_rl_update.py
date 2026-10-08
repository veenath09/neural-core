import numpy as np
import pytest
import torch

from controller.model import CONFIGS, Controller
from data.env import Episode
from data.episodes import generate_skeletons
from data.routerbench import OutcomeTable
from measurement.rl_reward import LAMS, load_cost_ref
from tests.conftest import make_table
from train.baseline import BucketedBaseline
from train.rl import COMPLEXITY_AUX, entropy_beta, rl_loss, rl_update, returns_to_go
from train.rollout import RLTrace, World, rollout


def test_complexity_aux_constant():
    assert COMPLEXITY_AUX == 0.2


def test_returns_to_go():
    reward = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
    got = returns_to_go(reward)
    assert torch.allclose(got, torch.tensor([[6.0, 5.0, 3.0, 0.0]]))


def test_entropy_beta_endpoints_and_clamp():
    assert entropy_beta(0, 100, start=0.01, end=0.001) == pytest.approx(0.01)
    assert entropy_beta(100, 100, start=0.01, end=0.001) == pytest.approx(0.001)
    assert entropy_beta(50, 100, start=0.01, end=0.001) == pytest.approx(0.0055)
    # clamp beyond the range at either end
    assert entropy_beta(-10, 100, start=0.01, end=0.001) == pytest.approx(0.01)
    assert entropy_beta(1000, 100, start=0.01, end=0.001) == pytest.approx(0.001)


def test_baseline_moves_five_percent_and_leaves_other_buckets():
    baseline = BucketedBaseline(n_bands=5, n_lams=5, alpha=0.05)
    assert torch.equal(baseline.table, torch.zeros(5, 5))

    band = torch.tensor([1, 1])
    lam = torch.tensor([2, 2])
    returns = torch.tensor([1.0, 3.0])

    baseline.update(band, lam, returns)

    expected = torch.zeros(5, 5)
    expected[1, 2] = 0.05 * 2.0  # b <- b + alpha*(mean-b) = 0 + 0.05*(2-0)
    assert torch.allclose(baseline.table, expected)

    # state_dict round trip
    sd = baseline.state_dict()
    baseline2 = BucketedBaseline(n_bands=5, n_lams=5, alpha=0.05)
    baseline2.load_state_dict(sd)
    assert torch.allclose(baseline2.table, baseline.table)


def _make_trace(n=2, T=3, device="cpu"):
    logp = torch.zeros(n, T, device=device, requires_grad=True)
    entropy = torch.zeros(n, T, device=device)
    complexity = torch.full((n, T), 0.5, device=device, requires_grad=True)
    reward = torch.zeros(n, T, dtype=torch.float32)
    band = torch.zeros(n, T, dtype=torch.long)
    band_target = torch.full((n, T), 0.5, dtype=torch.float32)
    lam_idx = torch.zeros(n, dtype=torch.long)
    choice = torch.zeros(n, T, dtype=torch.bool)
    valid = torch.zeros(n, T, dtype=torch.bool)
    return RLTrace(
        logp=logp,
        entropy=entropy,
        complexity=complexity,
        reward=reward,
        band=band,
        band_target=band_target,
        lam_idx=lam_idx,
        choice=choice,
        valid=valid,
    )


def test_single_choice_slot_gives_finite_loss():
    trace = _make_trace(n=1, T=1)
    trace.choice[0, 0] = True
    trace.valid[0, 0] = True
    trace.reward[0, 0] = 1.0

    baseline = BucketedBaseline()
    loss, parts = rl_loss(trace, baseline, beta=0.01)

    assert torch.isfinite(loss)
    assert np.isfinite(parts["pg_loss"])
    assert np.isfinite(parts["complexity_mse"])

    loss.backward()
    assert torch.isfinite(trace.logp.grad).all()
    assert torch.isfinite(trace.complexity.grad).all()


def test_single_choice_slot_equal_advantages_also_finite():
    trace = _make_trace(n=2, T=2)
    trace.choice[:, 0] = True
    trace.valid[:, 0] = True
    trace.reward[:, 0] = 1.0  # identical reward -> identical advantage -> zero std

    baseline = BucketedBaseline()
    loss, parts = rl_loss(trace, baseline, beta=0.01)

    assert torch.isfinite(loss)

    loss.backward()
    assert torch.isfinite(trace.logp.grad).all()
    assert torch.isfinite(trace.complexity.grad).all()


def _accelerator_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator_device() is None, reason="no accelerator (cuda/mps) available")
def test_rl_loss_on_accelerator_device():
    # RLTrace in real rollouts keeps logp/entropy/complexity on the model's
    # device (which may be an accelerator) while reward/band/band_target/masks/
    # lam_idx stay on CPU (plan ruling). BucketedBaseline.table is always CPU
    # (it's a small lookup table saved with the checkpoint). rl_loss must not
    # crash when these devices differ, and rl_update must apply its baseline
    # update against the resulting returns without a device mismatch either.
    device = _accelerator_device()
    trace = _make_trace(n=2, T=2, device=device)
    trace.choice[:, 0] = True
    trace.valid[:, 0] = True
    trace.reward[:, 0] = torch.tensor([1.0, 3.0])
    trace.band[:, 0] = torch.tensor([1, 2])
    trace.lam_idx = torch.tensor([0, 1])

    baseline = BucketedBaseline()
    loss, parts = rl_loss(trace, baseline, beta=0.01)

    assert loss.device.type == device
    assert torch.isfinite(loss)
    assert np.isfinite(parts["pg_loss"])

    class _TinyAcceleratorPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1, device=device))

    policy = _TinyAcceleratorPolicy()
    trace2 = _make_trace(n=2, T=2, device=device)
    trace2.logp = (trace2.logp + policy.w).requires_grad_(True)
    trace2.choice[:, 0] = True
    trace2.valid[:, 0] = True
    trace2.reward[:, 0] = torch.tensor([1.0, 3.0])
    trace2.band[:, 0] = torch.tensor([1, 2])
    trace2.lam_idx = torch.tensor([0, 1])

    opt = torch.optim.AdamW(policy.parameters(), lr=0.1)
    stats = rl_update(policy, trace2, baseline, opt, beta=0.01)

    assert np.isfinite(stats["loss"])
    assert baseline.table[1, 0] != 0.0 or baseline.table[2, 1] != 0.0


def test_complexity_term_pulls_toward_band_target():
    # Hand-made trace: zero advantage everywhere (equal rewards, matching baseline
    # value of 0 means the normalised advantage is exactly 0), and a complexity
    # logit far from its band_target. Only the complexity auxiliary can move it.
    raw = torch.nn.Parameter(torch.tensor(3.0))  # sigmoid(3.0) ~= 0.95, far from target 0.1

    class ComplexityOnlyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = raw

        def parameters(self, recurse: bool = True):
            yield self.raw

    def make_trace():
        logp = torch.zeros(1, 1)  # no grad needed: advantage is zero anyway
        entropy = torch.zeros(1, 1)
        complexity = torch.sigmoid(raw).reshape(1, 1)
        reward = torch.tensor([[1.0]])  # matches baseline's initial 0 only after norm below
        band = torch.zeros(1, 1, dtype=torch.long)
        band_target = torch.full((1, 1), 0.1)
        lam_idx = torch.zeros(1, dtype=torch.long)
        choice = torch.ones(1, 1, dtype=torch.bool)
        valid = torch.ones(1, 1, dtype=torch.bool)
        return RLTrace(logp, entropy, complexity, reward, band, band_target, lam_idx, choice, valid)

    policy = ComplexityOnlyPolicy()
    trace = make_trace()
    before = torch.sigmoid(raw).item()

    baseline = BucketedBaseline()
    opt = torch.optim.AdamW(policy.parameters(), lr=0.5)

    rl_update(policy, trace, baseline, opt, beta=0.0)

    after = torch.sigmoid(raw).item()
    target = 0.1

    assert abs(after - target) < abs(before - target)


class _OneParamPolicy(torch.nn.Module):
    """Hand-made one-parameter policy over two episodes: logp_i = c_i * theta for
    distinct coefficients c_i, so d logp_i / d theta = c_i and the two episodes'
    pg gradients on the shared theta don't cancel even under a symmetric reward
    pair. theta starts at 0, where logp is well-defined for any real c_i."""

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(1))
        self.coeffs = torch.tensor([1.0, 0.3])

    def trace(self, rewards: list[float]) -> RLTrace:
        n = len(rewards)
        logp = (self.coeffs * self.theta).reshape(n, 1)
        entropy = torch.zeros(n, 1)
        complexity = torch.full((n, 1), 0.5)
        reward = torch.tensor(rewards).reshape(n, 1)
        band = torch.zeros(n, 1, dtype=torch.long)
        band_target = torch.full((n, 1), 0.5)
        lam_idx = torch.zeros(n, dtype=torch.long)
        choice = torch.ones(n, 1, dtype=torch.bool)
        valid = torch.ones(n, 1, dtype=torch.bool)
        return RLTrace(logp, entropy, complexity, reward, band, band_target, lam_idx, choice, valid)


def test_positive_advantage_raises_logp():
    # Episode 0 gets the high reward and the larger logp coefficient (1.0);
    # episode 1 gets the low reward and the smaller coefficient (0.3). The
    # pg gradient on theta is -mean(A_norm * d logp/d theta) = -(A0*1.0 + A1*0.3)/2,
    # with A0 = +1, A1 = -1 after normalisation: gradient = -(0.7)/2 < 0, so a
    # gradient-descent step raises theta.
    policy = _OneParamPolicy()
    trace = policy.trace(rewards=[5.0, 1.0])
    baseline = BucketedBaseline()
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)

    before = policy.theta.item()
    rl_update(policy, trace, baseline, opt, beta=0.0)

    assert policy.theta.item() > before


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_policy_converges_on_easy_world(seed):
    from tests.test_env import IDS

    table = make_table(n=120, seed=7)
    correct = table.correct.copy()
    correct[:, 0] = True
    easy_table = OutcomeTable(
        prompt_ids=table.prompt_ids,
        step_type=table.step_type,
        correct=correct,
        cost=table.cost,
    )

    world = World(
        easy_table,
        IDS,
        np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4,
        {p: i for i, p in enumerate(easy_table.prompt_ids)},
        {m: i for i, m in enumerate(IDS)},
    )
    sks = generate_skeletons(easy_table, list(easy_table.prompt_ids), n=32, seed=0, split="t")
    episodes = [Episode(sk, easy_table, IDS) for sk in sks]

    torch.manual_seed(seed)
    model = Controller(CONFIGS["full"], torch.randn(120, 384), torch.rand(4, 6), n_types=2)

    cost_ref = load_cost_ref()
    baseline = BucketedBaseline()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    n_updates = 120

    for update in range(n_updates):
        lam_idx = rng.integers(len(LAMS), size=len(episodes))
        out = rollout(
            model,
            episodes,
            world,
            policy="sample",
            rng=rng,
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )
        beta = entropy_beta(update, n_updates)
        rl_update(model, out.rl, baseline, opt, beta=beta)

    model.eval()
    lam_idx = np.full(len(episodes), 3, dtype=int)

    with torch.no_grad():
        out = rollout(
            model,
            episodes,
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    choose_actions = [
        a
        for actions in out.decision_actions
        for a in actions
        if a["phase"] == "choose"
    ]
    assert choose_actions
    share_model_0 = sum(1 for a in choose_actions if a["model"] == 0) / len(choose_actions)
    assert share_model_0 >= 0.9
