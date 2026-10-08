import numpy as np
import pytest
import torch

from controller.model import CONFIGS, Controller
from data.env import Episode
from data.episodes import Skeleton, generate_skeletons
from tests.conftest import make_table
from tests.test_env import IDS
from train.rollout import World, rollout, tau_choice


def world_and_episodes(n=12):
    table = make_table(n=120, seed=7)
    world = World(table, IDS, np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4,
                  {p: i for i, p in enumerate(table.prompt_ids)}, {m: i for i, m in enumerate(IDS)})
    sks = generate_skeletons(table, list(table.prompt_ids), n=n, seed=0, split="t")
    return world, [Episode(sk, table, IDS) for sk in sks]


def tiny_model(name="full"):
    torch.manual_seed(0)
    return Controller(CONFIGS[name], torch.randn(120, 384), torch.rand(4, 6), n_types=2)


def test_tau_choice():
    p = np.array([0.2, 0.6, 0.9, 0.95])
    assert tau_choice(p, [0, 1, 2, 3], 0.5) == 1
    assert tau_choice(p, [0, 1, 2, 3], 0.99) == 3
    assert tau_choice(p, [2, 3], 0.5) == 2


@pytest.mark.parametrize("name", ["full", "gate_stateless", "memory_off", "context_free"])
def test_greedy_rollout_finishes_every_episode(name):
    world, eps = world_and_episodes()
    with torch.no_grad():
        out = rollout(tiny_model(name).eval(), eps, world)
    assert [r.episode_id for r in out.results] == [ep.skeleton.episode_id for ep in eps]
    assert all(0 <= r.quality <= 1 and r.cost_usd >= 0 for r in out.results) and out.loss is None


def test_mixture_loss_trains_the_gate_through_the_episode():
    world, eps = world_and_episodes()
    m = tiny_model("full")
    out = rollout(m, eps, world, policy="mixture", rng=np.random.default_rng(0), compute_loss=True, bp_steps=3)
    assert torch.isfinite(out.loss) and set(out.parts) == {"model", "halt", "complexity"}
    out.loss.backward()
    assert m.memory.gate.readout.weight.grad.abs().sum() > 0


def test_episode_result_does_not_depend_on_batch_mates():
    world, eps = world_and_episodes()
    m = tiny_model("full").eval()
    with torch.no_grad():
        together = rollout(m, eps, world).results
        alone = [rollout(m, [ep], world).results[0] for ep in eps]
    assert [(r.quality, round(r.cost_usd, 12)) for r in together] == [(r.quality, round(r.cost_usd, 12)) for r in alone]


def test_no_affordable_stronger_model_means_stop():
    world, _ = world_and_episodes()
    table = world.table
    sk = Skeleton("tight", "random", tuple(table.prompt_ids[:3]), float(table.cost[:3, 0].sum()) * 1.01)
    m = tiny_model("full").eval()
    with torch.no_grad():
        nn_bias = m.heads.halt.bias
        nn_bias.fill_(50.0)  # always wants to escalate
        out = rollout(m, [Episode(sk, table, IDS)], world, tau=0.0)  # tau 0: cheapest model each step
    assert out.results[0].cost_usd <= sk.budget_usd


def test_record_logs_memory_writes_and_residuals():
    world, eps = world_and_episodes(4)
    with torch.no_grad():
        out = rollout(tiny_model("full").eval(), eps, world, record=True)
    assert out.log and {"episode_id", "step_index", "rate", "surprise", "alpha", "verifier_ok"} <= set(out.log[0])
    assert out.residuals and len(out.residuals[0]) == 8
