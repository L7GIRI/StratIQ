import numpy as np
import pytest

from stratiq.lab import monte_carlo


def test_shuffle_keeps_final_equity_and_uses_only_the_two_hand_drawdowns():
    # orders: +10,-20 -> peak 110, trough 90 -> 20/110 ; -20,+10 -> 20/100
    r = monte_carlo([10.0, -20.0], initial_cash=100.0, n_sims=200, method="shuffle")
    assert np.allclose(r.final_equity, 90.0)
    allowed = np.array([20 / 110, 20 / 100])
    assert all(np.isclose(allowed, d).any() for d in r.max_drawdown)
    assert np.isclose(r.max_drawdown, allowed[0]).any()
    assert np.isclose(r.max_drawdown, allowed[1]).any()


def test_resample_final_equity_varies_and_stays_on_the_grid():
    r = monte_carlo([1.0, -1.0], initial_cash=100.0, n_sims=500, method="resample", seed=3)
    assert set(np.unique(r.final_equity)) <= {98.0, 100.0, 102.0}
    assert r.final_equity.std() > 0


def test_same_seed_same_result_and_different_seed_differs():
    pnls = np.random.default_rng(1).normal(0, 10, 40)
    a = monte_carlo(pnls, n_sims=300, method="resample", seed=5)
    b = monte_carlo(pnls, n_sims=300, method="resample", seed=5)
    c = monte_carlo(pnls, n_sims=300, method="resample", seed=6)
    assert np.array_equal(a.final_equity, b.final_equity)
    assert not np.array_equal(a.final_equity, c.final_equity)


def test_all_wins_have_zero_drawdown_and_zero_loss_probability():
    r = monte_carlo([5.0, 3.0, 4.0], initial_cash=100.0, n_sims=100)
    s = r.summary()
    assert (r.max_drawdown == 0).all()
    assert s["prob_loss"] == 0.0


def test_all_losses_always_end_below_start():
    s = monte_carlo([-5.0, -3.0], initial_cash=100.0, n_sims=100).summary()
    assert s["prob_loss"] == 1.0
    assert s["final_p50"] == 92.0


def test_summary_percentiles_are_ordered():
    pnls = np.random.default_rng(2).normal(1, 10, 60)
    s = monte_carlo(pnls, n_sims=500, method="resample").summary()
    assert s["final_p5"] <= s["final_p50"] <= s["final_p95"]
    assert s["max_dd_p50"] <= s["max_dd_p95"]


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        monte_carlo([])
    with pytest.raises(ValueError):
        monte_carlo([1.0], method="bogus")
    with pytest.raises(ValueError):
        monte_carlo([1.0], n_sims=0)
    with pytest.raises(ValueError):
        monte_carlo([1.0], initial_cash=0)
