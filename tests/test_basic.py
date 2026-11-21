import numpy as np
import pytest

from jointloss_qae import (LossModel, build_A, build_joint_pmf, exact_value, grover_probability,
                           joint_exceedance_payoff, layer_payoff, mlae_estimate, query_count)


@pytest.fixture(scope="module")
def small():
    model = LossModel(n_qubits=2, n_samples=200_000)
    pmf = build_joint_pmf(model)
    return model, pmf


def test_pmf_is_distribution(small):
    _, pmf = small
    assert pmf.shape == (4, 4)
    assert np.isclose(pmf.sum(), 1.0) and (pmf >= 0).all()


def test_tail_dependence_raises_joint_exceedance():
    base = dict(n_qubits=3, n_samples=300_000)
    p_gauss = exact_value(*(lambda m: (build_joint_pmf(m), joint_exceedance_payoff(m, 100, 100)))(LossModel(df=None, **base)))
    p_t = exact_value(*(lambda m: (build_joint_pmf(m), joint_exceedance_payoff(m, 100, 100)))(LossModel(df=3.0, **base)))
    assert p_t > p_gauss


@pytest.mark.parametrize("payoff_fn", ["exceed", "layer"])
def test_circuit_matches_exact_value_and_grover_theory(small, payoff_fn):
    model, pmf = small
    f = joint_exceedance_payoff(model, 20, 20) if payoff_fn == "exceed" else layer_payoff(model, 50, 200)
    a = exact_value(pmf, f)
    A = build_A(pmf, f)
    theta = np.arcsin(np.sqrt(a))
    for m in range(3):
        assert grover_probability(A, m) == pytest.approx(np.sin((2 * m + 1) * theta) ** 2, abs=1e-9)


def test_mlae_recovers_value_from_expected_counts():
    a, ms, shots = 0.02, (0, 1, 2, 4, 8), 1000
    theta = np.arcsin(np.sqrt(a))
    hits = shots * np.sin((2 * np.array(ms) + 1) * theta) ** 2
    assert mlae_estimate(ms, shots, hits) == pytest.approx(a, rel=1e-3)


def test_query_count():
    assert query_count((0, 1, 2), 10) == 10 * (1 + 3 + 5)
