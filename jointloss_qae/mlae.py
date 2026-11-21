
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

DEFAULT_SCHEDULE = (0, 1, 2, 4, 8, 16, 32)


def query_count(ms, shots: int) -> int:
    """Oracle-call (A application) count: each shot of Q^m A|0> costs 2m + 1 calls to A."""
    return int(shots * sum(2 * m + 1 for m in ms))


def ideal_probs(theta: float, ms) -> np.ndarray:
    return np.sin((2 * np.asarray(ms) + 1) * theta) ** 2


def mlae_estimate(ms, shots: int, hits: np.ndarray) -> float:
    """Return the MLE of the target value a = sin^2(theta) from Grover-power measurement counts."""
    ms = np.asarray(ms)
    hits = np.asarray(hits, dtype=float)

    def nll(theta):
        p = np.clip(np.sin((2 * ms + 1) * theta) ** 2, 1e-12, 1 - 1e-12)
        return -np.sum(hits * np.log(p) + (shots - hits) * np.log(1 - p))

    # Vectorised scan over the whole (0, pi/2) range, then local refinement.
    grid = np.linspace(1e-6, np.pi / 2 - 1e-6, 200001)
    p = np.clip(np.sin(np.outer(grid, 2 * ms + 1)) ** 2, 1e-12, 1 - 1e-12)
    vals = -(np.log(p) @ hits + np.log(1 - p) @ (shots - hits))
    t0 = grid[int(np.argmin(vals))]
    step = grid[1] - grid[0]
    res = minimize_scalar(nll, bounds=(max(t0 - 2 * step, 1e-9), min(t0 + 2 * step, np.pi / 2)),
                          method="bounded", options={"xatol": 1e-13})
    return float(np.sin(res.x) ** 2)


def simulate_mlae(true_value: float, ms, shots: int, trials: int, rng, depol: float = 0.0,
                  noise_floor: float = 0.5) -> np.ndarray:
    """Simulate `trials` independent MLAE runs on an ideal (or toy-noisy) device.

    depol: toy global-depolarising probability per oracle call. With probability
    1 - (1 - depol)^(2m+1) the measured objective bit is replaced by a coin with bias
    `noise_floor` (the maximally mixed state). This is a TOY model, not a hardware simulation.
    """
    theta = np.arcsin(np.sqrt(true_value))
    p_ideal = ideal_probs(theta, ms)
    keep = (1.0 - depol) ** (2 * np.asarray(ms) + 1)
    p_obs = keep * p_ideal + (1.0 - keep) * noise_floor
    est = np.empty(trials)
    for t in range(trials):
        hits = rng.binomial(shots, p_obs)
        est[t] = mlae_estimate(ms, shots, hits)
    return est


def simulate_monte_carlo(pmf: np.ndarray, payoff: np.ndarray, n_samples: int, trials: int, rng) -> np.ndarray:
    """Classical Monte Carlo on the SAME discretised distribution, one sample per oracle call."""
    p = pmf.reshape(-1)
    f = payoff.reshape(-1)
    est = np.empty(trials)
    for t in range(trials):
        counts = rng.multinomial(n_samples, p)
        est[t] = np.dot(counts, f) / n_samples
    return est
