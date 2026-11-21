
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class LossModel:
    n_qubits: int = 4                 # qubits per region -> 2**n_qubits log-spaced bins each
    mu: tuple = (np.log(20.0), np.log(15.0))   # lognormal location (loss units, e.g. USD m)
    sigma: tuple = (1.0, 1.1)         # lognormal scale
    rho: float = 0.5                  # copula correlation parameter
    df: float | None = 3.0            # Student-t copula d.o.f.; None -> Gaussian copula
    loss_min: float = 1.0
    loss_max: float = 1000.0          # losses above loss_max fall in the last (overflow) bin
    n_samples: int = 4_000_000        # samples used to build the discretised joint pmf
    seed: int = 12345

    @property
    def n_bins(self) -> int:
        return 2 ** self.n_qubits

    def edges(self) -> np.ndarray:
        """Log-spaced (geometric) bin edges, as proposed in the tail-focused encoding."""
        return np.geomspace(self.loss_min, self.loss_max, self.n_bins + 1)

    def representatives(self) -> np.ndarray:
        """Geometric bin midpoints used as the loss value of each bin."""
        e = self.edges()
        return np.sqrt(e[:-1] * e[1:])


def sample_losses(model: LossModel) -> np.ndarray:
    """Draw (loss_A, loss_B) samples with lognormal marginals and a Gaussian / t copula."""
    rng = np.random.default_rng(model.seed)
    cov = np.array([[1.0, model.rho], [model.rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=model.n_samples)
    if model.df is None:
        u = stats.norm.cdf(z)
    else:
        w = rng.chisquare(model.df, size=model.n_samples) / model.df
        u = stats.t.cdf(z / np.sqrt(w)[:, None], df=model.df)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    x = np.empty_like(u)
    for k in range(2):
        x[:, k] = stats.lognorm.ppf(u[:, k], s=model.sigma[k], scale=np.exp(model.mu[k]))
    return x


def build_joint_pmf(model: LossModel) -> np.ndarray:
    """Discretise the joint loss distribution on log-spaced bins.

    Returns pmf[i, j] = P(region A in bin i, region B in bin j); sums to 1.
    Values below/above the outer edges are clipped into the first/last bin.
    """
    x = sample_losses(model)
    e = model.edges()
    idx = np.clip(np.searchsorted(e, x, side="right") - 1, 0, model.n_bins - 1)
    pmf = np.zeros((model.n_bins, model.n_bins))
    np.add.at(pmf, (idx[:, 0], idx[:, 1]), 1.0)
    return pmf / pmf.sum()


def joint_exceedance_payoff(model: LossModel, trigger_a: float, trigger_b: float) -> np.ndarray:
    """f[i, j] = 1 if both regions' loss bins are at or above their trigger bin, else 0.

    Triggers are snapped to the first bin whose lower edge is >= the trigger,
    so that the discretised event is exactly representable.
    """
    e = model.edges()
    ta = int(np.searchsorted(e[:-1], trigger_a, side="left"))
    tb = int(np.searchsorted(e[:-1], trigger_b, side="left"))
    i, j = np.meshgrid(np.arange(model.n_bins), np.arange(model.n_bins), indexing="ij")
    return ((i >= ta) & (j >= tb)).astype(float)


def layer_payoff(model: LossModel, attach: float, limit: float) -> np.ndarray:
    """Normalised payoff in [0, 1] of an excess-of-loss layer on the combined loss."""
    r = model.representatives()
    total = r[:, None] + r[None, :]
    return np.clip(total - attach, 0.0, limit) / limit


def exact_value(pmf: np.ndarray, payoff: np.ndarray) -> float:
    """Exact (enumeration) expectation of the payoff under the discretised pmf."""
    return float(np.sum(pmf * payoff))
