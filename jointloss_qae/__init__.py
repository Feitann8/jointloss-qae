"""Lightweight prototype: quantum amplitude estimation for two-region joint-loss tail risk."""
from .model import LossModel, build_joint_pmf, joint_exceedance_payoff, layer_payoff, exact_value
from .circuits import build_A, grover_operator, grover_probability
from .mlae import mlae_estimate, simulate_mlae, simulate_monte_carlo, query_count

__all__ = [
    "LossModel", "build_joint_pmf", "joint_exceedance_payoff", "layer_payoff", "exact_value",
    "build_A", "grover_operator", "grover_probability",
    "mlae_estimate", "simulate_mlae", "simulate_monte_carlo", "query_count",
]
