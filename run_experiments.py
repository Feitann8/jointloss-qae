
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import scipy
from qiskit import transpile

from jointloss_qae import (LossModel, build_A, build_joint_pmf, exact_value, grover_operator,
                           grover_probability, joint_exceedance_payoff, layer_payoff, query_count,
                           simulate_mlae, simulate_monte_carlo)

OUT = Path("results")
TRIGGER = 100.0                # both regions' loss trigger (illustrative units)
ATTACH, LIMIT = 200.0, 400.0   # excess-of-loss layer on the combined loss
SHOTS = 100                    # shots per Grover power
SEED = 2026


def rel_rmse(est, truth):
    return float(np.sqrt(np.mean((np.asarray(est) - truth) ** 2)) / truth)


def targets(model, pmf):
    return {
        "joint_exceedance": joint_exceedance_payoff(model, TRIGGER, TRIGGER),
        "layer_payout": layer_payoff(model, ATTACH, LIMIT),
    }


def write_environment(args):
    lines = [
        f"python {sys.version.split()[0]}", f"qiskit {qiskit.__version__}",
        f"numpy {np.__version__}", f"scipy {scipy.__version__}", f"matplotlib {matplotlib.__version__}",
        f"platform {platform.platform()}", f"machine {platform.machine()}",
        "simulator: qiskit.quantum_info.Statevector (exact, noiseless) for circuit checks;",
        "           closed form sin^2((2m+1)*theta) + binomial shot sampling for the convergence runs;",
        "           toy per-oracle-call depolarising model for the noise study (NOT a hardware simulation)",
        f"quick_mode {args.quick}",
    ]
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n")


def circuit_checks(n_samples):
    """(a) Verify Qiskit circuits reproduce theory; (b) report circuit resources."""
    rows, resources = [], []
    for n in (2, 3, 4):
        model = LossModel(n_qubits=n, n_samples=n_samples)
        pmf = build_joint_pmf(model)
        f = joint_exceedance_payoff(model, TRIGGER, TRIGGER)
        a = exact_value(pmf, f)
        A = build_A(pmf, f)
        th = np.arcsin(np.sqrt(a))
        for m in (0, 1, 2):
            p_circ = grover_probability(A, m)
            rows.append({"qubits_per_region": n, "m": m, "exact_value": a, "circuit_P1": p_circ,
                         "theory_sin2": float(np.sin((2 * m + 1) * th) ** 2),
                         "abs_diff": abs(p_circ - np.sin((2 * m + 1) * th) ** 2)})
        t_A = transpile(A, basis_gates=["cx", "u"], optimization_level=1)
        t_Q = transpile(grover_operator(A), basis_gates=["cx", "u"], optimization_level=1)
        resources.append({"qubits_per_region": n, "total_qubits": A.num_qubits,
                          "A_cx": int(t_A.count_ops().get("cx", 0)), "A_depth": t_A.depth(),
                          "Q_cx": int(t_Q.count_ops().get("cx", 0)), "Q_depth": t_Q.depth()})
    write_csv(OUT / "circuit_verification.csv", rows)
    write_csv(OUT / "circuit_resources.csv", resources)
    return rows, resources


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


def exp_convergence(model, pmf, trials):
    rng = np.random.default_rng(SEED)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, f) in zip(axes, targets(model, pmf).items()):
        a = exact_value(pmf, f)
        q_list, e_q, e_c = [], [], []
        for top in range(0, 7):                      # max Grover power 2**top - 1 .. 64
            ms = (0,) + tuple(2 ** k for k in range(top))
            q = query_count(ms, SHOTS)
            est_q = simulate_mlae(a, ms, SHOTS, trials, rng)
            est_c = simulate_monte_carlo(pmf, f, q, trials, rng)
            rows.append({"target": name, "exact_value": a, "max_power": max(ms), "queries": q,
                         "mlae_rel_rmse": rel_rmse(est_q, a), "mc_rel_rmse": rel_rmse(est_c, a)})
            q_list.append(q); e_q.append(rel_rmse(est_q, a)); e_c.append(rel_rmse(est_c, a))
        ax.loglog(q_list, e_c, "o-", label="classical Monte Carlo")
        ax.loglog(q_list, e_q, "s-", label="MLAE (ideal circuits)")
        ax.set_title(f"{name.replace('_', ' ')}  (value = {a:.4f})")
        ax.set_xlabel("oracle calls (queries)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("relative RMSE")
    axes[0].legend()
    fig.suptitle("Error vs oracle calls on the same discretised joint distribution (noiseless)")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)
    write_csv(OUT / "convergence.csv", rows)
    return rows


def exp_dependence(n_samples, trials):
    rng = np.random.default_rng(SEED + 1)
    configs = [("Gaussian rho=0 (independent)", None, 0.0), ("Gaussian rho=0.5", None, 0.5),
               ("t df=30 rho=0.5", 30.0, 0.5), ("t df=10 rho=0.5", 10.0, 0.5),
               ("t df=5 rho=0.5", 5.0, 0.5), ("t df=3 rho=0.5", 3.0, 0.5), ("t df=2 rho=0.5", 2.0, 0.5)]
    ms = (0, 1, 2, 4, 8, 16, 32)
    rows = []
    for label, df, rho in configs:
        model = LossModel(n_qubits=4, df=df, rho=rho, n_samples=n_samples)
        pmf = build_joint_pmf(model)
        for name, f in targets(model, pmf).items():
            a = exact_value(pmf, f)
            est = simulate_mlae(a, ms, SHOTS, trials, rng)
            rows.append({"dependence": label, "target": name, "exact_value": a,
                         "mlae_mean": float(est.mean()), "mlae_sd": float(est.std(ddof=1))})
    write_csv(OUT / "dependence_sweep.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ("joint_exceedance", "layer_payout")):
        sub = [r for r in rows if r["target"] == name]
        x = np.arange(len(sub))
        ax.errorbar(x, [r["mlae_mean"] for r in sub], yerr=[r["mlae_sd"] for r in sub], fmt="s",
                    capsize=3, label="MLAE estimate (mean ± sd)")
        ax.plot(x, [r["exact_value"] for r in sub], "k_", markersize=18, label="exact (discretised)")
        ax.set_xticks(x)
        ax.set_xticklabels([r["dependence"] for r in sub], rotation=40, ha="right", fontsize=8)
        ax.set_title(name.replace("_", " "))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("value")
    axes[0].legend(fontsize=8)
    fig.suptitle("What-if: stronger tail dependence raises joint-loss risk (illustrative parameters)")
    fig.tight_layout()
    fig.savefig(OUT / "dependence_sweep.png", dpi=150)
    plt.close(fig)
    return rows


def exp_noise(model, pmf, trials):
    rng = np.random.default_rng(SEED + 2)
    f = joint_exceedance_payoff(model, TRIGGER, TRIGGER)
    a = exact_value(pmf, f)
    ms = (0, 1, 2, 4, 8, 16)
    q = query_count(ms, SHOTS)
    mc = rel_rmse(simulate_monte_carlo(pmf, f, q, trials, rng), a)
    rows = []
    for lam in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        est = simulate_mlae(a, ms, SHOTS, trials, rng, depol=lam)
        rows.append({"depol_per_oracle_call": lam, "queries": q, "mlae_rel_rmse": rel_rmse(est, a),
                     "mlae_rel_bias": float((est.mean() - a) / a), "mc_rel_rmse": mc})
    write_csv(OUT / "noise_toy.csv", rows)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    lam = [r["depol_per_oracle_call"] for r in rows]
    ax.semilogy(range(len(lam)), [r["mlae_rel_rmse"] for r in rows], "s-", label="MLAE (toy noise)")
    ax.axhline(mc, color="C0", ls="--", label="classical MC, same queries")
    ax.set_xticks(range(len(lam)))
    ax.set_xticklabels([f"{x:g}" for x in lam])
    ax.set_xlabel("toy depolarising prob. per oracle call")
    ax.set_ylabel("relative RMSE")
    ax.set_title("Where the advantage disappears (toy model)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "noise_toy.png", dpi=150)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    n_samples = 400_000 if args.quick else 4_000_000
    trials = 30 if args.quick else 200
    write_environment(args)

    print("1/4 circuit verification and resources ...")
    ver, res = circuit_checks(min(n_samples, 400_000))
    print("2/4 convergence ...")
    model = LossModel(n_qubits=4, df=3.0, rho=0.5, n_samples=n_samples)
    pmf = build_joint_pmf(model)
    conv = exp_convergence(model, pmf, trials)
    print("3/4 dependence sweep ...")
    dep = exp_dependence(n_samples, max(trials // 2, 20))
    print("4/4 toy noise study ...")
    noise = exp_noise(model, pmf, trials)

    summary = {"max_circuit_vs_theory_abs_diff": max(r["abs_diff"] for r in ver),
               "circuit_resources": res, "convergence": conv, "dependence": dep, "noise_toy": noise}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done; outputs in", OUT.resolve())


if __name__ == "__main__":
    main()
