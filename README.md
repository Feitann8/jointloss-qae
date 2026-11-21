# jointloss-qae

Lightweight prototype for the ADB Quantum Innovation Challenge proposal
*Quantum Amplitude Estimation for Joint-Loss Tail Risk in Disaster Risk Financing*.

It estimates two quantities on a **two-region joint-loss distribution with tunable tail dependence**
using quantum amplitude estimation (maximum-likelihood variant, Suzuki et al. 2020), and compares
the result with classical Monte Carlo at the same number of oracle calls:

1. **Joint exceedance probability**: P(loss in region A >= trigger *and* loss in region B >= trigger).
2. **Expected layer payout**: normalised excess-of-loss layer on the combined loss.

> **Status: proof-of-concept on synthetic data.** All loss parameters are illustrative placeholders,
> not calibrated to any real country, catalogue or ADB data. Results show how the method behaves on a
> small, noiseless simulation; they do **not** demonstrate a practical quantum advantage.

## Contents

| Path | What it is |
| --- | --- |
| `jointloss_qae/model.py` | Synthetic joint-loss model: lognormal marginals, Gaussian or Student-t copula, log-spaced bins |
| `jointloss_qae/circuits.py` | Qiskit circuits: distribution loading, payoff oracle, Grover operator |
| `jointloss_qae/mlae.py` | Maximum-likelihood amplitude estimation, classical Monte Carlo baseline, toy noise model |
| `run_experiments.py` | Runs all experiments and writes `results/` |
| `tests/test_basic.py` | Unit tests (circuits vs theory, MLAE, dependence effect) |
| `results/` | Sample outputs from a full run (CSV, PNG, JSON, environment) |
| `requirements.txt` | Pinned dependencies |

## Setup and execution

Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                  # 6 tests, ~2 s
python run_experiments.py --quick    # smoke test, well under a minute
python run_experiments.py            # full run, a few minutes on one CPU core; overwrites results/
```

Everything runs on a CPU; no GPU, quantum hardware or API keys are needed. Seeds are fixed, so reruns
reproduce the numbers below.

## Method in brief

1. **Scenario model.** Two lognormal loss marginals joined by a Gaussian or Student-t copula. The t
   copula's degrees of freedom is the tail-dependence parameter (lower = stronger tail dependence).
2. **Encoding.** Each region's loss is discretised on 2^4 = 16 **log-spaced bins** (tail-focused), giving
   a 16x16 joint probability table, loaded onto 8 qubits with `StatePreparation`.
3. **Payoff oracle.** A uniformly controlled RY rotation writes the payoff into an objective qubit so
   that P(objective = 1) equals the target expectation exactly. This lookup-table oracle is fine at
   this size but **does not scale** (2^(2n) angles); realistic models need comparator/adder circuits.
4. **Estimation.** Grover powers m in {0, 1, 2, 4, ...}, 100 shots each, combined by maximum likelihood.
   Classical Monte Carlo samples the *same* discretised distribution, one sample per oracle call.

## Simulator and hardware details

- **No quantum hardware was used.**
- Circuits (9 qubits for the main case) are built in Qiskit and checked against theory with the exact
  `qiskit.quantum_info.Statevector` simulator (`results/circuit_verification.csv`, max abs. difference
  about 1e-13 for Grover powers 0 to 2).
- Convergence runs use the closed form P_m = sin^2((2m+1)theta), exact for ideal circuits, plus binomial
  shot sampling. Powers above 2 are therefore **not** run as circuits.
- Noise study: a **toy** model in which each oracle call depolarises the output with a fixed
  probability. It is not a hardware or Aer noise simulation.
- Transpiled resources (basis `cx`, `u`; `results/circuit_resources.csv`):

| Qubits per region | Total qubits | A: CX / depth | Q (one Grover step): CX / depth |
| --- | --- | --- | --- |
| 2 | 5 | 27 / 54 | 90 / 172 |
| 3 | 7 | 121 / 194 | 364 / 517 |
| 4 | 9 | 503 / 1006 | 1232 / 2197 |

Circuit cost roughly quadruples per extra qubit per region because of the lookup-table oracle.

## Sample results (full run, `results/`)

Base case: t copula, df = 3, rho = 0.5; triggers 100 in both regions; layer attaches at 200 with
limit 400 on the combined loss (illustrative units); 200 trials per point.

**Convergence** (`convergence.png`, `convergence.csv`), relative RMSE at 13,300 oracle calls:

| Target | Exact value | Classical MC | MLAE (ideal) |
| --- | --- | --- | --- |
| Joint exceedance | 0.0126 | 0.083 | 0.012 |
| Layer payout | 0.0110 | 0.062 | 0.039 |

For joint exceedance the MLAE error falls faster than 1/sqrt(queries), as expected. For the layer
payout the advantage is smaller and the error stops improving at the largest budget (0.034 at 6,800
calls, 0.039 at 13,300), which we have not investigated; it may reflect occasional wrong-mode
likelihood maxima. Treat the layer-payout result as inconclusive.

**Tail-dependence what-if** (`dependence_sweep.png`, `dependence_sweep.csv`), exact joint exceedance:

| Dependence | Joint exceedance | Layer payout |
| --- | --- | --- |
| Gaussian rho = 0 (independent) | 0.0013 | 0.0072 |
| Gaussian rho = 0.5 | 0.0076 | 0.0105 |
| t, df = 10, rho = 0.5 | 0.0093 | 0.0107 |
| t, df = 3, rho = 0.5 | 0.0126 | 0.0110 |
| t, df = 2, rho = 0.5 | 0.0143 | 0.0111 |

At equal correlation, a heavier-tailed copula (df = 3) gives about 1.65 times the Gaussian joint
exceedance probability, while the combined-loss layer payout barely moves. The amplitude-estimation
estimates agree with the exact values to within about 1 to 3 percent of the value.

**Toy noise study** (`noise_toy.png`, `noise_toy.csv`): at 6,800 calls, MLAE relative RMSE rises from
0.022 (no noise) to 0.051 at 1e-2 depolarising probability per oracle call, versus 0.109 for classical
Monte Carlo. This must not be read as hardware-ready: one Grover step at 4 qubits per region is about
1,200 CX gates, so 1e-2 per oracle call corresponds to roughly 1e-5 error per CX, far better than
current devices.

## Limitations

- Synthetic, uncalibrated data; no real hazard catalogue, vulnerability model or ADB data is used yet.
- The discretised distribution is small enough that classical enumeration is exact and instant. The
  comparison counts oracle calls only; it says nothing about wall-clock cost or about a realistic
  black-box catastrophe simulator.
- Loading an arbitrary distribution and compiling a full simulator into a circuit are open problems and
  are not solved here.
- The quantum-inspired (tensor-network) comparison planned in the proposal is not implemented.
- Overflow losses above the top bin are clipped into the last bin, which understates the extreme tail.

## Disclosure of AI tools used

This prototype was drafted with the help of **Claude (Anthropic)** in a chat session.

## References

Montanaro (2015); Woerner and Egger (2019); Suzuki et al. (2020), *Amplitude estimation without phase
estimation*; Milek (2020), *Quantum implementation of risk analysis-relevant copulas*;
Kirke (2026), arXiv:2603.15664.
