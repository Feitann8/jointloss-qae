
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector


def build_A(pmf: np.ndarray, payoff: np.ndarray) -> QuantumCircuit:
    k = pmf.shape[0]
    n = int(np.log2(k))
    assert 2 ** n == k and pmf.shape == payoff.shape == (k, k)
    amplitudes = np.sqrt(pmf.T.reshape(-1))            # index s = i + k * j
    angles = 2.0 * np.arcsin(np.sqrt(np.clip(payoff.T.reshape(-1), 0.0, 1.0)))
    qc = QuantumCircuit(2 * n + 1, name="A")
    data = list(range(2 * n))
    qc.append(StatePreparation(amplitudes), data)
    qc.append(UCRYGate(list(angles)), [2 * n] + data)    # target first, then controls
    return qc


def grover_operator(A: QuantumCircuit) -> QuantumCircuit:
    """Q = A S_0 A^dagger S_good (up to an irrelevant global phase)."""
    nq = A.num_qubits
    obj = nq - 1
    qc = QuantumCircuit(nq, name="Q")
    qc.z(obj)                                            # S_good: phase flip when objective = 1
    qc.compose(A.inverse(), inplace=True)
    qc.x(range(nq))                                      # S_0: reflection about |0...0>
    qc.h(obj)
    qc.mcx(list(range(nq - 1)), obj)
    qc.h(obj)
    qc.x(range(nq))
    qc.compose(A, inplace=True)
    return qc


def grover_probability(A: QuantumCircuit, m: int) -> float:
    """Exact P(objective = 1) after Q^m A |0>, by statevector simulation of the circuit."""
    nq = A.num_qubits
    qc = QuantumCircuit(nq)
    qc.compose(A, inplace=True)
    Q = grover_operator(A)
    for _ in range(m):
        qc.compose(Q, inplace=True)
    return float(Statevector.from_instruction(qc).probabilities([nq - 1])[1])
