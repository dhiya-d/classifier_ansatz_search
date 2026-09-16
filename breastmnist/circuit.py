# ----------------------------------------------------------------
# circuit.py  —  Gate-application logic and QNode construction
#                (BreastMNIST binary, VARIABLE-LENGTH genotypes)
#
# ENCODING — picked at build time from C.PCA_COMPONENTS
# ─────────────────────────────────────────────────────────────────
#   PCA_COMPONENTS == N_QUBITS      → single encoding, 8 features → 8 qubits
#       qml.AngleEmbedding(inputs, wires=range(8), rotation="Y")
#
#   PCA_COMPONENTS == 2 * N_QUBITS  → dense encoding, 16 features → 8 qubits
#       qml.AngleEmbedding(inputs[:8],   wires=range(8), rotation="X")
#       qml.AngleEmbedding(inputs[8:16], wires=range(8), rotation="Y")
#
#   In the dense case qubit i is rotated about X by feature i, then
#   about Y by feature i+8 — doubling the information entering the
#   circuit while keeping register width, and therefore simulation
#   cost, at 8 qubits. Both halves arrive pre-scaled to [0, π].
#
# MEASUREMENT — first ONE surviving qubit, 2 probabilities → 2 classes
# ─────────────────────────────────────────────────────────────────
#   p(|0⟩) → class 0  (malignant)
#   p(|1⟩) → class 1  (normal, benign)
#
#   Binary classification only needs one qubit's basis-state
#   probabilities (N_READOUT_QUBITS=1 in config.py) — unlike the
#   4-class RetinaMNIST version, which measures 2 qubits for 4 joint
#   probabilities. get_gate_applications()/build_qcnn_qnode() are
#   otherwise unchanged: both already read N_READOUT_QUBITS from
#   config rather than hard-coding a qubit count.
#
# This file is AGNOSTIC to genotype length — it walks whatever
# sequence of Qcycle / Qmask layers it is handed and measures
# whichever qubit(s) are still alive at the end.
# ----------------------------------------------------------------

import pennylane as qml
import torch
from copy import deepcopy

from hierarqcal import Qcycle, Qmask, Qunmask

import config as C
from config import N_QUBITS, N_READOUT_QUBITS


def get_gate_applications(motif, n_qubits: int = N_QUBITS):
    """
    Walk a Qmotifs object of ANY length and return:

    applications : list of (gate_fn, [qubit_indices], n_symbols)
    remaining    : list of qubit indices still active after all layers,
                   in circuit execution order

    The first N_READOUT_QUBITS elements of `remaining` are measured.
    """
    applications = []
    Q_avail = list(range(n_qubits))

    for sub_motif in motif:

        # ── Convolutional layer (Qcycle) ───────────────────────
        if isinstance(sub_motif, Qcycle):
            if sub_motif.mapping is None:
                continue
            tmp = deepcopy(sub_motif)
            tmp.mapping = None
            tmp(Q_avail)
            edges   = tmp.E
            gate_fn = sub_motif.mapping.function
            n_sym   = sub_motif.mapping.n_symbols
            for edge in (edges or []):
                applications.append((gate_fn, list(edge), n_sym))

        # ── Pooling layer (Qmask) ──────────────────────────────
        elif isinstance(sub_motif, Qmask):
            Q_before = list(Q_avail)
            tmp = deepcopy(sub_motif)
            tmp.mapping = None
            tmp(Q_avail)
            Q_after = list(tmp.Q_avail)
            Q_avail = Q_after
            masked  = [q for q in Q_before if q not in Q_after]
            kept    = Q_after
            if sub_motif.mapping is not None and masked and kept:
                gate_fn = sub_motif.mapping.function
                n_sym   = sub_motif.mapping.n_symbols
                for q_masked in masked:
                    closest = min(kept, key=lambda q: abs(q - q_masked))
                    applications.append((gate_fn, [q_masked, closest], n_sym))

        # ── Unmask ─────────────────────────────────────────────
        elif isinstance(sub_motif, Qunmask):
            Q_avail = list(range(n_qubits))

    return applications, Q_avail


def build_qcnn_qnode(motif, n_qubits: int = N_QUBITS, dense: bool = None):
    """
    Compile a Qmotifs object into a PennyLane QNode for binary output.

    Circuit flow
    ------------
    AngleEmbedding(...)                            ← single or dense,
                                                       see module docstring
    [gate applications from motif]                 ← conv + pool, any order
    qml.probs(wires=remaining[:1])                 ← first surviving qubit

    Parameters
    ----------
    dense : which encoding to build. Defaults to C.is_dense_encoding()
        (correct for direct calls in the driver process). Ray remote
        workers hold a separate, possibly stale copy of the config
        module, so evolution.py's evaluate_genotype passes this
        explicitly rather than relying on the default.

    Returns
    -------
    circuit_as_tensor : callable(inputs, params) -> torch.Tensor shape (2,)
                        [p(|0⟩), p(|1⟩)]
    n_params          : int   total trainable parameters
    raw_qnode         : bare @qml.qnode (for draw_mpl)
    """
    gate_apps, remaining = get_gate_applications(motif, n_qubits)
    n_params = sum(n_sym for _, _, n_sym in gate_apps)

    if len(remaining) < N_READOUT_QUBITS:
        raise ValueError(
            f"Circuit leaves only {len(remaining)} qubit(s) active, "
            f"but {N_READOUT_QUBITS} are required for "
            f"{2**N_READOUT_QUBITS}-class readout."
        )

    # Measure the first N_READOUT_QUBITS survivors — NOT necessarily
    # wire 0, since pooling may have masked lower-index qubits.
    readout_wires = remaining[:N_READOUT_QUBITS]

    if dense is None:
        dense = C.is_dense_encoding()

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, params):
        if dense:
            # ── Dense encoding: 2 features per qubit ───────────
            # First half → RX on every qubit
            qml.AngleEmbedding(
                inputs[:n_qubits], wires=range(n_qubits), rotation="X"
            )
            # Second half → RY on every qubit, stacked on top
            qml.AngleEmbedding(
                inputs[n_qubits: 2 * n_qubits], wires=range(n_qubits), rotation="Y"
            )
        else:
            # ── Single encoding: 1 feature per qubit ───────────
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # ── Conv + pool layers, in genotype order ─────────────
        idx = 0
        for gate_fn, bits, n_sym in gate_apps:
            gate_fn(bits, params[idx: idx + n_sym])
            idx += n_sym

        # ── Basis-state probability measurement ────────────────
        return qml.probs(wires=readout_wires)

    def circuit_as_tensor(inputs, params):
        out = circuit(inputs, params)
        if isinstance(out, torch.Tensor):
            return out          # shape (2,)
        return torch.tensor(out, dtype=torch.float32)

    return circuit_as_tensor, n_params, circuit
