# ----------------------------------------------------------------
# gates.py  —  PennyLane gate definitions and gate registry
#
# Identical across all datasets in this project. Every gate is a
# 2-qubit unitary; conv gates are used in Qcycle layers and pool
# gates in Qmask layers.
# ----------------------------------------------------------------

import pennylane as qml
from hierarqcal import Qunitary

# ── Convolutional gate functions ───────────────────────────────────

def U_TTN(bits, symbols):                                   # 2 params
    qml.RY(symbols[0], wires=bits[0])
    qml.RY(symbols[1], wires=bits[1])
    qml.CNOT(wires=[bits[0], bits[1]])


def U_5(bits, symbols):                                     # 10 params
    qml.RX(symbols[0], wires=bits[0]); qml.RX(symbols[1], wires=bits[1])
    qml.RZ(symbols[2], wires=bits[0]); qml.RZ(symbols[3], wires=bits[1])
    qml.CRZ(symbols[4], wires=[bits[1], bits[0]])
    qml.CRZ(symbols[5], wires=[bits[0], bits[1]])
    qml.RX(symbols[6], wires=bits[0]); qml.RX(symbols[7], wires=bits[1])
    qml.RZ(symbols[8], wires=bits[0]); qml.RZ(symbols[9], wires=bits[1])


def U_6(bits, symbols):                                     # 10 params
    qml.RX(symbols[0], wires=bits[0]); qml.RX(symbols[1], wires=bits[1])
    qml.RZ(symbols[2], wires=bits[0]); qml.RZ(symbols[3], wires=bits[1])
    qml.CRX(symbols[4], wires=[bits[1], bits[0]])
    qml.CRX(symbols[5], wires=[bits[0], bits[1]])
    qml.RX(symbols[6], wires=bits[0]); qml.RX(symbols[7], wires=bits[1])
    qml.RZ(symbols[8], wires=bits[0]); qml.RZ(symbols[9], wires=bits[1])


def U_9(bits, symbols):                                     # 2 params
    qml.Hadamard(wires=bits[0]); qml.Hadamard(wires=bits[1])
    qml.CZ(wires=[bits[0], bits[1]])
    qml.RX(symbols[0], wires=bits[0]); qml.RX(symbols[1], wires=bits[1])


def U_13(bits, symbols):                                    # 6 params
    qml.RY(symbols[0], wires=bits[0]); qml.RY(symbols[1], wires=bits[1])
    qml.CRZ(symbols[2], wires=[bits[1], bits[0]])
    qml.RY(symbols[3], wires=bits[0]); qml.RY(symbols[4], wires=bits[1])
    qml.CRZ(symbols[5], wires=[bits[0], bits[1]])


def U_14(bits, symbols):                                    # 6 params
    qml.RY(symbols[0], wires=bits[0]); qml.RY(symbols[1], wires=bits[1])
    qml.CRX(symbols[2], wires=[bits[1], bits[0]])
    qml.RY(symbols[3], wires=bits[0]); qml.RY(symbols[4], wires=bits[1])
    qml.CRX(symbols[5], wires=[bits[0], bits[1]])


def U_15(bits, symbols):                                    # 4 params
    qml.RY(symbols[0], wires=bits[0]); qml.RY(symbols[1], wires=bits[1])
    qml.CNOT(wires=[bits[1], bits[0]])
    qml.RY(symbols[2], wires=bits[0]); qml.RY(symbols[3], wires=bits[1])
    qml.CNOT(wires=[bits[0], bits[1]])


def U_SO4(bits, symbols):                                   # 6 params
    qml.RY(symbols[0], wires=bits[0]); qml.RY(symbols[1], wires=bits[1])
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.RY(symbols[2], wires=bits[0]); qml.RY(symbols[3], wires=bits[1])
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.RY(symbols[4], wires=bits[0]); qml.RY(symbols[5], wires=bits[1])


def U_SU4(bits, symbols):                                   # 15 params
    qml.U3(symbols[0],  symbols[1],  symbols[2],  wires=bits[0])
    qml.U3(symbols[3],  symbols[4],  symbols[5],  wires=bits[1])
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.RY(symbols[6],  wires=bits[0]); qml.RZ(symbols[7],  wires=bits[1])
    qml.CNOT(wires=[bits[1], bits[0]])
    qml.RY(symbols[8],  wires=bits[0])
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.U3(symbols[9],  symbols[10], symbols[11], wires=bits[0])
    qml.U3(symbols[12], symbols[13], symbols[14], wires=bits[1])


def U_gate(bits, symbols):                                  # 9 params
    qml.U3(symbols[0], symbols[1], symbols[2], wires=bits[0])
    qml.ctrl(qml.U3, control=bits[0])(symbols[3], symbols[4], symbols[5], wires=bits[1])
    qml.ctrl(qml.U3, control=bits[1])(symbols[6], symbols[7], symbols[8], wires=bits[0])


# ── Pooling gate functions ─────────────────────────────────────────

def pool_U(bits, symbols):                                  # 6 params
    qml.ctrl(qml.U3, control=bits[0])(symbols[0], symbols[1], symbols[2], wires=bits[1])
    qml.ctrl(qml.U3, control=bits[1])(symbols[3], symbols[4], symbols[5], wires=bits[0])


def pool_ansatz(bits, symbols):                             # 2 params
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
    qml.CRX(symbols[1], wires=[bits[0], bits[1]])


# ── Gate registry ──────────────────────────────────────────────────

MAPPING_DICT = {}
conv_gates   = []
pool_gates   = []

_conv_specs = [
    ("U_TTN",  U_TTN,   2),
    ("U_5",    U_5,    10),
    ("U_6",    U_6,    10),
    ("U_9",    U_9,     2),
    ("U_13",   U_13,    6),
    ("U_14",   U_14,    6),
    ("U_15",   U_15,    4),
    ("U_SO4",  U_SO4,   6),
    ("U_SU4",  U_SU4,  15),
    ("U_gate", U_gate,  9),
]
_pool_specs = [
    ("pool_U",      pool_U,      6),
    ("pool_ansatz", pool_ansatz, 2),
]

for _name, _fn, _n_sym in _conv_specs:
    _gate = Qunitary(_fn, n_symbols=_n_sym, arity=2)
    _gate.name = _name
    MAPPING_DICT[_name] = _gate
    conv_gates.append(_gate)

for _name, _fn, _n_sym in _pool_specs:
    _gate = Qunitary(_fn, n_symbols=_n_sym, arity=2)
    _gate.name = _name
    MAPPING_DICT[_name] = _gate
    pool_gates.append(_gate)

hierq_gates = conv_gates + pool_gates


def verify_gates() -> None:
    """Sanity-check all registered gates at import time."""
    for gate in hierq_gates:
        assert hasattr(gate, "function"),  f"{gate.name} missing .function"
        assert hasattr(gate, "n_symbols"), f"{gate.name} missing .n_symbols"
        assert hasattr(gate, "arity"),     f"{gate.name} missing .arity"
        assert gate.arity == 2,            f"{gate.name} arity should be 2"


verify_gates()