# ----------------------------------------------------------------
# config.py  —  All hyperparameters and constants
#               BloodMNIST 8-class / PCA-8 or PCA-16 qubit encoding
#               VARIABLE-LENGTH QCNN
#
# Ported from PCA_search_breast (BreastMNIST, binary). The search
# mechanics — variable-length genotypes, PCA-variant loop, sweep grid,
# L1/L2/L3 regularisation — are unchanged. What's different for this
# 8-class task:
#   N_CLASSES = 8, W_CLASS has 8 entries, and N_READOUT_QUBITS = 3
#   (3 qubits' 8 basis-state probabilities [p(000)...p(111)] map
#   directly onto the 8 classes — one basis state per class — instead
#   of 1 qubit's 2 probabilities for BreastMNIST's binary task).
# ----------------------------------------------------------------

# ── Circuit ───────────────────────────────────────────────────────
N_QUBITS  = 8
N_CLASSES = 8

# The two supported PCA feature counts, and the qubit encoding each
# implies:
#   PCA_COMPONENTS == N_QUBITS      → single encoding, 1 feature/qubit
#                                      (RY rotation only)
#   PCA_COMPONENTS == 2 * N_QUBITS  → dense encoding, 2 features/qubit
#                                      (RX then RY rotation)
# main.py loops over PCA_VARIANTS, setting C.PCA_COMPONENTS to each
# value in turn before data prep / search / training for that variant.
PCA_VARIANTS   = [8, 16]
PCA_COMPONENTS = PCA_VARIANTS[-1]   # default for standalone imports


def is_dense_encoding(pca_components: int = None) -> bool:
    """True → 2 features/qubit (RX+RY). False → 1 feature/qubit (RY)."""
    if pca_components is None:
        pca_components = PCA_COMPONENTS
    return pca_components == 2 * N_QUBITS


def assert_valid_pca_components(pca_components: int = None) -> None:
    """Every supported PCA size must map cleanly onto N_QUBITS."""
    if pca_components is None:
        pca_components = PCA_COMPONENTS
    assert pca_components in (N_QUBITS, 2 * N_QUBITS), (
        f"PCA_COMPONENTS must equal N_QUBITS ({N_QUBITS}, single encoding) "
        f"or 2 * N_QUBITS ({2 * N_QUBITS}, dense encoding); "
        f"got {pca_components}"
    )


assert_valid_pca_components()

# ================================================================
# VARIABLE-LENGTH QCNN STRUCTURE
# ================================================================
# A genotype is ANY ordered sequence of Qcycle (conv) and Qmask
# (pool) layers — e.g. all of these are legal:
#
#   [C, C, C]                    (no pooling at all, 8 qubits survive)
#   [C, P, C, C, C, P]
#   [C, C, P, C, C, C, C, P, C]
#   [C, P, C]
#
# The ONLY hard structural requirement is that after all pooling
# layers have executed, at least N_READOUT_QUBITS qubit(s) remain
# active so the class measurement is still possible. With
# N_READOUT_QUBITS = 3 this means a genotype that pools 8 qubits down
# to only 1 or 2 survivors is automatically REJECTED by
# evolution.is_valid_motif() — no extra tuning needed beyond setting
# N_READOUT_QUBITS below; this was verified empirically (random and
# structurally-biased genotypes reliably find >=3-qubit-survivor
# solutions within a handful of retries).
# ================================================================

MIN_LAYERS = 3      # shortest allowed genotype (total conv + pool)
MAX_LAYERS = 10     # longest allowed — bounds runtime and n_params

MIN_CONV_LAYERS = 1 # must be a *convolutional* NN, so at least one Qcycle
MIN_POOL_LAYERS = 0 # zero pooling is allowed

P_CONV = 0.60       # probability a randomly sampled layer is convolutional
                    # >0.5 keeps qubits alive, so fewer genotypes get
                    # rejected for pooling below N_READOUT_QUBITS

N_READOUT_QUBITS = 3   # measure 3 qubits → 8 basis-state probabilities
                       #   [p(000), p(001), ..., p(111)] → 8 classes
                       # Since N_CLASSES == 2**N_READOUT_QUBITS here,
                       # circuit.py's fold step is a no-op: each basis
                       # state IS a class, 1:1.
                       # Whichever 3 qubits are FIRST among the survivors
                       # after all pooling — not necessarily wires 0-2.

# N_CLASSES == 2**N_READOUT_QUBITS here, so circuit.py's basis-state
# fold is a no-op; this is only defined for interface parity with the
# DermaMNIST config, which DOES need it.
EXTRA_STATE_TARGET_CLASS = None

# Degeneracy guard: reject circuits whose conv layers barely connect
# anything.  Required total conv gate applications = this × n_conv_layers.
MIN_CONV_APPS_PER_LAYER = 2

# ── Class mapping ─────────────────────────────────────────────────
# BloodMNIST labels 8 types of peripheral blood cells (already the
# dataset's native classes — no merging step, unlike RetinaMNIST's
# grade-merging). Label order matches the MedMNIST BloodMNIST release:
CLASS_NAMES = [
    "basophil",     # 0
    "eosinophil",   # 1
    "erythroblast", # 2
    "ig",           # 3  immature granulocytes (myelocytes / metamyelocytes / promyelocytes)
    "lymphocyte",   # 4
    "monocyte",     # 5
    "neutrophil",   # 6
    "platelet",     # 7
]

# ── Class-imbalance weights ────────────────────────────────────────
# Unlike BreastMNIST's malignant/benign clinical-severity framing,
# BloodMNIST has no single "costly error" class — it's an 8-way cell
# typing task. Instead these weights are set INVERSELY PROPORTIONAL to
# each class's natural training-set frequency, so the fitness function
# doesn't let the search coast on the majority cell types (neutrophil,
# eosinophil) at the expense of the rarer ones (basophil, lymphocyte).
# Natural full-dataset counts: basophil=1218 eosinophil=3117
# erythroblast=1551 ig=2895 lymphocyte=1214 monocyte=1420
# neutrophil=3329 platelet=2348. Must sum to 1.0.
W_CLASS = [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097]
assert abs(sum(W_CLASS) - 1.0) < 1e-6, "W_CLASS must sum to 1.0"
assert len(W_CLASS) == N_CLASSES,      "W_CLASS must have N_CLASSES entries"

# ── Architecture search ───────────────────────────────────────────
N_EPOCHS_EVAL   = 60
LR_EVAL         = 0.003
BATCH_SIZE_EVAL = 64
INIT_POP        = 10
MAX_EVO_STEPS   = 10

# ── Tournament / exploration ──────────────────────────────────────
P_EXPLORE = 0.50

# ── Mutation operator weights ──────────────────────────────────────
#   replace_tail : keep a prefix, regrow a fresh random tail
#   insert       : insert one new layer at a random position (grows)
#   delete       : remove one layer at a random position   (shrinks)
#   point        : swap one layer for a fresh layer of the same kind
P_MUTATE = {
    "replace_tail": 0.40,
    "insert":       0.20,
    "delete":       0.20,
    "point":        0.20,
}
assert abs(sum(P_MUTATE.values()) - 1.0) < 1e-6, "P_MUTATE must sum to 1.0"

# Attempts allowed per child before giving up
MAX_CHILD_ATTEMPTS = 25

# ── Regularisation ───────────────────────────────────────────────
# With variable length, L2 is the primary pressure keeping circuits
# short.  Raise it if the search drifts toward MAX_LAYERS.
L1 = 5e-4
L2 = 1e-4

# Light, direct penalty on genotype depth (total layer count), on top
# of L2's indirect pressure via parameter count. Kept small so it
# nudges rather than dominates the recall terms.
L3 = 2e-3

# ── Evolution misc ────────────────────────────────────────────────
BAD_FITNESS   = 2.0
NQ_MAX        = N_QUBITS
SAVE_INTERVAL = 1

# Soft penalty applied when the worst class falls below this recall.
# Prevents the search converging on a circuit that ignores one class
# entirely (e.g. always predicting "neutrophil").
MIN_RECALL_TARGET  = 0.10
MIN_RECALL_PENALTY = 2.0    # multiplier on the shortfall

# ── Final training ─────────────────────────────────────────────────
FINAL_N_EPOCHS   = 200
FINAL_LR         = 0.005
FINAL_BATCH_SIZE = 25
FINAL_LR_STEP    = 50
FINAL_LR_GAMMA   = 0.5
FINAL_EXP_DIR    = "experiments/qcnn_final"

# ── Search subsampling ────────────────────────────────────────────
# Subsample the training set during the SEARCH only; final training
# always uses everything.  Set to None to disable.
MAX_SEARCH_TRAIN = 2000

# ================================================================
# PARAMETER SWEEP GRID
# ================================================================
# Each entry is run once per PCA_VARIANTS value (main.py nests this
# grid inside the PCA-variant loop). W_CLASS entries must always have
# 8 values summing to 1.0.
# ================================================================
SWEEP_GRID = [
    {
        "label":         "baseline",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    # ── Learning rate ─────────────────────────────────────────
    {
        "label":         "lr_high",
        "LR_EVAL":       0.01,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    {
        "label":         "lr_low",
        "LR_EVAL":       0.001,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    # ── Population / steps ────────────────────────────────────
    {
        "label":         "large_pop",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      20,
        "MAX_EVO_STEPS": 15,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    # ── Structure sweeps ──────────────────────────────────────
    {
        "label":         "short_circuits",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "MIN_LAYERS":    2,
        "MAX_LAYERS":    5,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    {
        "label":         "deep_circuits",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "MIN_LAYERS":    6,
        "MAX_LAYERS":    14,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    {
        "label":         "conv_heavy",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "P_CONV":        0.85,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    {
        "label":         "pool_required",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "P_CONV":        0.50,
        "MIN_POOL_LAYERS": 2,
        "W_CLASS":       [0.187, 0.073, 0.147, 0.079, 0.188, 0.161, 0.068, 0.097],
    },
    # ── Class weighting ────────────────────────────────────────
    {
        # Sharper inverse-frequency-squared weighting — pushes the
        # search harder toward getting the rarest cell types
        # (basophil, lymphocyte) right, at the cost of some majority
        # -class (neutrophil, eosinophil) recall.
        "label":         "rare_class_focused",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.244, 0.037, 0.151, 0.043, 0.246, 0.180, 0.033, 0.066],
    },
    {
        "label":         "uniform_weights",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.125] * 8,
    },
    # ── Combined ──────────────────────────────────────────────
    {
        "label":         "combined_best",
        "LR_EVAL":       0.01,
        "N_EPOCHS_EVAL": 80,
        "INIT_POP":      20,
        "MAX_EVO_STEPS": 15,
        "MAX_LAYERS":    12,
        "P_CONV":        0.70,
        "W_CLASS":       [0.244, 0.037, 0.151, 0.043, 0.246, 0.180, 0.033, 0.066],
    },
]
