# ----------------------------------------------------------------
# config.py  —  All hyperparameters and constants
#               RetinaMNIST 4-class / PCA-8 or PCA-16 qubit encoding
#               VARIABLE-LENGTH QCNN
# ----------------------------------------------------------------

# ── Circuit ───────────────────────────────────────────────────────
N_QUBITS  = 8
N_CLASSES = 4

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
# layers have executed, at least N_READOUT_QUBITS qubits remain
# active so the 4-class measurement is still possible.
# ================================================================

MIN_LAYERS = 3      # shortest allowed genotype (total conv + pool)
MAX_LAYERS = 10     # longest allowed — bounds runtime and n_params

MIN_CONV_LAYERS = 1 # must be a *convolutional* NN, so at least one Qcycle
MIN_POOL_LAYERS = 0 # zero pooling is allowed

P_CONV = 0.60       # probability a randomly sampled layer is convolutional
                    # >0.5 keeps qubits alive, so fewer genotypes get
                    # rejected for pooling below N_READOUT_QUBITS

N_READOUT_QUBITS = 2   # measure 2 qubits → 4 joint probabilities
                       #   p(|00⟩) → class 0  (No DR)
                       #   p(|01⟩) → class 1  (Mild NPDR)
                       #   p(|10⟩) → class 2  (Moderate NPDR)
                       #   p(|11⟩) → class 3  (Severe+PDR)

# Degeneracy guard: reject circuits whose conv layers barely connect
# anything.  Required total conv gate applications = this × n_conv_layers.
MIN_CONV_APPS_PER_LAYER = 2

# ── Class mapping ─────────────────────────────────────────────────
# RetinaMNIST ships 5 grades (0–4).  Grade 4 (proliferative DR) is
# merged into grade 3 (severe NPDR), leaving 4 classes.
MERGE_FROM  = 4
MERGE_TO    = 3
CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe+PDR"]

# ── Clinical severity weights ─────────────────────────────────────
# Grade 3 (severe + proliferative) is the most critical to detect,
# so missing it costs 4× more than missing grade 0.  Must sum to 1.0.
W_CLASS = [0.10, 0.20, 0.30, 0.40]
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

# ── Mutation operator weights ─────────────────────────────────────
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
SAVE_INTERVAL = 5

# Soft penalty applied when the worst class falls below this recall.
# Prevents the search converging on circuits that ignore a whole grade.
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
# 4 values summing to 1.0.
# ================================================================
SWEEP_GRID = [
    {
        "label":         "baseline",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    # ── Learning rate ─────────────────────────────────────────
    {
        "label":         "lr_high",
        "LR_EVAL":       0.01,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    {
        "label":         "lr_low",
        "LR_EVAL":       0.001,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    # ── Population / steps ────────────────────────────────────
    {
        "label":         "large_pop",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      20,
        "MAX_EVO_STEPS": 15,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
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
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    {
        "label":         "deep_circuits",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "MIN_LAYERS":    6,
        "MAX_LAYERS":    14,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    {
        "label":         "conv_heavy",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "P_CONV":        0.85,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    {
        "label":         "pool_required",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "P_CONV":        0.50,
        "MIN_POOL_LAYERS": 2,
        "W_CLASS":       [0.10, 0.20, 0.30, 0.40],
    },
    # ── Clinical weighting ────────────────────────────────────
    {
        "label":         "grade3_focused",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.05, 0.10, 0.25, 0.60],
    },
    {
        "label":         "uniform_weights",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.25, 0.25, 0.25, 0.25],
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
        "W_CLASS":       [0.05, 0.10, 0.25, 0.60],
    },
]
