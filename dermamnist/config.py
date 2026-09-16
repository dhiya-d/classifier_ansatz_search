# ----------------------------------------------------------------
# config.py  —  All hyperparameters and constants
#               DermaMNIST 7-class / PCA-8 or PCA-16 qubit encoding
#               VARIABLE-LENGTH QCNN
#
# Ported from PCA_search_breast (BreastMNIST, binary). The search
# mechanics — variable-length genotypes, PCA-variant loop, sweep grid,
# L1/L2/L3 regularisation — are unchanged. What's different for this
# 7-class task:
#   N_CLASSES = 7, W_CLASS has 7 entries, and N_READOUT_QUBITS = 3
#   (3 qubits give 8 basis-state probabilities [p(000)...p(111)], one
#   more than the 7 classes need. circuit.py folds the single leftover
#   basis state into EXTRA_STATE_TARGET_CLASS below rather than
#   dropping it, so every measured probability still counts.)
# ----------------------------------------------------------------

# ── Circuit ───────────────────────────────────────────────────────
N_QUBITS  = 8
N_CLASSES = 7

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
                       #   [p(000), p(001), ..., p(111)] → 7 classes
                       # 8 states for 7 classes leaves ONE leftover
                       # state; circuit.py sums it into
                       # EXTRA_STATE_TARGET_CLASS instead of discarding
                       # it, so the output is still a valid 7-way
                       # probability distribution.
                       # Whichever 3 qubits are FIRST among the survivors
                       # after all pooling — not necessarily wires 0-2.

# Which class index absorbs the one leftover basis state left over
# from folding 8 quantum outcomes down to 7 classes (see circuit.py's
# _fold_probs_to_classes). Set to class 3 ("df" / dermatofibroma), the
# RAREST class in the training data (115 samples) — giving it two
# basis states' worth of probability mass to draw on makes it easier
# for the circuit to route probability toward this otherwise
# hardest-to-learn class, helping the MIN_RECALL_PENALTY floor below.
EXTRA_STATE_TARGET_CLASS = 3

# Degeneracy guard: reject circuits whose conv layers barely connect
# anything.  Required total conv gate applications = this × n_conv_layers.
MIN_CONV_APPS_PER_LAYER = 2

# ── Class mapping ─────────────────────────────────────────────────
# DermaMNIST labels 7 types of pigmented skin lesions from the HAM10000
# dataset (already the dataset's native classes — no merging step,
# unlike RetinaMNIST's grade-merging). Label order matches the
# MedMNIST DermaMNIST release:
CLASS_NAMES = [
    "akiec",  # 0  actinic keratoses / intraepithelial carcinoma (pre-malignant)
    "bcc",    # 1  basal cell carcinoma (malignant)
    "bkl",    # 2  benign keratosis-like lesions
    "df",     # 3  dermatofibroma (benign, rarest class)
    "mel",    # 4  melanoma (malignant, most dangerous — highest weight)
    "nv",     # 5  melanocytic nevi (benign, majority class)
    "vasc",   # 6  vascular lesions
]

# ── Clinical severity weights ──────────────────────────────────────
# Missing melanoma (a false negative on class 4) is the costly error —
# same rationale as BreastMNIST's malignant-vs-benign weighting — so
# it gets by far the largest weight. The other malignant/pre-malignant
# classes (bcc, akiec) get elevated but smaller weights; the four
# benign classes (bkl, df, nv, vasc) split the remainder evenly.
# Must sum to 1.0.
W_CLASS = [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125]
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
# entirely (e.g. always predicting "nv", the majority class).
MIN_RECALL_TARGET  = 0.10
MIN_RECALL_PENALTY = 2.0    # multiplier on the shortfall

# ── Final training ─────────────────────────────────────────────────
# Was 200; with the per-sample circuit-call loop in train.py/evolution.py
# replaced by a single batched circuit call (~25-60x faster per epoch,
# verified against the old per-sample loop for identical loss/gradients),
# 200 epochs is no longer prohibitively slow on its own — this cut is a
# second, independent lever to shorten sweep turnaround further. Kept as
# a multiple of FINAL_LR_STEP so the LR decay schedule below isn't
# truncated mid-step (still gets 2 full halvings: 0.005 -> 0.0025).
FINAL_N_EPOCHS   = 100
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
# 7 values summing to 1.0.
# ================================================================
SWEEP_GRID = [
    {
        "label":         "baseline",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    # ── Learning rate ─────────────────────────────────────────
    {
        "label":         "lr_high",
        "LR_EVAL":       0.01,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    {
        "label":         "lr_low",
        "LR_EVAL":       0.001,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    # ── Population / steps ────────────────────────────────────
    {
        "label":         "large_pop",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      20,
        "MAX_EVO_STEPS": 15,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
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
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    {
        "label":         "deep_circuits",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "MIN_LAYERS":    6,
        "MAX_LAYERS":    14,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    {
        "label":         "conv_heavy",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "P_CONV":        0.85,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    {
        "label":         "pool_required",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "P_CONV":        0.50,
        "MIN_POOL_LAYERS": 2,
        "W_CLASS":       [0.08, 0.14, 0.125, 0.125, 0.28, 0.125, 0.125],
    },
    # ── Clinical weighting ────────────────────────────────────
    {
        "label":         "melanoma_focused",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.10, 0.15, 0.075, 0.075, 0.45, 0.075, 0.075],
    },
    {
        "label":         "uniform_weights",
        "LR_EVAL":       0.003,
        "N_EPOCHS_EVAL": 60,
        "INIT_POP":      10,
        "MAX_EVO_STEPS": 10,
        "W_CLASS":       [0.14285714] * 7,
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
        "W_CLASS":       [0.10, 0.15, 0.075, 0.075, 0.45, 0.075, 0.075],
    },
]
