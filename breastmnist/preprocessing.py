# ----------------------------------------------------------------
# preprocessing.py  —  PCA feature extraction for BreastMNIST
#                      (PCA-8 single or PCA-16 dense qubit encoding)
#
# Pipeline:
#   1. Flatten grayscale images (N, 28, 28) → (N, 784)
#   2. StandardScaler       zero-mean, unit-variance per pixel
#   3. PCA(C.PCA_COMPONENTS components)   fitted on TRAINING data only
#   4. Scale to [0, π]      valid angle range for the rotation axis/axes
#
# Unlike RetinaMNIST, BreastMNIST is already binary (see config.py),
# so there is no class-merging step here — labels are used as-is.
#
# C.PCA_COMPONENTS is read fresh (not imported by value) so that
# main.py can switch it between 8 and 16 across loop iterations and
# have fit_pca_pipeline() pick up the change immediately.
#
#   PCA_COMPONENTS == N_QUBITS      → single encoding (see circuit.py):
#                                      all components → RY on qubits 0–7
#   PCA_COMPONENTS == 2 * N_QUBITS  → dense encoding:
#                                      components  0–7   → RX on qubits 0–7
#                                      components  8–15  → RY on qubits 0–7
#
# Because every component is scaled to the same [0, π] range, in the
# dense case neither rotation axis is systematically favoured.
# ----------------------------------------------------------------

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as C
from config import N_QUBITS


def _flatten(x_images: np.ndarray) -> np.ndarray:
    """
    Flatten to (N, n_features) float32 in [0, 1].
    Grayscale (N, 28, 28) → (N, 784). Handles an explicit channel
    axis too, so RGB-shaped input would work the same way.
    """
    return x_images.reshape(len(x_images), -1).astype(np.float32) / 255.0


def fit_pca_pipeline(x_train_images: np.ndarray):
    """
    Fit the full PCA preprocessing pipeline on training images, using
    C.PCA_COMPONENTS as currently set (8 or 16).
    Must only be called on the training split — never on val/test.

    Parameters
    ----------
    x_train_images : np.ndarray  (N, 28, 28)  uint8

    Returns
    -------
    scaler : fitted StandardScaler
    pca    : fitted PCA(n_components=C.PCA_COMPONENTS)
    x_min  : np.ndarray (PCA_COMPONENTS,)  per-component min of train set
    x_max  : np.ndarray (PCA_COMPONENTS,)  per-component max of train set
    """
    n_components = C.PCA_COMPONENTS
    C.assert_valid_pca_components(n_components)

    x_flat   = _flatten(x_train_images)
    scaler   = StandardScaler()
    x_scaled = scaler.fit_transform(x_flat)

    pca   = PCA(n_components=n_components, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    x_min = x_pca.min(axis=0)
    x_max = x_pca.max(axis=0)

    ratios    = pca.explained_variance_ratio_
    total_var = ratios.sum()

    print(
        f"PCA fitted on {len(x_flat)} images "
        f"({x_flat.shape[1]} raw features → {n_components} components)"
    )
    print(f"  Total variance explained : {total_var*100:.1f}%")

    if C.is_dense_encoding(n_components):
        # Dense encoding: report the RX half and RY half separately.
        print(f"  RX half (components 0–{N_QUBITS-1})    : "
              f"{ratios[:N_QUBITS].sum()*100:.1f}%")
        print(f"  RY half (components {N_QUBITS}–{n_components-1})  : "
              f"{ratios[N_QUBITS:].sum()*100:.1f}%")
    else:
        # Single encoding: one RY rotation per qubit, nothing to split.
        print(f"  RY (components 0–{n_components-1})      : "
              f"{ratios.sum()*100:.1f}%")

    print(f"  Per-component            : {ratios.round(4)}")

    return scaler, pca, x_min, x_max


def apply_pca_pipeline(x_images: np.ndarray, y: np.ndarray,
                       scaler, pca, x_min, x_max):
    """
    Apply a fitted PCA pipeline to a data split and return tensors
    ready for AngleEmbedding (single or dense, decided in circuit.py).

    Parameters
    ----------
    x_images : np.ndarray  (N, 28, 28)  uint8
    y        : np.ndarray  (N,) or (N,1)   int labels (0/1)
    scaler   : fitted StandardScaler
    pca      : fitted PCA
    x_min    : per-component min from training set
    x_max    : per-component max from training set

    Returns
    -------
    x_t : torch.FloatTensor  (N, PCA_COMPONENTS)  values in [0, π]
    y_t : torch.LongTensor   (N,)
    """
    y        = y.squeeze().astype(np.int64)
    x_flat   = _flatten(x_images)
    x_scaled = scaler.transform(x_flat)
    x_pca    = pca.transform(x_scaled)

    # Scale every component to [0, π].  In the dense case both encoding
    # halves share the same range so neither rotation axis dominates.
    x_angle = (x_pca - x_min) / (x_max - x_min + 1e-8) * np.pi

    return (
        torch.tensor(x_angle, dtype=torch.float32),
        torch.tensor(y,       dtype=torch.long),
    )


def save_pipeline(scaler, pca, x_min, x_max, path: str) -> None:
    """Save the fitted pipeline for later reuse (inference, winner.py)."""
    joblib.dump(
        {"scaler": scaler, "pca": pca, "x_min": x_min, "x_max": x_max},
        path,
    )
    print(f"PCA pipeline saved → {path}")


def load_pipeline(path: str):
    """Load a saved PCA pipeline."""
    d = joblib.load(path)
    return d["scaler"], d["pca"], d["x_min"], d["x_max"]
