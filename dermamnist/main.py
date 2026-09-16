#!/usr/bin/env python3
# ----------------------------------------------------------------
# main.py  —  Entry point: load data → PCA (8 & 16) → baseline →
#             search → final training → save
#             DermaMNIST 7-class QCNN
#             Variable-length architecture, covering both the
#             single (PCA-8) and dense (PCA-16) qubit encodings.
#
# PCA-8 and PCA-16 are fully independent runs (separate output dirs,
# separate Ray clusters), so "--variant both" (the default) launches
# them as two PARALLEL SUBPROCESSES rather than running one after the
# other in a single process. Pass --serial to fall back to the old
# sequential single-process behaviour.
#
# Usage
# -----
#   python main.py                      both PCA variants, IN PARALLEL
#   python main.py --serial             both PCA variants, sequentially
#   python main.py --variant 8          PCA-8 only, this process
#   python main.py --variant 16         PCA-16 only, this process
#   python main.py --sweep              both variants (parallel) × every SWEEP_GRID entry
#   python main.py --sweep --variant 8  PCA-8 × every SWEEP_GRID entry
#   python main.py --sweep-index 0      both variants (parallel), one SWEEP_GRID entry
# ----------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # hide GPUs from Ray's NVML probe

# Must be set before numpy/scipy/sklearn are imported anywhere in this
# process. Without this, sklearn's PCA (randomized SVD -> scipy.linalg.lu
# -> OpenBLAS) can deadlock permanently the first time it's called after
# ray.init() — OpenBLAS lazily spins up its own thread pool, and doing
# that post-fork (Ray pre-forks worker processes at init) hangs forever.
# Confirmed via a captured stack trace stuck in blas_thread_init while
# fitting the PCA-16 pipeline. Single-threaded BLAS sidesteps it entirely
# and costs nothing here — this fit is a one-time, sub-second call.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import os
import argparse
import csv
import json
import logging

import numpy as np
import ray
import torch

from preprocessing import (
    fit_pca_pipeline, apply_pca_pipeline, save_pipeline,
)
import config as C
from evolution import reset_caches, run_evolution, count_layers, layer_signature
from train import final_train_and_test, save_best_circuit


# ================================================================
# UTILITIES
# ================================================================
def make_run_dir(base: str, label: str = "") -> str:
    """Create base/<label>_001, _002, … directory."""
    os.makedirs(base, exist_ok=True)
    prefix   = f"{label}_" if label else "run_"
    existing = [d for d in os.listdir(base) if d.startswith(prefix)]
    path     = os.path.join(base, f"{prefix}{len(existing) + 1:03d}")
    os.makedirs(path)
    return path


def setup_logging(exp_dir: str) -> None:
    """Reset handlers so each run logs to its own file."""
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, "run.log")),
            logging.StreamHandler(),
        ],
    )


def apply_sweep_params(params: dict) -> None:
    """
    Overwrite config module globals with one sweep entry.

    Values needed inside the Ray remote fitness function are ALSO passed
    explicitly at the call site in evolution.py, because Ray workers are
    separate processes with their own copy of the config module.

    PCA_COMPONENTS is deliberately NOT one of these keys — it is set by
    the outer PCA-variant loop in __main__, not by SWEEP_GRID entries,
    so the two axes (encoding size vs. search hyperparameters) stay
    independent.
    """
    # ── Training hyperparameters ──────────────────────────────
    C.LR_EVAL       = params.get("LR_EVAL",       C.LR_EVAL)
    C.N_EPOCHS_EVAL = params.get("N_EPOCHS_EVAL", C.N_EPOCHS_EVAL)
    C.BATCH_SIZE_EVAL = params.get("BATCH_SIZE_EVAL", C.BATCH_SIZE_EVAL)
    C.INIT_POP      = params.get("INIT_POP",      C.INIT_POP)
    C.MAX_EVO_STEPS = params.get("MAX_EVO_STEPS", C.MAX_EVO_STEPS)

    # ── Variable-length structure knobs ───────────────────────
    C.MIN_LAYERS      = params.get("MIN_LAYERS",      C.MIN_LAYERS)
    C.MAX_LAYERS      = params.get("MAX_LAYERS",      C.MAX_LAYERS)
    C.MIN_CONV_LAYERS = params.get("MIN_CONV_LAYERS", C.MIN_CONV_LAYERS)
    C.MIN_POOL_LAYERS = params.get("MIN_POOL_LAYERS", C.MIN_POOL_LAYERS)
    C.P_CONV          = params.get("P_CONV",          C.P_CONV)

    # ── Regularisation / penalties ────────────────────────────
    C.L1                 = params.get("L1", C.L1)
    C.L2                 = params.get("L2", C.L2)
    C.L3                 = params.get("L3", C.L3)
    C.MIN_RECALL_TARGET  = params.get("MIN_RECALL_TARGET",  C.MIN_RECALL_TARGET)
    C.MIN_RECALL_PENALTY = params.get("MIN_RECALL_PENALTY", C.MIN_RECALL_PENALTY)

    assert C.MIN_LAYERS <= C.MAX_LAYERS, \
        f"MIN_LAYERS ({C.MIN_LAYERS}) must be <= MAX_LAYERS ({C.MAX_LAYERS})"

    # ── Per-class fitness weights ─────────────────────────────
    if "W_CLASS" in params:
        w = params["W_CLASS"]
        assert len(w) == C.N_CLASSES, \
            f"W_CLASS must have {C.N_CLASSES} entries, got {len(w)}"
        assert abs(sum(w) - 1.0) < 1e-6, f"W_CLASS must sum to 1.0, got {w}"
        C.W_CLASS = w


# ================================================================
# DATA SPLITTING
# ================================================================
def split_data(x, y, train_frac: float = 0.75, val_frac: float = 0.10,
               random_state: int = 42):
    """
    Stratified split — each class is split independently so every class
    appears in all three sets at its natural proportion.

    train_frac : fraction of each class going to training   (0.75)
    val_frac   : fraction of each class going to validation (0.10)
    remainder  : goes to test                               (0.15)
    """
    rng     = np.random.default_rng(random_state)
    classes = np.unique(y)

    train_idx, val_idx, test_idx = [], [], []
    for cls in classes:
        idx     = rng.permutation(np.where(y == cls)[0])
        n       = len(idx)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        train_idx.append(idx[:n_train])
        val_idx  .append(idx[n_train: n_train + n_val])
        test_idx .append(idx[n_train + n_val:])

    train_idx = rng.permutation(np.concatenate(train_idx))
    val_idx   = rng.permutation(np.concatenate(val_idx))
    test_idx  = rng.permutation(np.concatenate(test_idx))

    return (
        x[train_idx], y[train_idx],
        x[val_idx],   y[val_idx],
        x[test_idx],  y[test_idx],
    )


def oversample_to_balance(x_train, y_train, random_state: int = 42):
    """
    Randomly oversample every minority class until all classes have equal
    count.  Only ever applied to the training set — val and test keep the
    natural distribution so metrics reflect real class imbalance.
    """
    rng     = np.random.default_rng(random_state)
    classes = np.unique(y_train)
    n_max   = max(int(np.sum(y_train == c)) for c in classes)

    all_idx = list(range(len(y_train)))
    for cls in classes:
        idx_c    = np.where(y_train == cls)[0]
        n_needed = n_max - len(idx_c)
        if n_needed > 0:
            all_idx.extend(
                rng.choice(idx_c, size=n_needed, replace=True).tolist()
            )

    all_idx = rng.permutation(all_idx)
    return x_train[all_idx], y_train[all_idx]


# ================================================================
# LINEAR CLASSIFIER BASELINE
# ================================================================
def run_linear_baseline(x_train_t, y_train_t,
                        x_val_t,   y_val_t,
                        x_test_t,  y_test_t,
                        exp_dir: str) -> dict:
    """
    Fit a logistic regression on the SAME PCA features the QCNN sees
    (8 or 16, whichever C.PCA_COMPONENTS is currently set to).

    This establishes how much classification signal exists in the
    compressed representation before any quantum processing, so the QCNN
    result can be reported against a fair classical reference.
    Saved to exp_dir/linear_baseline.json.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    logging.info(
        f"\n--- Linear classifier baseline (on {C.PCA_COMPONENTS} PCA "
        f"features) ---"
    )

    X_train, y_train = x_train_t.numpy(), y_train_t.numpy()
    X_val,   y_val   = x_val_t.numpy(),   y_val_t.numpy()
    X_test,  y_test  = x_test_t.numpy(),  y_test_t.numpy()

    clf = LogisticRegression(
        max_iter     = 1000,
        solver       = "lbfgs",
        C            = 1.0,
        random_state = 42,
    )
    clf.fit(X_train, y_train)

    results = {}
    for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = clf.predict(X_s)
        probs = clf.predict_proba(X_s)

        acc     = float(np.mean(preds == y_s))
        bal_acc = float(balanced_accuracy_score(y_s, preds))

        recalls = []
        for c in range(C.N_CLASSES):
            tp = int(np.sum((preds == c) & (y_s == c)))
            fn = int(np.sum((preds != c) & (y_s == c)))
            recalls.append(float(tp / (tp + fn + 1e-8)))

        try:
            y_bin = np.eye(C.N_CLASSES)[y_s]
            auc   = float(roc_auc_score(y_bin, probs,
                                        multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")

        results[split_name] = {
            "accuracy":          acc,
            "bal_acc":           bal_acc,
            "per_class_recalls": recalls,
            "min_recall":        float(min(recalls)),
            "roc_auc_ovr":       auc,
        }

        logging.info(
            f"  {split_name:5s} | acc={acc:.4f} | bal_acc={bal_acc:.4f} | "
            f"min_recall={min(recalls):.4f} | auc={auc:.4f}"
        )
        logging.info(
            f"         per-class: {[f'{r:.3f}' for r in recalls]}"
        )

    baseline_path = os.path.join(exp_dir, "linear_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({
            "description": (
                f"Logistic regression on the same {C.PCA_COMPONENTS} PCA "
                "components fed to the QCNN. Shows the classification "
                "signal available in the compressed representation "
                "before any quantum processing."
            ),
            "n_features": int(X_train.shape[1]),
            "n_train":    int(len(X_train)),
            "n_val":      int(len(X_val)),
            "n_test":     int(len(X_test)),
            "results":    results,
        }, f, indent=2)
    logging.info(f"Linear baseline saved → {baseline_path}")

    return results


# ================================================================
# DATA PREPARATION
# ================================================================
# Split into two phases because PCA_COMPONENTS now changes across the
# PCA-variant loop:
#   load_raw_split() — load, stratified split. Runs ONCE; identical raw
#                       images feed every variant, so the train/val/test
#                       split is directly comparable between PCA-8 and
#                       PCA-16 runs. DermaMNIST's 7 lesion-type labels
#                       are already the dataset's native classes, so
#                       there is no class-merging step here.
#   fit_and_encode()  — fit PCA (depends on C.PCA_COMPONENTS), oversample,
#                       encode. Runs ONCE PER VARIANT.
# ================================================================
def load_raw_split():
    """Load DermaMNIST, stratified split."""
    logging.info("Loading DermaMNIST data...")

    x_all = np.concatenate([
        np.load("dermamnist/train_images.npy"),
        np.load("dermamnist/val_images.npy"),
        np.load("dermamnist/test_images.npy"),
    ], axis=0)
    y_all = np.concatenate([
        np.load("dermamnist/train_labels.npy"),
        np.load("dermamnist/val_labels.npy"),
        np.load("dermamnist/test_labels.npy"),
    ], axis=0).squeeze().astype(int)

    logging.info(
        f"Full dataset: {len(x_all)} samples | image shape {x_all.shape[1:]}"
    )
    dist = {int(k): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))}
    logging.info(
        f"  Class distribution: "
        + ", ".join(f"{C.CLASS_NAMES[c]}={n}" for c, n in dist.items())
    )

    # ── Stratified split ──────────────────────────────────────
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = split_data(
        x_all, y_all, train_frac=0.75, val_frac=0.10
    )
    return x_tr, y_tr, x_vl, y_vl, x_ts, y_ts


def fit_and_encode(x_tr, y_tr, x_vl, y_vl, x_ts, y_ts):
    """
    Fit PCA (n_components = C.PCA_COMPONENTS) on the raw training split,
    oversample training for class balance, then PCA-encode all three
    splits into angle-ready tensors.

    Order matters: PCA is fitted BEFORE oversampling so the components
    are not biased by repeated minority samples.
    """
    # ── PCA fitted on RAW training split BEFORE oversampling ──
    logging.info(f"Fitting PCA-{C.PCA_COMPONENTS} pipeline on training split...")
    scaler, pca, x_min, x_max = fit_pca_pipeline(x_tr)

    # ── Oversample training only ──────────────────────────────
    x_tr, y_tr = oversample_to_balance(x_tr, y_tr)

    for name, y in [("train", y_tr), ("val", y_vl), ("test", y_ts)]:
        u, c = np.unique(y, return_counts=True)
        tag  = "(oversampled)" if name == "train" else "(natural)"
        logging.info(
            f"{name:5s}: {len(y):5d} samples | "
            f"dist: { {int(k): int(v) for k, v in zip(u, c)} } {tag}"
        )

    # ── Encode every split ────────────────────────────────────
    x_train_t, y_train_t = apply_pca_pipeline(x_tr, y_tr, scaler, pca, x_min, x_max)
    x_val_t,   y_val_t   = apply_pca_pipeline(x_vl, y_vl, scaler, pca, x_min, x_max)
    x_test_t,  y_test_t  = apply_pca_pipeline(x_ts, y_ts, scaler, pca, x_min, x_max)

    logging.info(
        f"PCA tensors — train: {tuple(x_train_t.shape)} | "
        f"val: {tuple(x_val_t.shape)} | test: {tuple(x_test_t.shape)}"
    )
    if C.is_dense_encoding():
        logging.info(
            f"Encoding: {C.PCA_COMPONENTS} features → {C.N_QUBITS} qubits "
            f"(components 0–{C.N_QUBITS-1} as RX, "
            f"{C.N_QUBITS}–{C.PCA_COMPONENTS-1} as RY)"
        )
    else:
        logging.info(
            f"Encoding: {C.PCA_COMPONENTS} features → {C.N_QUBITS} qubits "
            f"(1 feature/qubit as RY)"
        )

    return (
        x_train_t, y_train_t,
        x_val_t,   y_val_t,
        x_test_t,  y_test_t,
        scaler, pca, x_min, x_max,
    )


def _search_subset(x_train_t, y_train_t):
    """
    Optionally subsample the training set for the SEARCH phase only.
    Final training always uses the full set.  Set
    C.MAX_SEARCH_TRAIN = None to disable.
    """
    if C.MAX_SEARCH_TRAIN is None or len(x_train_t) <= C.MAX_SEARCH_TRAIN:
        return x_train_t, y_train_t
    perm = torch.randperm(len(x_train_t))[:C.MAX_SEARCH_TRAIN]
    logging.info(
        f"Subsampling search training set: "
        f"{len(x_train_t)} → {C.MAX_SEARCH_TRAIN}"
    )
    return x_train_t[perm], y_train_t[perm]


# ================================================================
# SINGLE RUN  (one PCA variant, one hyperparameter setting)
# ================================================================
def run_single(sweep_params, x_train_t, y_train_t,
               x_val_t, y_val_t, x_test_t, y_test_t,
               scaler, pca, x_min, x_max, sweep_base: str,
               resume_dir: str = None):

    # Run N must not see motif IDs from run N-1
    reset_caches()

    label = sweep_params.get("label", "run")
    if resume_dir:
        run_dir = resume_dir
        setup_logging(run_dir)
        logging.info(f"\n{'='*60}")
        logging.info(f"RESUMING SWEEP RUN: {label}  (PCA-{C.PCA_COMPONENTS})  → {run_dir}")
        logging.info(f"{'='*60}")
    else:
        run_dir = make_run_dir(sweep_base, label=label)
        setup_logging(run_dir)
        logging.info(f"\n{'='*60}")
        logging.info(f"SWEEP RUN: {label}  (PCA-{C.PCA_COMPONENTS})")
        logging.info(f"Params: {sweep_params}")
        logging.info(f"{'='*60}")

    apply_sweep_params(sweep_params)

    run_config = {
        "label":              label,
        "dataset":            "dermamnist",
        "task":               "7-class (pigmented skin lesion type)",
        "feature_extraction": "pca",
        "pca_components":     C.PCA_COMPONENTS,
        "encoding":           ("dense (RX + RY per qubit)" if C.is_dense_encoding()
                               else "single (RY per qubit)"),
        "n_classes":          C.N_CLASSES,
        "class_names":        C.CLASS_NAMES,
        "W_CLASS":            C.W_CLASS,
        "N_QUBITS":           C.N_QUBITS,
        "N_READOUT_QUBITS":   C.N_READOUT_QUBITS,
        # variable-length structure bounds
        "MIN_LAYERS":         C.MIN_LAYERS,
        "MAX_LAYERS":         C.MAX_LAYERS,
        "MIN_CONV_LAYERS":    C.MIN_CONV_LAYERS,
        "MIN_POOL_LAYERS":    C.MIN_POOL_LAYERS,
        "P_CONV":             C.P_CONV,
        # search
        "LR_EVAL":            C.LR_EVAL,
        "N_EPOCHS_EVAL":      C.N_EPOCHS_EVAL,
        "BATCH_SIZE_EVAL":    C.BATCH_SIZE_EVAL,
        "INIT_POP":           C.INIT_POP,
        "MAX_EVO_STEPS":      C.MAX_EVO_STEPS,
        "L1":                 C.L1,
        "L2":                 C.L2,
        "L3":                 C.L3,
        "MIN_RECALL_TARGET":  C.MIN_RECALL_TARGET,
        # final training
        "FINAL_N_EPOCHS":     C.FINAL_N_EPOCHS,
        "FINAL_LR":           C.FINAL_LR,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # ── Linear baseline on the same features ──────────────────
    baseline = run_linear_baseline(
        x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t,
        exp_dir=run_dir,
    )

    # ── Evolutionary search ───────────────────────────────────
    x_search, y_search = _search_subset(x_train_t, y_train_t)
    best_motif, best_info, _ = run_evolution(
        x_search, y_search, x_val_t, y_val_t, exp_dir=run_dir,
        resume=bool(resume_dir),
    )

    # ── Final training on the FULL training set ───────────────
    best_params, test_metrics = final_train_and_test(
        best_motif,
        x_train_t, y_train_t,
        x_val_t,   y_val_t,
        x_test_t,  y_test_t,
        exp_dir=run_dir,
    )
    save_best_circuit(best_motif, best_info, best_params, exp_dir=run_dir)
    save_pipeline(scaler, pca, x_min, x_max,
                  os.path.join(run_dir, "pca_pipeline.pkl"))

    n_conv, n_pool, n_total = count_layers(best_motif)
    logging.info(f"\nRun '{label}' complete → {test_metrics}")

    result = {
        "label":          label,
        "pca_components": C.PCA_COMPONENTS,
        "run_dir":        run_dir,
        # architecture found
        "n_layers":       n_total,
        "n_conv":         n_conv,
        "n_pool":         n_pool,
        "signature":      layer_signature(best_motif),
        "n_params":       best_info.n_params,
        # validation (from search)
        "val_acc":        best_info.val_acc,
        "val_fitness":    best_info.fitness,
        "val_bal_acc":    best_info.bal_acc,
        "val_min_recall": best_info.min_recall,
        # test (from final training)
        "test_acc":       test_metrics["accuracy"],
        "test_bal_acc":   test_metrics["bal_acc"],
        "test_min_recall":min(test_metrics["per_class_recalls"]),
        "test_roc_auc":   test_metrics["roc_auc_ovr"],
        # classical reference
        "lin_test_acc":     baseline["test"]["accuracy"],
        "lin_test_bal_acc": baseline["test"]["bal_acc"],
        "lin_test_auc":     baseline["test"]["roc_auc_ovr"],
    }
    for c in range(C.N_CLASSES):
        result[f"test_recall_{c}"] = test_metrics["per_class_recalls"][c]
    result.update({k: v for k, v in sweep_params.items() if k != "label"})
    return result


# ================================================================
# RESUME: pick up an interrupted sweep run from its directory
# (e.g. experiments/pca8/sweep/baseline_001) and continue with
# whatever SWEEP_GRID entries come after it.
# ================================================================
def resume_variant(resume_dir: str, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts):
    config_path = os.path.join(resume_dir, "config.json")
    with open(config_path) as f:
        run_config = json.load(f)

    label = run_config.get("label")
    if label is None:
        raise ValueError(
            f"{resume_dir} has no 'label' in config.json — it isn't a "
            "sweep run, so --resume doesn't know how to continue it."
        )

    pca_components = run_config["pca_components"]
    C.PCA_COMPONENTS = pca_components
    C.assert_valid_pca_components()

    variant_base = os.path.join("experiments", f"pca{pca_components}")
    os.makedirs(variant_base, exist_ok=True)

    (x_train_t, y_train_t,
     x_val_t,   y_val_t,
     x_test_t,  y_test_t,
     scaler, pca, x_min, x_max) = fit_and_encode(x_tr, y_tr, x_vl, y_vl, x_ts, y_ts)

    grid = C.SWEEP_GRID
    start_idx = next((i for i, g in enumerate(grid) if g["label"] == label), None)
    if start_idx is None:
        raise ValueError(
            f"Label '{label}' from {resume_dir} not found in the current "
            "SWEEP_GRID — has config.py changed since this run started?"
        )

    sweep_base = os.path.join(variant_base, "sweep")
    os.makedirs(sweep_base, exist_ok=True)

    logging.info(
        f"Resuming PCA-{pca_components} sweep at entry {start_idx + 1}/{len(grid)} "
        f"('{label}') → {resume_dir}"
    )

    all_results = []
    for i, sweep_params in enumerate(grid):
        if i < start_idx:
            logging.info(f"Skipping entry {i + 1}/{len(grid)} ('{sweep_params['label']}') "
                         "— already completed before this run was interrupted")
            continue
        logging.info(
            f"\n>>> PCA-{pca_components} sweep run {i + 1}/{len(grid)}: "
            f"{sweep_params['label']}"
        )
        result = run_single(
            sweep_params,
            x_train_t, y_train_t,
            x_val_t,   y_val_t,
            x_test_t,  y_test_t,
            scaler, pca, x_min, x_max,
            sweep_base=sweep_base,
            resume_dir=resume_dir if i == start_idx else None,
        )
        all_results.append(result)

    summary_path = os.path.join(sweep_base, f"sweep_summary_resumed_at_{label}.csv")
    fieldnames   = list(all_results[0].keys())
    for r in all_results[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logging.info(f"\nResumed sweep summary saved → {summary_path}")

    ranked = sorted(all_results, key=lambda r: r["val_fitness"])
    print(f"\n===== PCA-{pca_components} SWEEP RESULTS resumed from '{label}' (ranked by val fitness) =====")
    header = (f"{'label':<18} {'structure':<16} {'par':>4} "
              f"{'val_fit':>8} {'tst_acc':>8} {'tst_bal':>8} "
              f"{'min_rec':>8} {'tst_auc':>8} {'lin_bal':>8}")
    print(header)
    print("-" * len(header))
    for r in ranked:
        sig = r["signature"]
        if len(sig) > 15:
            sig = sig[:14] + "…"
        print(
            f"{r['label']:<18} {sig:<16} {r['n_params']:>4} "
            f"{r['val_fitness']:>8.4f} {r['test_acc']:>8.4f} "
            f"{r['test_bal_acc']:>8.4f} {r['test_min_recall']:>8.4f} "
            f"{r['test_roc_auc']:>8.4f} {r['lin_test_bal_acc']:>8.4f}"
        )
    print("=" * len(header) + "\n")
    return all_results


# ================================================================
# ONE PCA VARIANT: single run or full sweep, using its own
# experiments/pcaN/ directory so PCA-8 and PCA-16 outputs never collide
# ================================================================
def run_variant(pca_components: int, args, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts):
    C.PCA_COMPONENTS = pca_components
    C.assert_valid_pca_components()

    variant_base = os.path.join("experiments", f"pca{pca_components}")
    os.makedirs(variant_base, exist_ok=True)

    (x_train_t, y_train_t,
     x_val_t,   y_val_t,
     x_test_t,  y_test_t,
     scaler, pca, x_min, x_max) = fit_and_encode(x_tr, y_tr, x_vl, y_vl, x_ts, y_ts)

    # ================================================================
    # SWEEP MODE
    # ================================================================
    if args.sweep or args.sweep_index is not None:
        grid = C.SWEEP_GRID
        if args.sweep_index is not None:
            grid = [grid[args.sweep_index]]

        sweep_base = os.path.join(variant_base, "sweep")
        os.makedirs(sweep_base, exist_ok=True)

        all_results = []
        for i, sweep_params in enumerate(grid):
            logging.info(
                f"\n>>> PCA-{pca_components} sweep run {i + 1}/{len(grid)}: "
                f"{sweep_params['label']}"
            )
            result = run_single(
                sweep_params,
                x_train_t, y_train_t,
                x_val_t,   y_val_t,
                x_test_t,  y_test_t,
                scaler, pca, x_min, x_max,
                sweep_base=sweep_base,
            )
            all_results.append(result)

        # ── Sweep summary CSV ─────────────────────────────────
        summary_path = os.path.join(sweep_base, "sweep_summary.csv")
        fieldnames   = list(all_results[0].keys())
        for r in all_results[1:]:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logging.info(f"\nSweep summary saved → {summary_path}")

        ranked = sorted(all_results, key=lambda r: r["val_fitness"])
        print(f"\n===== PCA-{pca_components} SWEEP RESULTS (ranked by val fitness) =====")
        header = (f"{'label':<18} {'structure':<16} {'par':>4} "
                  f"{'val_fit':>8} {'tst_acc':>8} {'tst_bal':>8} "
                  f"{'min_rec':>8} {'tst_auc':>8} {'lin_bal':>8}")
        print(header)
        print("-" * len(header))
        for r in ranked:
            sig = r["signature"]
            if len(sig) > 15:
                sig = sig[:14] + "…"
            print(
                f"{r['label']:<18} {sig:<16} {r['n_params']:>4} "
                f"{r['val_fitness']:>8.4f} {r['test_acc']:>8.4f} "
                f"{r['test_bal_acc']:>8.4f} {r['test_min_recall']:>8.4f} "
                f"{r['test_roc_auc']:>8.4f} {r['lin_test_bal_acc']:>8.4f}"
            )
        print("=" * len(header) + "\n")
        return all_results

    # ================================================================
    # SINGLE RUN MODE
    # ================================================================
    else:
        RUN_DIR = make_run_dir(variant_base, label="run")
        setup_logging(RUN_DIR)
        reset_caches()

        with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
            json.dump({
                "dataset":            "dermamnist",
                "task":               "7-class (pigmented skin lesion type)",
                "feature_extraction": "pca",
                "pca_components":     C.PCA_COMPONENTS,
                "encoding":           ("dense (RX + RY per qubit)" if C.is_dense_encoding()
                                       else "single (RY per qubit)"),
                "n_classes":          C.N_CLASSES,
                "class_names":        C.CLASS_NAMES,
                "W_CLASS":            C.W_CLASS,
                "N_QUBITS":           C.N_QUBITS,
                "N_READOUT_QUBITS":   C.N_READOUT_QUBITS,
                "MIN_LAYERS":         C.MIN_LAYERS,
                "MAX_LAYERS":         C.MAX_LAYERS,
                "MIN_CONV_LAYERS":    C.MIN_CONV_LAYERS,
                "MIN_POOL_LAYERS":    C.MIN_POOL_LAYERS,
                "P_CONV":             C.P_CONV,
                "LR_EVAL":            C.LR_EVAL,
                "N_EPOCHS_EVAL":      C.N_EPOCHS_EVAL,
                "BATCH_SIZE_EVAL":    C.BATCH_SIZE_EVAL,
                "INIT_POP":           C.INIT_POP,
                "MAX_EVO_STEPS":      C.MAX_EVO_STEPS,
                "L1":                 C.L1,
                "L2":                 C.L2,
                "L3":                 C.L3,
                "MIN_RECALL_TARGET":  C.MIN_RECALL_TARGET,
                "FINAL_N_EPOCHS":     C.FINAL_N_EPOCHS,
                "FINAL_LR":           C.FINAL_LR,
            }, f, indent=2)

        # ── Linear baseline ───────────────────────────────────
        baseline = run_linear_baseline(
            x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t,
            exp_dir=RUN_DIR,
        )

        # ── Evolutionary search ───────────────────────────────
        x_search, y_search = _search_subset(x_train_t, y_train_t)
        best_motif, best_info, _ = run_evolution(
            x_search, y_search, x_val_t, y_val_t, exp_dir=RUN_DIR,
        )

        # ── Final training on the FULL training set ───────────
        best_params, test_metrics = final_train_and_test(
            best_motif,
            x_train_t, y_train_t,
            x_val_t,   y_val_t,
            x_test_t,  y_test_t,
            exp_dir=RUN_DIR,
        )
        save_best_circuit(best_motif, best_info, best_params, exp_dir=RUN_DIR)
        save_pipeline(scaler, pca, x_min, x_max,
                      os.path.join(RUN_DIR, "pca_pipeline.pkl"))

        n_conv, n_pool, n_total = count_layers(best_motif)
        recalls = test_metrics["per_class_recalls"]

        print("\n" + "=" * 56)
        print(f"FINAL RESULTS — DermaMNIST 7-class — PCA-{pca_components}")
        print("=" * 56)
        print(f"  Architecture found : {layer_signature(best_motif)}")
        print(f"  Layers             : {n_total} ({n_conv} conv + {n_pool} pool)")
        print(f"  Parameters         : {best_info.n_params}")
        enc = "dense" if C.is_dense_encoding() else "single"
        print(f"  Encoding           : {C.PCA_COMPONENTS} PCA → "
              f"{C.N_QUBITS} qubits ({enc})")
        print("-" * 56)
        print("  QCNN (test set)")
        print(f"    accuracy         : {test_metrics['accuracy']:.4f}")
        print(f"    balanced acc     : {test_metrics['bal_acc']:.4f}")
        print(f"    min recall       : {min(recalls):.4f}")
        print(f"    ROC-AUC (OvR)    : {test_metrics['roc_auc_ovr']:.4f}")
        print(f"    per-class recall : {[f'{r:.3f}' for r in recalls]}")
        print("-" * 56)
        print(f"  Logistic regression on same {pca_components} PCA features (test set)")
        print(f"    accuracy         : {baseline['test']['accuracy']:.4f}")
        print(f"    balanced acc     : {baseline['test']['bal_acc']:.4f}")
        print(f"    min recall       : {baseline['test']['min_recall']:.4f}")
        print(f"    ROC-AUC (OvR)    : {baseline['test']['roc_auc_ovr']:.4f}")
        print("=" * 56 + "\n")

        return None


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evolutionary QCNN architecture search — DermaMNIST"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Run every entry in SWEEP_GRID")
    parser.add_argument("--sweep-index", type=int, default=None,
                        help="Run one SWEEP_GRID entry by 0-based index")
    parser.add_argument("--variant", choices=["8", "16", "both"], default="both",
                        help="Which PCA encoding(s) to run (default: both)")
    parser.add_argument("--serial", action="store_true",
                        help="With --variant both, run PCA-8 then PCA-16 "
                             "sequentially in this one process instead of "
                             "as two parallel subprocesses (default: parallel)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume an interrupted sweep run from its directory "
                             "(e.g. experiments/pca8/sweep/baseline_001) and "
                             "continue with the remaining SWEEP_GRID entries")
    args = parser.parse_args()

    # ── PCA-8 and PCA-16 are fully independent (separate experiments/pcaN/
    # output dirs, separate Ray clusters, no shared state) — for --variant
    # both, launch them as two parallel subprocesses by default instead of
    # running them back-to-back in one process. Pass --serial to opt back
    # into the old single-process, sequential behaviour. --resume always
    # targets one specific run directory, so it skips this split.
    if args.variant == "both" and not args.serial and not args.resume:
        import subprocess
        import sys

        cmd_base = [sys.executable, __file__]
        if args.sweep:
            cmd_base.append("--sweep")
        if args.sweep_index is not None:
            cmd_base += ["--sweep-index", str(args.sweep_index)]

        procs = []
        for v in ("8", "16"):
            cmd = cmd_base + ["--variant", v]
            print(f"Launching PCA-{v} as a subprocess: {' '.join(cmd)}")
            procs.append((v, subprocess.Popen(cmd)))

        exit_code = 0
        for v, p in procs:
            ret = p.wait()
            print(f"PCA-{v} subprocess exited with code {ret}")
            exit_code = exit_code or ret

        raise SystemExit(exit_code)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ray.init(num_gpus=0)

    # ── Load raw data once — same split feeds every PCA variant ────
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = load_raw_split()

    if args.resume:
        resume_variant(args.resume, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts)
    else:
        variants = C.PCA_VARIANTS if args.variant == "both" else [int(args.variant)]
        for pca_components in variants:
            run_variant(pca_components, args, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts)

    ray.shutdown()
