#!/usr/bin/env python3
# ----------------------------------------------------------------
# winner.py  —  Extract and retrain any circuit from a completed
#               (or interrupted) DermaMNIST search run.
#
# HOW TO USE
# ----------
# 1. Set RUN_DIR to the experiment folder you want to inspect.
# 2. Choose a SELECTION_MODE and set its corresponding value.
# 3. Tune the training hyperparameters in the TRAINING CONFIG block.
# 4. Run:  python winner.py
#
# SELECTION MODES
# ---------------
#   "list"      →  print every circuit in the cache and exit (no training)
#   "best"      →  best fitness from memory_table.pkl
#   "rank"      →  Nth best (1-indexed) from memory_table.pkl
#   "params"    →  first circuit with exactly SELECT_PARAMS parameters
#   "id"        →  circuit whose motif_id starts with SELECT_ID_PREFIX
#   "layers"    →  first circuit with exactly SELECT_N_LAYERS layers
#   "signature" →  first circuit matching SELECT_SIGNATURE, e.g. "C-P-C-C"
#
# Start with "list" to see what the search actually found.
# ----------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# See main.py for why: sklearn PCA's randomized-SVD path can deadlock
# in OpenBLAS's lazy thread-pool init the first time it runs after
# ray.init() has pre-forked worker processes. Must be set before numpy
# is imported anywhere in this process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import os
import csv
import json
import logging
import shelve

import dill
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    confusion_matrix, roc_auc_score,
)
from torch.optim.lr_scheduler import StepLR

import config as C
from circuit import build_qcnn_qnode, get_gate_applications
from evolution import (
    _per_class_recalls, _weighted_fitness,
    count_layers, layer_signature, log_circuit_structure,
)
from gates import MAPPING_DICT
from main import split_data, oversample_to_balance
from preprocessing import (
    fit_pca_pipeline, apply_pca_pipeline, load_pipeline,
)
from serialisation import motif_from_dict

# ================================================================
# ── CONFIGURE THESE ─────────────────────────────────────────────
# ================================================================

RUN_DIR    = "experiments/run_001"   # run to inspect
OUTPUT_DIR = "experiments/winner"    # where to save retraining results

# ── Selection ────────────────────────────────────────────────────
SELECTION_MODE   = "best"   # list | best | rank | params | id | layers | signature
SELECT_RANK      = 1        # for "rank"      (1 = best)
SELECT_PARAMS    = None     # for "params"    e.g. 43
SELECT_ID_PREFIX = None     # for "id"        e.g. "a3f9"
SELECT_N_LAYERS  = None     # for "layers"    e.g. 4
SELECT_SIGNATURE = None     # for "signature" e.g. "C-P-C-C"

# ── Training hyperparameters ─────────────────────────────────────
N_EPOCHS   = 200
LR         = 0.005
LR_STEP    = 50       # halve LR every this many epochs
LR_GAMMA   = 0.5
BATCH_SIZE = 25

# ── Clinical weights for the val-fitness curve ───────────────────
# Does not affect which circuit is selected, only the tracked curve.
W_CLASS = C.W_CLASS

# ================================================================
# (nothing below normally needs changing)
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "winner.log")),
        logging.StreamHandler(),
    ],
)

# ================================================================
# LOAD CACHE AND MEMORY TABLE
# ================================================================
def load_cache(run_dir: str) -> dict:
    """Load every motif the search evaluated, from the shelve database."""
    cache_path = os.path.join(run_dir, "motif_cache.db")
    if not (os.path.exists(cache_path) or os.path.exists(cache_path + ".db")):
        raise FileNotFoundError(
            f"No motif cache found at {cache_path}. "
            "Check that RUN_DIR points to a valid experiment folder."
        )
    with shelve.open(cache_path, flag="r") as db:
        return dict(db)


def load_memory_table(run_dir: str):
    """
    Load the memory table if present.

    Returns None if the run was interrupted before the first save — the
    motif cache is still usable, but fitness and rank info is not.
    """
    pkl_path = os.path.join(run_dir, "memory_table.pkl")
    if not os.path.exists(pkl_path):
        logging.warning(
            "memory_table.pkl not found — run may have been interrupted. "
            "Fitness/rank unavailable; use 'params', 'layers' or "
            "'signature' selection mode."
        )
        return None
    with open(pkl_path, "rb") as f:
        return dill.load(f)

# ================================================================
# CIRCUIT SELECTION
# ================================================================
def _describe(motif):
    """Return (n_params, signature, n_layers) for a motif."""
    _, n_params, _ = build_qcnn_qnode(motif)
    n_conv, n_pool, n_total = count_layers(motif)
    return n_params, layer_signature(motif), n_total


def select_circuit(cache: dict, memory_table, mode: str):
    """Return (motif_id, motif, info_or_None) for the configured mode."""

    # ── list ──────────────────────────────────────────────────
    if mode == "list":
        print("\n" + "=" * 92)
        print(f"Circuits in cache: {len(cache)}")
        print("=" * 92)

        if memory_table:
            ranked = sorted(memory_table.values(), key=lambda x: x.fitness)
            print(f"{'Rank':<5} {'ID':<11} {'structure':<18} {'lyr':>4} "
                  f"{'par':>4} {'fitness':>9} {'acc':>7} {'bal':>7} {'minrec':>7}")
            print("-" * 92)
            for rank, info in enumerate(ranked, 1):
                mid = info.motif_id
                if mid not in cache:
                    continue
                motif = motif_from_dict(cache[mid], MAPPING_DICT)
                try:
                    n_params, sig, n_layers = _describe(motif)
                except Exception:
                    n_params, sig, n_layers = -1, "?", -1
                if len(sig) > 17:
                    sig = sig[:16] + "…"
                print(f"{rank:<5} {mid[:9]:<11} {sig:<18} {n_layers:>4} "
                      f"{n_params:>4} {info.fitness:>9.4f} "
                      f"{info.val_acc:>7.4f} {info.bal_acc:>7.4f} "
                      f"{info.min_recall:>7.4f}")
        else:
            print(f"{'ID':<11} {'structure':<18} {'lyr':>4} {'par':>4}")
            print("-" * 42)
            for mid, motif_dict in cache.items():
                motif = motif_from_dict(motif_dict, MAPPING_DICT)
                try:
                    n_params, sig, n_layers = _describe(motif)
                except Exception:
                    continue
                if len(sig) > 17:
                    sig = sig[:16] + "…"
                print(f"{mid[:9]:<11} {sig:<18} {n_layers:>4} {n_params:>4}")
        print("=" * 92)
        return None, None, None

    # ── best / rank ───────────────────────────────────────────
    if mode in ("best", "rank"):
        if memory_table is None:
            raise RuntimeError(
                f"Mode '{mode}' requires memory_table.pkl. Use 'params', "
                "'layers' or 'signature' instead."
            )
        ranked = sorted(memory_table.values(), key=lambda x: x.fitness)
        rank   = SELECT_RANK if mode == "rank" else 1
        if rank > len(ranked):
            raise ValueError(
                f"Rank {rank} requested but only {len(ranked)} circuits exist."
            )
        info  = ranked[rank - 1]
        motif = motif_from_dict(cache[info.motif_id], MAPPING_DICT)
        logging.info(
            f"Selected rank {rank}: id={info.motif_id[:10]}  "
            f"fitness={info.fitness:.4f}  params={info.n_params}  "
            f"[{layer_signature(motif)}]"
        )
        return info.motif_id, motif, info

    # ── attribute-matching modes ──────────────────────────────
    for mid, motif_dict in cache.items():
        motif = motif_from_dict(motif_dict, MAPPING_DICT)
        try:
            n_params, sig, n_layers = _describe(motif)
        except Exception:
            continue

        hit = (
            (mode == "params"    and n_params == SELECT_PARAMS) or
            (mode == "id"        and mid.startswith(SELECT_ID_PREFIX or "\0")) or
            (mode == "layers"    and n_layers == SELECT_N_LAYERS) or
            (mode == "signature" and sig == SELECT_SIGNATURE)
        )
        if hit:
            info = memory_table.get(mid) if memory_table else None
            logging.info(
                f"Selected by {mode}: id={mid[:10]}  [{sig}]  "
                f"params={n_params}"
                + (f"  fitness={info.fitness:.4f}" if info else "")
            )
            return mid, motif, info

    raise ValueError(
        f"No circuit matched selection mode '{mode}'. "
        "Run with SELECTION_MODE='list' to see what is available."
    )

# ================================================================
# DATA LOADING
# ================================================================
def load_data():
    """
    Rebuild the exact data the search used.

    If the run directory contains pca_pipeline.pkl the saved scaler and
    PCA are reused, guaranteeing identical features.  Otherwise the
    pipeline is refitted from scratch using the same fixed seed.
    """
    logging.info("Loading DermaMNIST...")
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

    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = split_data(x_all, y_all)

    pipeline_path = os.path.join(RUN_DIR, "pca_pipeline.pkl")
    if os.path.exists(pipeline_path):
        logging.info(f"Reusing saved PCA pipeline from {pipeline_path}")
        scaler, pca, x_min, x_max = load_pipeline(pipeline_path)
    else:
        logging.info("No saved pipeline — refitting PCA (same seed).")
        scaler, pca, x_min, x_max = fit_pca_pipeline(x_tr)

    x_tr, y_tr = oversample_to_balance(x_tr, y_tr)

    x_train_t, y_train_t = apply_pca_pipeline(x_tr, y_tr, scaler, pca, x_min, x_max)
    x_val_t,   y_val_t   = apply_pca_pipeline(x_vl, y_vl, scaler, pca, x_min, x_max)
    x_test_t,  y_test_t  = apply_pca_pipeline(x_ts, y_ts, scaler, pca, x_min, x_max)

    logging.info(
        f"Data ready — train: {tuple(x_train_t.shape)}  "
        f"val: {tuple(x_val_t.shape)}  test: {tuple(x_test_t.shape)}"
    )
    return x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t

# ================================================================
# TRAINING LOOP
# ================================================================
def train_circuit(motif, x_train, y_train, x_val, y_val, x_test, y_test,
                  out_dir: str):
    """Train the selected circuit and produce a full evaluation report."""
    os.makedirs(out_dir, exist_ok=True)

    circuit, n_params, raw_qnode = build_qcnn_qnode(motif, n_qubits=C.N_QUBITS)
    n_conv, n_pool, n_total = count_layers(motif)

    logging.info("\nCircuit structure:")
    log_circuit_structure(motif)
    logging.info(
        f"Parameters: {n_params} | Layers: {n_total} "
        f"({n_conv} conv + {n_pool} pool)"
    )
    logging.info(
        f"Training config: epochs={N_EPOCHS}  lr={LR}  batch={BATCH_SIZE}  "
        f"lr_step={LR_STEP}  lr_gamma={LR_GAMMA}"
    )

    if n_params == 0:
        logging.warning("Circuit has 0 parameters — no training possible.")
        return

    params    = torch.rand(n_params, requires_grad=True)
    opt       = torch.optim.Adam([params], lr=LR)
    scheduler = StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)

    n                = len(x_train)
    train_losses     = []
    val_fitnesses    = []
    best_val_fitness = float("inf")
    params_best      = params.detach().clone()

    recall_cols = [f"val_recall_{c}" for c in range(C.N_CLASSES)]
    csv_path = os.path.join(out_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_fitness", "val_acc", "val_bal_acc"]
            + recall_cols + ["lr"]
        )

        for epoch in range(1, N_EPOCHS + 1):
            perm   = torch.randperm(n)
            x_shuf = x_train[perm]
            y_shuf = y_train[perm]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, BATCH_SIZE):
                x_b = x_shuf[start: start + BATCH_SIZE]
                y_b = y_shuf[start: start + BATCH_SIZE]
                opt.zero_grad()
                probs_b = circuit(x_b, params).float()
                loss    = F.nll_loss(torch.log(probs_b + 1e-8), y_b)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss   = epoch_loss / max(n_batches, 1)
            current_lr = scheduler.get_last_lr()[0]
            train_losses.append(avg_loss)

            with torch.no_grad():
                probs_val = circuit(x_val, params).float()
            preds_np  = torch.argmax(probs_val, dim=1).numpy()
            labels_np = y_val.numpy()
            val_acc   = float(np.mean(preds_np == labels_np))
            recalls   = _per_class_recalls(preds_np, labels_np)
            val_bal   = float(np.mean(recalls))
            val_fitness = _weighted_fitness(
                recalls, 0, 0, W_CLASS, C.N_QUBITS, 0, 0, 0, 0
            )
            val_fitnesses.append(val_fitness)

            writer.writerow(
                [epoch, avg_loss, val_fitness, val_acc, val_bal]
                + list(recalls) + [current_lr]
            )

            if val_fitness < best_val_fitness:
                best_val_fitness = val_fitness
                params_best      = params.detach().clone()

            if epoch % 20 == 0 or epoch == 1:
                logging.info(
                    f"Epoch {epoch:4d}/{N_EPOCHS} | loss={avg_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | bal={val_bal:.4f} | "
                    f"min_rec={min(recalls):.3f} | lr={current_lr:.6f}"
                )

    logging.info(f"Training log → {csv_path}")
    logging.info(f"Best val fitness: {best_val_fitness:.4f}")

    # ── Test evaluation ───────────────────────────────────────
    logging.info("\n--- Test evaluation (best-val checkpoint) ---")
    with torch.no_grad():
        probs_test = circuit(x_test, params_best).float()
    preds_test = torch.argmax(probs_test, dim=1)
    preds_np   = preds_test.numpy()
    labels_np  = y_test.numpy()

    test_acc     = float(np.mean(preds_np == labels_np))
    test_recalls = _per_class_recalls(preds_np, labels_np)
    test_bal     = float(np.mean(test_recalls))

    try:
        y_bin = np.eye(C.N_CLASSES)[labels_np]
        auc   = float(roc_auc_score(y_bin, probs_test.numpy(),
                                    multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    logging.info(
        f"Test accuracy    : {test_acc:.4f}\n"
        f"Test bal. acc    : {test_bal:.4f}\n"
        f"Per-class recall : {[f'{r:.4f}' for r in test_recalls]}\n"
        f"Min recall       : {min(test_recalls):.4f}\n"
        f"ROC-AUC OvR      : {auc:.4f}"
    )
    logging.info("\n" + classification_report(
        labels_np, preds_np, target_names=C.CLASS_NAMES
    ))

    # ── Save results ──────────────────────────────────────────
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "n_params":         n_params,
            "n_layers":         n_total,
            "n_conv_layers":    n_conv,
            "n_pool_layers":    n_pool,
            "layer_signature":  layer_signature(motif),
            "n_epochs":         N_EPOCHS,
            "lr":               LR,
            "lr_step":          LR_STEP,
            "lr_gamma":         LR_GAMMA,
            "batch_size":       BATCH_SIZE,
            "best_val_fitness": best_val_fitness,
            "test_accuracy":    test_acc,
            "test_bal_acc":     test_bal,
            "test_per_class":   test_recalls,
            "test_min_recall":  min(test_recalls),
            "test_roc_auc_ovr": auc,
        }, f, indent=2)

    torch.save(params_best, os.path.join(out_dir, "trained_params.pt"))

    # ── Training curves ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set(xlabel="Epoch", ylabel="NLL loss", title="Training loss")
    axes[1].plot(val_fitnesses, color="orange", label="Val fitness")
    axes[1].axhline(best_val_fitness, linestyle="--", color="red",
                    label="Best checkpoint")
    axes[1].set(xlabel="Epoch", ylabel="Fitness (lower = better)",
                title="Validation fitness")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(labels_np, preds_np)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASS_NAMES).plot(
        ax=ax, colorbar=False, xticks_rotation=45
    )
    ax.set_title(f"acc={test_acc:.3f}  bal_acc={test_bal:.3f}  AUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── Per-class recall bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = [plt.get_cmap("tab10")(i % 10) for i in range(C.N_CLASSES)]
    bars = ax.bar(C.CLASS_NAMES, test_recalls, color=colours)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class recall on test set")
    ax.tick_params(axis="x", rotation=30)
    for bar, r in zip(bars, test_recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02,
                f"{r:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_recall.png"), dpi=150)
    plt.close()

    # ── Circuit diagram ───────────────────────────────────────
    try:
        fig_c, _ = qml.draw_mpl(raw_qnode, decimals=2, style="pennylane")(
            x_test[0], params_best
        )
        fig_c.suptitle(
            f"Extracted QCNN  [{layer_signature(motif)}]", fontsize=11, y=1.01
        )
        fig_c.tight_layout()
        fig_c.savefig(os.path.join(out_dir, "circuit.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_c)
    except Exception as e:
        logging.warning(f"Circuit diagram failed: {e}")

    logging.info(f"\nAll results saved → {out_dir}/")

    print("\n" + "=" * 50)
    print("WINNER RESULTS")
    print("=" * 50)
    print(f"  Structure        : {layer_signature(motif)}")
    print(f"  Layers           : {n_total} ({n_conv} conv + {n_pool} pool)")
    print(f"  Parameters       : {n_params}")
    print(f"  Best val fitness : {best_val_fitness:.4f}")
    print(f"  Test accuracy    : {test_acc:.4f}")
    print(f"  Test bal. acc    : {test_bal:.4f}")
    print(f"  Test min recall  : {min(test_recalls):.4f}")
    print(f"  Test ROC-AUC     : {auc:.4f}")
    print(f"  Per-class recall : {[f'{r:.3f}' for r in test_recalls]}")
    print("=" * 50 + "\n")

# ================================================================
# MAIN
# ================================================================
def _load_run_pca_components(run_dir: str) -> None:
    """
    RUN_DIR may hold a PCA-8 or a PCA-16 run — set C.PCA_COMPONENTS from
    its own config.json so build_qcnn_qnode() picks the matching
    encoding, rather than relying on whatever this module happened to
    default to.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        logging.warning(
            f"No config.json in {run_dir} — assuming PCA_COMPONENTS="
            f"{C.PCA_COMPONENTS} (this module's default). Set it manually "
            "if that's wrong."
        )
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    pca_components = cfg.get("pca_components")
    if pca_components is None:
        logging.warning(f"{cfg_path} has no 'pca_components' key; leaving "
                        f"C.PCA_COMPONENTS={C.PCA_COMPONENTS} unchanged.")
        return
    C.PCA_COMPONENTS = pca_components
    logging.info(f"PCA_COMPONENTS set from {cfg_path}: {pca_components} "
                f"({'dense' if C.is_dense_encoding() else 'single'} encoding)")


if __name__ == "__main__":
    logging.info(f"RUN_DIR    : {RUN_DIR}")
    logging.info(f"SELECTION  : {SELECTION_MODE}")
    logging.info(f"OUTPUT_DIR : {OUTPUT_DIR}")

    _load_run_pca_components(RUN_DIR)

    cache        = load_cache(RUN_DIR)
    memory_table = load_memory_table(RUN_DIR)

    logging.info(f"Cache contains {len(cache)} circuits.")
    if memory_table:
        logging.info(f"Memory table contains {len(memory_table)} entries.")

    mid, motif, search_info = select_circuit(cache, memory_table, SELECTION_MODE)

    if SELECTION_MODE == "list":
        raise SystemExit(0)

    if search_info is not None:
        logging.info("\n--- Search metrics (from the evolutionary run) ---")
        logging.info(f"  Fitness      : {search_info.fitness:.4f}")
        logging.info(f"  Val accuracy : {search_info.val_acc:.4f}")
        logging.info(f"  Balanced acc : {search_info.bal_acc:.4f}")
        logging.info(f"  Min recall   : {search_info.min_recall:.4f}")
        logging.info(f"  Generation   : {search_info.generation}")
        logging.info(f"  Evo step     : {search_info.evo_step}")
        logging.info(f"  Origin       : {search_info.mutation_type}")

    x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t = load_data()

    train_circuit(
        motif,
        x_train_t, y_train_t,
        x_val_t,   y_val_t,
        x_test_t,  y_test_t,
        out_dir=OUTPUT_DIR,
    )