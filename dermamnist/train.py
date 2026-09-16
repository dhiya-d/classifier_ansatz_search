# ----------------------------------------------------------------
# train.py  —  Final training loop, evaluation and artefact saving
#              (DermaMNIST 7-class, variable-length QCNN,
#               PCA-8 single or PCA-16 dense encoding)
# ----------------------------------------------------------------

import csv
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    confusion_matrix, roc_auc_score,
)
from torch.optim.lr_scheduler import StepLR

import pennylane as qml

import config as C
from circuit import build_qcnn_qnode
from evolution import (
    _per_class_recalls, _weighted_fitness,
    count_layers, layer_signature, log_circuit_structure,
)
from serialisation import motif_to_dict

# Qualitative palette for N_CLASSES bars/labels — tab10 comfortably
# covers up to 10 classes with visually distinct colours.
_CLASS_COLOURS = [plt.get_cmap("tab10")(i % 10) for i in range(C.N_CLASSES)]

# ================================================================
# METRIC HELPERS
# ================================================================
def _compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    Returns
    -------
    acc     : overall accuracy
    bal_acc : balanced accuracy = mean per-class recall
    recalls : list[float] of length N_CLASSES
    """
    preds_np  = preds.numpy()
    labels_np = labels.numpy()
    acc       = float(np.mean(preds_np == labels_np))
    recalls   = _per_class_recalls(preds_np, labels_np)
    bal_acc   = float(np.mean(recalls))
    return acc, bal_acc, recalls


def _run_inference(circuit, params, dataset: torch.Tensor) -> torch.Tensor:
    """Run circuit on every sample and return predicted class indices."""
    with torch.no_grad():
        probs = circuit(dataset, params).float()
    return torch.argmax(probs, dim=1)

# ================================================================
# FINAL TRAINING + EVALUATION
# ================================================================
def final_train_and_test(
    best_motif,
    x_train, y_train,
    x_val,   y_val,
    x_test,  y_test,
    exp_dir: str = None,
):
    """
    1. Build the QNode for the best evolved motif.
    2. Train on the full training set with LR schedule.
    3. Track validation fitness; checkpoint the best parameters.
    4. Evaluate that checkpoint on the test set.
    5. Save curves, confusion matrix, recall chart, circuit diagram.
    """
    if exp_dir is None:
        exp_dir = C.FINAL_EXP_DIR
    os.makedirs(exp_dir, exist_ok=True)

    logging.info("\n" + "=" * 60)
    logging.info("FINAL TRAINING — DermaMNIST 7-class QCNN")
    logging.info("=" * 60)
    log_circuit_structure(best_motif)

    circuit, n_params, raw_qnode = build_qcnn_qnode(
        best_motif, n_qubits=C.N_QUBITS
    )
    n_conv, n_pool, n_total = count_layers(best_motif)
    logging.info(
        f"Circuit: {n_total} layers ({n_conv} conv + {n_pool} pool), "
        f"{n_params} trainable parameters."
    )

    # ── Zero-parameter edge case ──────────────────────────────
    if n_params == 0:
        logging.warning("Best motif has 0 parameters — skipping training.")
        params = torch.tensor([])
        preds  = _run_inference(circuit, params, x_test)
        acc, bal_acc, recalls = _compute_metrics(preds, y_test)
        return params, {
            "accuracy": acc, "bal_acc": bal_acc,
            "per_class_recalls": recalls, "roc_auc_ovr": float("nan"),
        }

    # ── Initialise ────────────────────────────────────────────
    params    = torch.rand(n_params, requires_grad=True)
    opt       = torch.optim.Adam([params], lr=C.FINAL_LR)
    scheduler = StepLR(opt, step_size=C.FINAL_LR_STEP, gamma=C.FINAL_LR_GAMMA)

    n                = len(x_train)
    train_losses     = []
    val_fitnesses    = []
    best_val_fitness = float("inf")
    params_best      = params.detach().clone()

    # ── Training loop ─────────────────────────────────────────
    recall_cols = [f"val_recall_{c}" for c in range(C.N_CLASSES)]
    csv_path = os.path.join(exp_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["epoch", "train_loss", "val_fitness", "val_acc", "val_bal_acc"]
            + recall_cols + ["lr"]
        )

        for epoch in range(1, C.FINAL_N_EPOCHS + 1):
            perm   = torch.randperm(n)
            x_shuf = x_train[perm]
            y_shuf = y_train[perm]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, C.FINAL_BATCH_SIZE):
                x_b = x_shuf[start: start + C.FINAL_BATCH_SIZE]
                y_b = y_shuf[start: start + C.FINAL_BATCH_SIZE]
                opt.zero_grad()
                probs_b = circuit(x_b, params).float()
                # NLLLoss on log-probabilities: the circuit already
                # outputs normalised probabilities, so no softmax.
                loss = F.nll_loss(torch.log(probs_b + 1e-8), y_b)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss   = epoch_loss / max(n_batches, 1)
            current_lr = scheduler.get_last_lr()[0]
            train_losses.append(avg_loss)

            preds_val = _run_inference(circuit, params, x_val)
            val_acc, val_bal_acc, val_recalls = _compute_metrics(preds_val, y_val)

            # Complexity penalties zeroed: the architecture is fixed now,
            # so fitness here is purely the per-class recall term.
            val_fitness = _weighted_fitness(
                val_recalls, 0, 0, C.W_CLASS, C.N_QUBITS, 0, 0, 0, 0
            )
            val_fitnesses.append(val_fitness)

            writer.writerow(
                [epoch, avg_loss, val_fitness, val_acc, val_bal_acc]
                + list(val_recalls) + [current_lr]
            )

            if val_fitness < best_val_fitness:
                best_val_fitness = val_fitness
                params_best      = params.detach().clone()

            if epoch % 20 == 0 or epoch == 1:
                logging.info(
                    f"Epoch {epoch:4d}/{C.FINAL_N_EPOCHS} | "
                    f"loss={avg_loss:.4f} | val_acc={val_acc:.4f} | "
                    f"bal_acc={val_bal_acc:.4f} | "
                    f"min_recall={min(val_recalls):.3f} | "
                    f"lr={current_lr:.6f}"
                )

    logging.info(f"Training log saved → {csv_path}")
    logging.info(f"Best val fitness: {best_val_fitness:.4f}")

    # ── Test evaluation (best-val checkpoint) ─────────────────
    logging.info("\n--- Test evaluation (best-val checkpoint) ---")
    preds_test = _run_inference(circuit, params_best, x_test)
    test_acc, test_bal_acc, test_recalls = _compute_metrics(preds_test, y_test)

    with torch.no_grad():
        probs_test = circuit(x_test, params_best).float().numpy()

    try:
        y_test_bin = np.eye(C.N_CLASSES)[y_test.numpy()]
        auc = float(roc_auc_score(y_test_bin, probs_test,
                                  multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    logging.info(
        f"Test accuracy    : {test_acc:.4f}\n"
        f"Test bal. acc    : {test_bal_acc:.4f}  (mean per-class recall)\n"
        f"Per-class recall : {[f'{r:.4f}' for r in test_recalls]}\n"
        f"Min recall       : {min(test_recalls):.4f}\n"
        f"Test ROC-AUC OvR : {auc:.4f}"
    )
    logging.info("\n" + classification_report(
        y_test.numpy(), preds_test.numpy(),
        target_names=C.CLASS_NAMES
    ))

    # ── Training curves ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="Train loss")
    axes[0].set(xlabel="Epoch", ylabel="NLL loss", title="Training loss")
    axes[0].legend()
    axes[1].plot(val_fitnesses, label="Val fitness", color="orange")
    axes[1].axhline(best_val_fitness, linestyle="--", color="red",
                    label="Best checkpoint")
    axes[1].set(xlabel="Epoch", ylabel="Fitness (lower = better)",
                title="Validation fitness")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"), dpi=150)
    plt.close()

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test.numpy(), preds_test.numpy())
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASS_NAMES).plot(
        ax=ax, colorbar=False, xticks_rotation=45
    )
    ax.set_title(
        f"Test | acc={test_acc:.3f}  bal_acc={test_bal_acc:.3f}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── Per-class recall bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(C.CLASS_NAMES, test_recalls, color=_CLASS_COLOURS)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class recall on test set")
    ax.tick_params(axis="x", rotation=30)
    for bar, r in zip(bars, test_recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02,
                f"{r:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "per_class_recall.png"), dpi=150)
    plt.close()
    logging.info("Plots saved.")

    # ── Save parameters ───────────────────────────────────────
    torch.save(params_best, os.path.join(exp_dir, "best_params.pt"))

    # ── Circuit diagram ───────────────────────────────────────
    try:
        fig_circ, _ = qml.draw_mpl(
            raw_qnode, decimals=2, style="pennylane",
        )(x_train[0], params_best)
        fig_circ.suptitle(
            f"Best evolved QCNN — DermaMNIST 7-class  "
            f"[{layer_signature(best_motif)}]",
            fontsize=11, y=1.01,
        )
        fig_circ.tight_layout()
        fig_circ.savefig(os.path.join(exp_dir, "best_circuit.png"),
                         dpi=150, bbox_inches="tight")
        plt.close(fig_circ)
        logging.info("Circuit diagram saved.")
    except Exception as e:
        logging.warning(f"Circuit drawing failed: {e}")

    return params_best, {
        "accuracy":          test_acc,
        "bal_acc":           test_bal_acc,
        "per_class_recalls": test_recalls,
        "roc_auc_ovr":       auc,
    }

# ================================================================
# PERSIST BEST CIRCUIT ARTEFACTS
# ================================================================
def save_best_circuit(best_motif, best_info, best_params,
                      exp_dir: str = None) -> None:
    """Save architecture JSON, search metadata JSON and trained weights."""
    if exp_dir is None:
        exp_dir = C.FINAL_EXP_DIR
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "best_circuit_arch.json"), "w") as f:
        json.dump(motif_to_dict(best_motif), f, indent=2)

    n_conv, n_pool, n_total = count_layers(best_motif)
    info_dict = {
        "motif_id":         best_info.motif_id,
        "val_acc":          best_info.val_acc,
        "fitness":          best_info.fitness,
        "bal_acc":          best_info.bal_acc,
        "min_recall":       best_info.min_recall,
        "n_params":         best_info.n_params,
        "n_gate_arities":   best_info.n_gate_arities,
        "generation":       best_info.generation,
        "evo_step":         best_info.evo_step,
        "mutation_type":    best_info.mutation_type,
        # variable-length structure record
        "n_layers":         n_total,
        "n_conv_layers":    n_conv,
        "n_pool_layers":    n_pool,
        "layer_signature":  layer_signature(best_motif),
        # config context
        "class_names":      C.CLASS_NAMES,
        "W_CLASS":          C.W_CLASS,
        "N_QUBITS":         C.N_QUBITS,
        "N_READOUT_QUBITS": C.N_READOUT_QUBITS,
        "PCA_COMPONENTS":   C.PCA_COMPONENTS,
        "encoding":         ("dense (RX + RY per qubit)" if C.is_dense_encoding()
                             else "single (RY per qubit)"),
    }
    with open(os.path.join(exp_dir, "best_circuit_info.json"), "w") as f:
        json.dump(info_dict, f, indent=2)

    torch.save(best_params, os.path.join(exp_dir, "best_params.pt"))
    print(f"Artefacts saved → {exp_dir}")
