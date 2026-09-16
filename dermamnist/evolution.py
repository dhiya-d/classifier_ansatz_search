# ----------------------------------------------------------------
# evolution.py  —  Fitness evaluation, genetic operators, search loop
#                  (DermaMNIST 7-class, VARIABLE-LENGTH genotypes)
#
# A genotype is any ordered sequence of Qcycle (conv) and Qmask (pool)
# layers, of length between C.MIN_LAYERS and C.MAX_LAYERS.  Layer types
# may appear in any order and any proportion — including zero pooling.
#
# The only hard structural requirement is that at least
# C.N_READOUT_QUBITS (= 3) qubits survive all pooling.
#
# Genetic operators (all length-changing):
#   crossover(A, B)  → A[:ka] + B[kb:]   single-point, independent cuts
#   mutate(A)        → one of: replace_tail | insert | delete | point
# ----------------------------------------------------------------

import logging
import random
import shelve
from collections import namedtuple
from copy import deepcopy

import dill
import numpy as np
import ray
import torch
import torch.nn.functional as F

from hierarqcal import Qcycle, Qmask, Qunmask, Qmotifs

import config as C
from circuit import build_qcnn_qnode, get_gate_applications
from gates import MAPPING_DICT, conv_gates, pool_gates
from serialisation import motif_from_dict, motif_to_dict

# ── Module-level caches ────────────────────────────────────────────
MOTIF_CACHE     = {}
TASK_CACHE      = []
DISK_CACHE_PATH = "motif_cache.db"

# Raw pool gate functions, used to distinguish conv from pool gate
# applications when checking for degenerate circuits.
_POOL_FNS = {g.function for g in pool_gates}

# ── Named-tuples ───────────────────────────────────────────────────
Task_Information = namedtuple("Task_Information", [
    "motif_id", "parent_id", "mutation_type", "count",
    "val_acc", "fitness", "n_gate_arities", "n_params",
    "generation", "evo_step",
    "bal_acc",    # mean per-class recall (balanced accuracy)
    "min_recall", # recall of the worst-predicted class
])

Task = namedtuple("Task", [
    "motif_object", "motif_id", "parent_id",
    "mutation_type", "generation",
])

MUTATION_TYPES = {
    "cj": "crossover",
    "mu": "mutate",
    "ra": "random",
}

_ALL_PATTERNS = [
    "*1", "1*", "*1*", "1*1",
    "!0", "0!", "!0!", "0!0",
    "*!", "!*", "01",  "10",
    "101", "010",
]

# ================================================================
# CACHE RESET
# ================================================================
def reset_caches() -> None:
    """Clear in-memory caches between sweep runs."""
    global MOTIF_CACHE, TASK_CACHE
    MOTIF_CACHE = {}
    TASK_CACHE  = []

# ================================================================
# METRIC HELPERS
# ================================================================
def _per_class_recalls(preds_np, labels_np, n_classes=None):
    """Return per-class recall for each class 0…n_classes-1."""
    if n_classes is None:
        n_classes = C.N_CLASSES
    recalls = []
    for c in range(n_classes):
        tp = int(np.sum((preds_np == c) & (labels_np == c)))
        fn = int(np.sum((preds_np != c) & (labels_np == c)))
        recalls.append(tp / (tp + fn + 1e-8))
    return recalls


def _weighted_fitness(recalls, n_gates_arity, n_params, w_class, n_qubits, l1, l2,
                       n_layers, l3):
    """Severity-weighted complement of per-class recall + complexity penalty."""
    return (
        sum(w_class[c] * (1.0 - recalls[c]) for c in range(len(w_class)))
        + (n_gates_arity / n_qubits) * l1
        + n_params * l2
        + n_layers * l3
    )

# ================================================================
# HYPERPARAMETER SAMPLERS
# ================================================================
_get_stride    = lambda: int(np.random.choice(range(1, C.NQ_MAX)))
_get_step      = lambda: int(np.random.choice(range(1, C.NQ_MAX)))
_get_offset    = lambda: int(np.random.choice(range(4), p=[0.5, 0.4, 0.05, 0.05]))
_get_boundary  = lambda: str(np.random.choice(["open", "periodic"]))
_get_strides   = lambda: [_get_stride(),   _get_stride(),   _get_stride()]
_get_steps     = lambda: [_get_step(),     _get_step(),     _get_step()]
_get_offsets   = lambda: [_get_offset(),   _get_offset(),   _get_offset()]
_get_boundaries= lambda: [_get_boundary(), _get_boundary(), _get_boundary()]

# ================================================================
# VALIDITY CHECK
# ================================================================
def is_valid_motif(motif, n_qubits: int = None) -> bool:
    """
    A valid variable-length motif must:

    - Have between MIN_LAYERS and MAX_LAYERS layers in total
    - Contain at least MIN_CONV_LAYERS Qcycle layers (it is a *conv* net)
    - Contain at least MIN_POOL_LAYERS Qmask layers (default 0 — optional)
    - Produce at least one gate application
    - Leave at least N_READOUT_QUBITS qubits active after all pooling
    - Not be degenerate: conv layers must actually connect qubit pairs
    """
    if n_qubits is None:
        n_qubits = C.N_QUBITS
    try:
        n_layers = len(motif)
        if not (C.MIN_LAYERS <= n_layers <= C.MAX_LAYERS):
            return False

        n_conv = sum(1 for s in motif if isinstance(s, Qcycle))
        n_pool = sum(1 for s in motif if isinstance(s, Qmask))
        if n_conv < C.MIN_CONV_LAYERS:
            return False
        if n_pool < C.MIN_POOL_LAYERS:
            return False

        apps, remaining = get_gate_applications(motif, n_qubits)
        if len(apps) == 0:
            return False

        # Enough qubits must survive for the 3-qubit readout
        if len(remaining) < C.N_READOUT_QUBITS:
            return False

        # Degeneracy guard — conv layers must do real work
        conv_apps = sum(1 for fn, _, _ in apps if fn not in _POOL_FNS)
        if conv_apps < C.MIN_CONV_APPS_PER_LAYER * n_conv:
            return False

        return True
    except Exception:
        return False


def count_layers(motif):
    """Return (n_conv, n_pool, n_total) for logging and reporting."""
    n_conv = sum(1 for s in motif if isinstance(s, Qcycle))
    n_pool = sum(1 for s in motif if isinstance(s, Qmask))
    return n_conv, n_pool, len(motif)


def layer_signature(motif) -> str:
    """Compact readable structure string, e.g. 'C-C-P-C-P'."""
    return "-".join(
        "C" if isinstance(s, Qcycle) else "P" if isinstance(s, Qmask) else "U"
        for s in motif
    )

# ================================================================
# RANDOM LAYER CONSTRUCTORS
# ================================================================
def _rand_conv(exclude_gate_names=None) -> Qcycle:
    """Sample a random Qcycle layer, optionally avoiding used gate names."""
    exclude_gate_names = exclude_gate_names or []
    candidates = [g for g in conv_gates if g.name not in exclude_gate_names]
    if not candidates:
        candidates = conv_gates
    gate = random.choice(candidates)
    return Qcycle(
        mapping      = gate,
        stride       = _get_stride(),
        step         = _get_step(),
        offset       = _get_offset(),
        boundary     = _get_boundary(),
        share_weights= False,
    )


def _rand_pool(exclude_patterns=None) -> Qmask:
    """Sample a random Qmask layer, optionally avoiding used patterns."""
    exclude_patterns = exclude_patterns or []
    candidates = [p for p in _ALL_PATTERNS if p not in exclude_patterns]
    if not candidates:
        candidates = _ALL_PATTERNS
    pattern = random.choice(candidates)
    gate    = random.choice(pool_gates)
    return Qmask(
        global_pattern = pattern,
        mapping        = gate,
        strides        = _get_strides(),
        steps          = _get_steps(),
        offsets        = _get_offsets(),
        boundaries     = _get_boundaries(),
    )


def _rand_layer(used_conv_names=None, used_patterns=None):
    """Sample a single random layer, conv with probability C.P_CONV."""
    if random.random() < C.P_CONV:
        return _rand_conv(exclude_gate_names=used_conv_names)
    return _rand_pool(exclude_patterns=used_patterns)


def _used_names(layers):
    """Collect gate names and pool patterns already present in a layer list."""
    conv_names = [
        s.mapping.name for s in layers
        if isinstance(s, Qcycle) and s.mapping is not None
    ]
    patterns = [s.global_pattern for s in layers if isinstance(s, Qmask)]
    return conv_names, patterns

# ================================================================
# GENOTYPE CREATION
# ================================================================
def create_genotype() -> Qmotifs:
    """
    Create a random valid variable-length genotype.

    Length is sampled uniformly from [MIN_LAYERS, MAX_LAYERS], then each
    slot is independently filled with a conv layer (prob C.P_CONV) or a
    pool layer.  Gate types and pool patterns are sampled without
    replacement while distinct options remain, so short circuits stay
    diverse and long ones simply start reusing options.
    """
    for _ in range(200):
        n_layers = random.randint(C.MIN_LAYERS, C.MAX_LAYERS)

        layers, used_conv, used_pat = [], [], []
        for _ in range(n_layers):
            layer = _rand_layer(used_conv, used_pat)
            if isinstance(layer, Qcycle):
                used_conv.append(layer.mapping.name)
            else:
                used_pat.append(layer.global_pattern)
            layers.append(layer)

        # Guarantee at least MIN_CONV_LAYERS convolutional layers
        n_conv = sum(1 for s in layers if isinstance(s, Qcycle))
        while n_conv < C.MIN_CONV_LAYERS:
            pos = random.randrange(len(layers))
            if not isinstance(layers[pos], Qcycle):
                layers[pos] = _rand_conv(exclude_gate_names=used_conv)
                n_conv += 1

        g = Qmotifs(tuple(layers))
        if is_valid_motif(g):
            return g

    return _fallback_genotype()


def _fallback_genotype() -> Qmotifs:
    """
    Deterministic genotype guaranteed valid: conv-heavy with two gentle
    pools, leaving well above N_READOUT_QUBITS qubits alive.
    Only used if 200 random attempts all fail.
    """
    layers = [
        Qcycle(mapping=conv_gates[0], stride=1, step=1, offset=0,
               boundary="open", share_weights=False),
        Qmask(global_pattern="*1*", mapping=pool_gates[0],
              strides=[1,1,1], steps=[1,1,1], offsets=[0,0,0],
              boundaries=["open","open","open"]),
        Qcycle(mapping=conv_gates[1], stride=1, step=1, offset=0,
               boundary="open", share_weights=False),
        Qmask(global_pattern="1*1", mapping=pool_gates[1],
              strides=[1,1,1], steps=[1,1,1], offsets=[0,0,0],
              boundaries=["open","open","open"]),
        Qcycle(mapping=conv_gates[2], stride=1, step=1, offset=0,
               boundary="open", share_weights=False),
    ]
    return Qmotifs(tuple(layers))

# ================================================================
# GENETIC OPERATORS  (variable length)
# ================================================================
def crossover(m_a: Qmotifs, m_b: Qmotifs) -> Qmotifs:
    """
    Single-point crossover with independent cut points.

        child = m_a[:ka] + m_b[kb:]

    Because ka and kb are sampled independently, the child's length
    differs from both parents — this is what lets circuit depth evolve.
    Result is clipped to MAX_LAYERS.
    """
    ka = random.randint(1, len(m_a))
    kb = random.randint(0, max(0, len(m_b) - 1))

    child = tuple(m_a)[:ka] + tuple(m_b)[kb:]

    if len(child) > C.MAX_LAYERS:
        child = child[:C.MAX_LAYERS]

    return Qmotifs(child)


def mutate_replace_tail(m: Qmotifs) -> Qmotifs:
    """Keep a random prefix, regrow a fresh random tail of random length."""
    k    = random.randint(1, max(1, len(m) - 1))
    head = tuple(m)[:k]

    max_new = max(1, C.MAX_LAYERS - k)
    n_new   = random.randint(1, max_new)

    used_conv, used_pat = _used_names(head)
    tail = []
    for _ in range(n_new):
        layer = _rand_layer(used_conv, used_pat)
        if isinstance(layer, Qcycle):
            used_conv.append(layer.mapping.name)
        else:
            used_pat.append(layer.global_pattern)
        tail.append(layer)

    return Qmotifs(head + tuple(tail))


def mutate_insert(m: Qmotifs) -> Qmotifs:
    """Insert one freshly sampled layer at a random position (grows by 1)."""
    used_conv, used_pat = _used_names(list(m))
    layer = _rand_layer(used_conv, used_pat)
    pos   = random.randint(0, len(m))
    lst   = list(m)
    lst.insert(pos, layer)
    return Qmotifs(tuple(lst))


def mutate_delete(m: Qmotifs) -> Qmotifs:
    """Remove one layer at a random position (shrinks by 1)."""
    lst = list(m)
    pos = random.randrange(len(lst))
    lst.pop(pos)
    return Qmotifs(tuple(lst))


def mutate_point(m: Qmotifs) -> Qmotifs:
    """
    Replace exactly one layer with a fresh layer of the SAME kind.
    Length unchanged — explores stride / step / offset / gate type /
    pattern without altering the C/P skeleton.
    """
    lst = list(m)
    pos = random.randrange(len(lst))
    used_conv, used_pat = _used_names(lst[:pos] + lst[pos + 1:])

    if isinstance(lst[pos], Qcycle):
        lst[pos] = _rand_conv(exclude_gate_names=used_conv)
    else:
        lst[pos] = _rand_pool(exclude_patterns=used_pat)

    return Qmotifs(tuple(lst))


def mutate(m: Qmotifs) -> Qmotifs:
    """
    Dispatch to one structural mutation, sampled from C.P_MUTATE.
    Operators that would violate the length bounds are skipped.
    """
    ops   = list(C.P_MUTATE.keys())
    probs = [C.P_MUTATE[o] for o in ops]

    legal, legal_p = [], []
    for o, p in zip(ops, probs):
        if o == "insert" and len(m) >= C.MAX_LAYERS:
            continue
        if o == "delete" and len(m) <= C.MIN_LAYERS:
            continue
        legal.append(o)
        legal_p.append(p)

    if not legal:
        legal, legal_p = ["point"], [1.0]

    total   = sum(legal_p)
    legal_p = [p / total for p in legal_p]
    op      = np.random.choice(legal, p=legal_p)

    if op == "replace_tail":
        return mutate_replace_tail(m)
    if op == "insert":
        return mutate_insert(m)
    if op == "delete":
        return mutate_delete(m)
    return mutate_point(m)

# ================================================================
# POPULATION UTILITIES
# ================================================================
def get_id_and_is_new(motif):
    mid = str(hash(str(motif_to_dict(motif))))
    return mid, (mid not in MOTIF_CACHE)


def get_initial_population(size: int) -> list:
    tasks    = []
    attempts = 0
    while len(tasks) < size and attempts < size * C.MAX_CHILD_ATTEMPTS:
        attempts += 1
        motif = create_genotype()
        mid   = str(hash(str(motif_to_dict(motif))))
        if mid not in TASK_CACHE and mid not in MOTIF_CACHE:
            tasks.append(Task(motif, mid, (None,), MUTATION_TYPES["ra"], 0))
            TASK_CACHE.append(mid)
    return tasks


def store_motif(mid, motif_dict) -> None:
    MOTIF_CACHE[str(mid)] = motif_dict
    with shelve.open(DISK_CACHE_PATH, flag="c") as db:
        db[str(mid)] = motif_dict

# ================================================================
# TOP-3 SELECTION
# ================================================================
def tournament_select_top3(memory_table: dict, p_explore: float = None):
    """
    Return the 3 best-fitness entries.  With probability p_explore the
    third slot is swapped for a random lower-ranked entry, which keeps
    the crossover pool changing between steps.
    """
    if p_explore is None:
        p_explore = C.P_EXPLORE

    sorted_entries = sorted(memory_table.values(), key=lambda x: x.fitness)
    top3 = list(sorted_entries[:3])

    if len(top3) < 3:
        while len(top3) < 3:
            top3.append(top3[-1])
        return top3

    if np.random.rand() < p_explore and len(sorted_entries) > 3:
        random_entry = random.choice(sorted_entries[3:])
        top3 = [top3[0], top3[1], random_entry]

    return top3

# ================================================================
# FITNESS EVALUATION  (Ray remote)
# ================================================================
@ray.remote
def evaluate_genotype(motif, x_train, y_train, x_val, y_val,
                      lr_eval, n_epochs_eval, batch_size_eval,
                      w_class, n_qubits, l1, l2, l3, bad_fitness,
                      min_recall_target, min_recall_penalty, dense):
    """
    Build → short-train → evaluate one genotype.

    Circuit output: [p(class 0), ..., p(class 6)] from 3-qubit probs,
                    folded from 8 basis states down to 7 classes (the
                    leftover state is summed into
                    C.EXTRA_STATE_TARGET_CLASS — see circuit.py).
    Prediction:     argmax over the 7 class probabilities.

    Returns (fitness, val_acc, n_gates_arity, n_params, bal_acc, min_recall).
    Lower fitness = better.

    All hyperparameters are passed EXPLICITLY rather than read from the
    config module, because Ray workers are separate processes that would
    otherwise import stale config values during a sweep — this includes
    `dense`, which picks single vs dense encoding (PCA8 vs PCA16).
    """
    try:
        circuit, n_params, _ = build_qcnn_qnode(motif, n_qubits, dense=dense)
        gate_apps, _  = get_gate_applications(motif, n_qubits)
        n_gates_arity = sum(len(bits) for _, bits, _ in gate_apps)
        n_layers      = len(motif)

        # ── Zero-parameter circuit ──────────────────────────────
        if n_params == 0:
            with torch.no_grad():
                probs = circuit(x_val, torch.tensor([])).float()
            preds_np  = torch.argmax(probs, dim=1).numpy()
            labels_np = y_val.numpy()
            recalls   = _per_class_recalls(preds_np, labels_np)
            bal_acc   = float(np.mean(recalls))
            val_acc   = float(np.mean(preds_np == labels_np))
            fitness   = _weighted_fitness(recalls, n_gates_arity, n_params,
                                          w_class, n_qubits, l1, l2, n_layers, l3)
            return (fitness, val_acc, n_gates_arity, n_params,
                    bal_acc, min(recalls))

        # ── Short training run ──────────────────────────────────
        params = (torch.rand(n_params) * 2 * np.pi).requires_grad_(True)
        opt    = torch.optim.Adam([params], lr=lr_eval)
        n      = len(x_train)

        for _ in range(n_epochs_eval):
            perm   = torch.randperm(n)
            x_shuf = x_train[perm]
            y_shuf = y_train[perm]
            for start in range(0, n, batch_size_eval):
                x_b = x_shuf[start: start + batch_size_eval]
                y_b = y_shuf[start: start + batch_size_eval]
                opt.zero_grad()
                # probs shape (B, 4) — NLLLoss on log-probabilities
                probs_b = circuit(x_b, params).float()
                loss = F.nll_loss(torch.log(probs_b + 1e-8), y_b)
                loss.backward()
                opt.step()

        # ── Validation metrics ──────────────────────────────────
        with torch.no_grad():
            probs_val = circuit(x_val, params).float()
        preds     = torch.argmax(probs_val, dim=1)
        preds_np  = preds.numpy()
        labels_np = y_val.numpy()

        # Reject fully trivial classifiers (everything one class)
        if len(torch.unique(preds)) < 2:
            return bad_fitness, 0.0, n_gates_arity, n_params, 0.0, 0.0

        recalls = _per_class_recalls(preds_np, labels_np)
        bal_acc = float(np.mean(recalls))
        val_acc = float(np.mean(preds_np == labels_np))

        # Soft penalty when the worst class falls below target recall.
        # Soft rather than a hard reject so the search keeps gradient
        # information from partially-working circuits.
        penalty = max(0.0, min_recall_target - min(recalls)) * min_recall_penalty

        fitness = _weighted_fitness(recalls, n_gates_arity, n_params,
                                    w_class, n_qubits, l1, l2, n_layers, l3) + penalty

        return (fitness, val_acc, n_gates_arity, n_params,
                bal_acc, min(recalls))

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.warning(f"Genotype evaluation failed: {type(e).__name__}: {e}")
        return bad_fitness, 0.0, 0, 0, 0.0, 0.0

# ================================================================
# OFFSPRING GENERATION
# ================================================================
def _try_make_child(build_fn, parents, mut_type, gen, new_tasks):
    """
    Attempt to build one valid, novel child.

    build_fn is retried up to C.MAX_CHILD_ATTEMPTS times because
    variable-length operators fail validity more often than fixed-length
    ones (a delete can drop below MIN_LAYERS, an insert can add a pool
    that kills the readout qubits).

    Returns True if a child was appended to new_tasks.
    """
    for _ in range(C.MAX_CHILD_ATTEMPTS):
        try:
            child = build_fn()
        except Exception:
            continue
        if not is_valid_motif(child):
            continue
        mid, is_new = get_id_and_is_new(child)
        if is_new and mid not in TASK_CACHE:
            new_tasks.append(Task(child, mid, parents, mut_type, gen))
            TASK_CACHE.append(mid)
            return True
    return False


def generate_offspring(memory_table: dict, evo_step: int) -> list:
    """
    Generate up to 9 children from the top-3:
      6 crossover children (all ordered pairs i != j)
      3 mutation children  (one per parent)

    Because crossover and mutation both change length, children can be
    shorter or longer than either parent.
    """
    top3   = tournament_select_top3(memory_table)
    mids   = [e.motif_id for e in top3]
    motifs = [motif_from_dict(MOTIF_CACHE[mid], MAPPING_DICT) for mid in mids]

    new_tasks = []

    # ── 6 crossover children ──────────────────────────────────
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            gen = max(top3[i].generation, top3[j].generation) + 1
            _try_make_child(
                build_fn = lambda a=motifs[i], b=motifs[j]: crossover(a, b),
                parents  = (mids[i], mids[j]),
                mut_type = MUTATION_TYPES["cj"],
                gen      = gen,
                new_tasks= new_tasks,
            )

    # ── 3 mutation children ───────────────────────────────────
    for i in range(3):
        gen = top3[i].generation + 1
        _try_make_child(
            build_fn = lambda a=motifs[i]: mutate(a),
            parents  = (mids[i],),
            mut_type = MUTATION_TYPES["mu"],
            gen      = gen,
            new_tasks= new_tasks,
        )

    # ── Top up with fresh random genotypes if operators stalled ──
    # Without this, a converged top-3 can produce zero valid novel
    # children and the step would silently do nothing.
    while len(new_tasks) < 3:
        if not _try_make_child(
            build_fn = create_genotype,
            parents  = (None,),
            mut_type = MUTATION_TYPES["ra"],
            gen      = evo_step,
            new_tasks= new_tasks,
        ):
            break

    return new_tasks

# ================================================================
# CIRCUIT STRUCTURE SUMMARY
# ================================================================
def log_circuit_structure(motif) -> None:
    n_conv, n_pool, n_total = count_layers(motif)
    logging.info(
        f"  Structure: {layer_signature(motif)}  "
        f"({n_total} layers = {n_conv} conv + {n_pool} pool)"
    )
    for i, sub in enumerate(motif):
        if isinstance(sub, Qcycle):
            logging.info(
                f"  Layer {i} CONV  gate={sub.mapping.name:10s} "
                f"stride={sub.stride} step={sub.step} "
                f"offset={sub.offset} boundary={sub.boundary}"
            )
        elif isinstance(sub, Qmask):
            logging.info(
                f"  Layer {i} POOL  gate={sub.mapping.name:12s} "
                f"pattern={sub.global_pattern}"
            )

    try:
        _, remaining = get_gate_applications(motif, C.N_QUBITS)
        logging.info(
            f"  Qubits surviving: {len(remaining)} → {remaining}  "
            f"| readout wires = {remaining[:C.N_READOUT_QUBITS]}"
        )
    except Exception:
        pass

# ================================================================
# MAIN EVOLUTIONARY LOOP
# ================================================================
def run_evolution(x_train, y_train, x_val, y_val,
                  exp_dir: str = "experiments/run",
                  resume: bool = False):
    """
    Run the full evolutionary architecture search.

    If resume=True and exp_dir/memory_table.pkl exists (written every
    C.SAVE_INTERVAL steps), the search picks up from that checkpoint
    instead of evaluating a fresh initial population.

    Returns
    -------
    best_motif   : Qmotifs
    best         : Task_Information
    memory_table : dict[motif_id → Task_Information]
    """
    import os
    global MOTIF_CACHE, TASK_CACHE, DISK_CACHE_PATH

    DISK_CACHE_PATH = os.path.join(exp_dir, "motif_cache.db")
    os.makedirs(exp_dir, exist_ok=True)

    memory_table_path = os.path.join(exp_dir, "memory_table.pkl")
    checkpoint_found = resume and os.path.exists(memory_table_path)

    if checkpoint_found:
        with open(memory_table_path, "rb") as f:
            memory_table = dill.load(f)
        evo_step = max(t.evo_step for t in memory_table.values())

        with shelve.open(DISK_CACHE_PATH, flag="c") as db:
            MOTIF_CACHE.update(dict(db))

        logging.info(
            f"Resumed from checkpoint → {len(memory_table)} genotypes, "
            f"evo_step={evo_step}/{C.MAX_EVO_STEPS}, "
            f"{len(MOTIF_CACHE)} motifs loaded from disk cache"
        )
    else:
        if resume:
            logging.info(
                f"--resume given but no checkpoint found at {memory_table_path} "
                "→ starting fresh"
            )

        memory_table = {}
        evo_step     = 0

        logging.info("Evaluating initial population...")
        logging.info(
            f"Config: LR={C.LR_EVAL}  epochs={C.N_EPOCHS_EVAL}  "
            f"pop={C.INIT_POP}  steps={C.MAX_EVO_STEPS}  "
            f"W_CLASS={C.W_CLASS}"
        )
        _enc_desc = "dense (RX + RY per qubit)" if C.is_dense_encoding() else "single (RY per qubit)"
        logging.info(
            f"Encoding: {_enc_desc}, {C.PCA_COMPONENTS} PCA components → "
            f"{C.N_QUBITS} qubits"
        )
        logging.info(
            f"Structure: variable length {C.MIN_LAYERS}–{C.MAX_LAYERS} layers  "
            f"| P_CONV={C.P_CONV}  | min conv={C.MIN_CONV_LAYERS}  "
            f"| min pool={C.MIN_POOL_LAYERS}  "
            f"| readout qubits={C.N_READOUT_QUBITS}"
        )

        initial_tasks = get_initial_population(C.INIT_POP)
        logging.info(f"Initial population: {len(initial_tasks)} valid genotypes")

        futures = [
            evaluate_genotype.remote(
                task.motif_object, x_train, y_train, x_val, y_val,
                C.LR_EVAL, C.N_EPOCHS_EVAL, C.BATCH_SIZE_EVAL,
                C.W_CLASS, C.N_QUBITS, C.L1, C.L2, C.L3, C.BAD_FITNESS,
                C.MIN_RECALL_TARGET, C.MIN_RECALL_PENALTY, C.is_dense_encoding(),
            )
            for task in initial_tasks
        ]
        results = ray.get(futures)

        for task, (fitness, val_acc, n_gates, n_params, bal_acc, min_recall) \
                in zip(initial_tasks, results):
            store_motif(task.motif_id, motif_to_dict(task.motif_object))
            memory_table[task.motif_id] = Task_Information(
                motif_id      = task.motif_id,
                parent_id     = task.parent_id,
                mutation_type = task.mutation_type,
                count         = 1,
                val_acc       = val_acc,
                fitness       = fitness,
                n_gate_arities= n_gates,
                n_params      = n_params,
                generation    = task.generation,
                evo_step      = evo_step,
                bal_acc       = bal_acc,
                min_recall    = min_recall,
            )
            logging.info(
                f"  {task.mutation_type:10s} | {layer_signature(task.motif_object):<22s} | "
                f"fitness={fitness:.4f} | val_acc={val_acc:.4f} | "
                f"bal_acc={bal_acc:.4f} | min_recall={min_recall:.4f} | "
                f"params={n_params}"
            )

    # ── Evolutionary loop ──────────────────────────────────────
    while evo_step < C.MAX_EVO_STEPS:
        evo_step += 1
        logging.info(f"\n=== Evo step {evo_step}/{C.MAX_EVO_STEPS} ===")

        top3 = tournament_select_top3(memory_table)
        logging.info(
            "Top-3: " + "  |  ".join(
                f"{t.motif_id[:8]}… fit={t.fitness:.4f}" for t in top3
            )
        )

        new_tasks = generate_offspring(memory_table, evo_step)
        logging.info(f"Generated {len(new_tasks)} valid novel children")

        futures = [
            evaluate_genotype.remote(
                task.motif_object, x_train, y_train, x_val, y_val,
                C.LR_EVAL, C.N_EPOCHS_EVAL, C.BATCH_SIZE_EVAL,
                C.W_CLASS, C.N_QUBITS, C.L1, C.L2, C.L3, C.BAD_FITNESS,
                C.MIN_RECALL_TARGET, C.MIN_RECALL_PENALTY, C.is_dense_encoding(),
            )
            for task in new_tasks
        ]
        results = ray.get(futures)

        for task, (fitness, val_acc, n_gates, n_params, bal_acc, min_recall) \
                in zip(new_tasks, results):
            store_motif(task.motif_id, motif_to_dict(task.motif_object))
            memory_table[task.motif_id] = Task_Information(
                motif_id      = task.motif_id,
                parent_id     = task.parent_id,
                mutation_type = task.mutation_type,
                count         = 1,
                val_acc       = val_acc,
                fitness       = fitness,
                n_gate_arities= n_gates,
                n_params      = n_params,
                generation    = task.generation,
                evo_step      = evo_step,
                bal_acc       = bal_acc,
                min_recall    = min_recall,
            )
            logging.info(
                f"  {task.mutation_type:10s} | {layer_signature(task.motif_object):<22s} | "
                f"fitness={fitness:.4f} | val_acc={val_acc:.4f} | "
                f"bal_acc={bal_acc:.4f} | min_recall={min_recall:.4f} | "
                f"params={n_params}"
            )

        best = min(memory_table.values(), key=lambda x: x.fitness)
        logging.info(
            f"Best so far → val_acc={best.val_acc:.4f}  "
            f"bal_acc={best.bal_acc:.4f}  min_recall={best.min_recall:.4f}  "
            f"fitness={best.fitness:.4f}  params={best.n_params}  "
            f"id={best.motif_id[:8]}…"
        )

        if evo_step % C.SAVE_INTERVAL == 0:
            with open(os.path.join(exp_dir, "memory_table.pkl"), "wb") as f:
                dill.dump(memory_table, f)
            logging.info("Memory table saved.")

    # ── Return best ────────────────────────────────────────────
    best       = min(memory_table.values(), key=lambda x: x.fitness)
    best_motif = motif_from_dict(MOTIF_CACHE[best.motif_id], MAPPING_DICT)
    n_conv, n_pool, n_total = count_layers(best_motif)

    logging.info("\n=== SEARCH COMPLETE ===")
    logging.info(f"Best val_acc  : {best.val_acc:.4f}")
    logging.info(f"Best bal_acc  : {best.bal_acc:.4f}  (mean per-class recall)")
    logging.info(f"Min recall    : {best.min_recall:.4f}  (worst class)")
    logging.info(f"Best fitness  : {best.fitness:.4f}")
    logging.info(f"Params        : {best.n_params}")
    logging.info(f"Layers        : {n_total}  ({n_conv} conv + {n_pool} pool)")
    logging.info("Circuit structure:")
    log_circuit_structure(best_motif)

    return best_motif, best, memory_table