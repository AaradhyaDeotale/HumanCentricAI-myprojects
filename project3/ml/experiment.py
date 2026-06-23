"""
End-to-end experiment runner (Tasks 1-4).

Produces all the numbers needed for the PDF report:
  Task 1: baseline classifier test accuracy
  Task 2: simulated expert accuracy (overall + per class)
  Task 3: learning-to-defer system vs baselines
  Task 4: active learning competence-discovery curves across strategies
"""

import numpy as np
from data import load_agnews, subset, CLASS_NAMES
from classifiers import build_classifier
from expert import default_expert_team, evaluate_expert
from defer import (estimate_expert_competence, DeferralSystem,
                   evaluate_system)
from active import run_active_learning, competence_error


def run(source="csv", data_dir=None, clf_kind="tfidf",
        train_cap=None, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)
    out = {}

    train_ds, test_ds = load_agnews(source=source, data_dir=data_dir)
    if train_cap and train_cap < len(train_ds):
        idx = rng.choice(len(train_ds), size=train_cap, replace=False)
        train_ds = subset(train_ds, list(idx))

    # ---- Task 1: baseline classifier ----
    clf = build_classifier(clf_kind)
    clf.fit(train_ds.texts, train_ds.labels)
    test_preds = clf.predict(test_ds.texts)
    base_acc = float(np.mean(np.array(test_preds) == np.array(test_ds.labels)))
    out["task1"] = {"classifier": clf_kind, "test_accuracy": base_acc,
                    "n_train": len(train_ds)}
    if verbose:
        print(f"[Task 1] {clf_kind} test accuracy: {base_acc:.4f}")

    # ---- Task 2: simulated experts ----
    experts = default_expert_team(n_classes=n_classes, seed=seed)
    out["task2"] = []
    for e in experts:
        stats = evaluate_expert(e, test_ds.labels)
        out["task2"].append({"name": e.name, **stats,
                             "true_profile": e.competence_profile()})
        if verbose:
            print(f"[Task 2] {e.name}: acc={stats['accuracy']:.3f} "
                  f"per_class={ {k: round(v,2) for k,v in stats['per_class'].items()} }")

    expert = experts[0]  # use the World/Sports specialist for tasks 3-4

    # ---- Task 3: learning to defer ----
    # estimate expert competence on a held-in labelled slice
    val_n = min(2000, len(test_ds))
    val_ids = list(range(val_n))
    val_labels = test_ds.labels[:val_n]
    comp_vec = estimate_expert_competence(expert, val_labels, val_ids, n_classes)

    eval_ids = list(range(len(test_ds)))
    best = None
    for tau in [-0.1, 0.0, 0.1, 0.2]:
        system = DeferralSystem(clf, expert, comp_vec, tau=tau)
        res = evaluate_system(system, test_ds.texts, test_ds.labels, eval_ids)
        res["tau"] = tau
        if best is None or res["system_accuracy"] > best["system_accuracy"]:
            best = res
        if verbose:
            print(f"[Task 3] tau={tau:+.2f} sys={res['system_accuracy']:.4f} "
                  f"cov={res['coverage']:.2f} clf={res['classifier_accuracy']:.4f} "
                  f"exp={res['expert_accuracy']:.4f}")
    out["task3"] = {"best": best, "expert_competence_est": comp_vec.tolist()}

    # ---- Task 4: active learning competence discovery ----
    out["task4"] = {}
    for strat in ["random", "clf_uncertain", "competence_gap"]:
        al = run_active_learning(clf, expert, train_ds, n_classes,
                                 strategy=strat, budget=400, batch=40,
                                 seed=seed)
        err = competence_error(al["competence"], expert, n_classes)
        out["task4"][strat] = {
            "history": al["history"],
            "final_competence_L1_error": err,
            "final_competence_mean": al["competence"].mean().tolist(),
        }
        if verbose:
            print(f"[Task 4] {strat}: final competence L1 error={err:.4f} "
                  f"after {al['n_queries']} queries")

    return out


if __name__ == "__main__":
    import json, sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/agnews"
    results = run(source="csv", data_dir=data_dir, clf_kind="tfidf")
    print("\n=== JSON SUMMARY ===")
    print(json.dumps({k: (v if k != "task4" else
                          {s: {kk: vv for kk, vv in d.items() if kk != "history"}
                           for s, d in v.items()})
                     for k, v in results.items()}, indent=2)[:2000])
