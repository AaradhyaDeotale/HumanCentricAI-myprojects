"""
Service layer for Project 3.

Bridges the Django views and the ML core in `project3.ml`. Responsibilities:
  - locate the AG News data
  - train / cache the baseline classifier (so requests don't retrain)
  - run Tasks 2/3/4 on demand
  - render matplotlib figures into MEDIA and return their URLs

The trained classifier is held in a module-level cache keyed by config,
because retraining on every HTTP request would be far too slow. For a
multi-process deployment you'd persist to disk (artifact_path) instead;
the single-dev-server setup here keeps it in memory.
"""

import json
import os
import threading

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend for the server
import matplotlib.pyplot as plt

from django.conf import settings

from .ml.data import load_agnews, subset, CLASS_NAMES
from .ml.classifiers import build_classifier
from .ml.expert import default_expert_team, evaluate_expert
from .ml.defer import (estimate_expert_competence, DeferralSystem,
                       evaluate_system)
from .ml.active import run_active_learning, competence_error

_LOCK = threading.Lock()
_CACHE = {}     # config_key -> dict(clf, train_ds, test_ds, experts, ...)

N_CLASSES = len(CLASS_NAMES)


def _data_dir():
    """
    Where to find train.csv / test.csv. Override with env var
    AGNEWS_DIR. Defaults to <BASE_DIR>/data/agnews.
    """
    return os.environ.get(
        "AGNEWS_DIR", os.path.join(settings.BASE_DIR, "data", "agnews"))


def _media_url(filename):
    return settings.MEDIA_URL + filename


def _media_path(filename):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    return os.path.join(settings.MEDIA_ROOT, filename)


def get_state(clf_kind="tfidf", train_cap=20000, seed=0, force=False):
    """
    Train (or fetch from cache) the baseline classifier and expert team.
    train_cap subsamples the training set so the demo trains in seconds;
    set to None to use all 120k examples.
    """
    key = (clf_kind, train_cap, seed)
    with _LOCK:
        if key in _CACHE and not force:
            return _CACHE[key]

    train_ds, test_ds = load_agnews(source="csv", data_dir=_data_dir())
    rng = np.random.default_rng(seed)
    if train_cap and train_cap < len(train_ds):
        idx = rng.choice(len(train_ds), size=train_cap, replace=False)
        train_ds = subset(train_ds, list(idx))

    clf = build_classifier(clf_kind)
    clf.fit(train_ds.texts, train_ds.labels)
    preds = clf.predict(test_ds.texts)
    test_acc = float(np.mean(np.array(preds) == np.array(test_ds.labels)))

    experts = default_expert_team(n_classes=N_CLASSES, seed=seed)

    state = {
        "clf": clf, "clf_kind": clf_kind, "test_accuracy": test_acc,
        "n_train": len(train_ds), "train_ds": train_ds, "test_ds": test_ds,
        "experts": experts,
    }
    with _LOCK:
        _CACHE[key] = state
    return state


# ---------- Task 2 ----------
def expert_report(state):
    rows = []
    for e in state["experts"]:
        stats = evaluate_expert(e, state["test_ds"].labels)
        rows.append({
            "name": e.name,
            "accuracy": round(stats["accuracy"], 3),
            "per_class": {CLASS_NAMES[c]: round(v, 2)
                          for c, v in stats["per_class"].items()},
        })
    return rows


# ---------- Task 3 ----------
def run_deferral(state, tau, expert_index=0):
    expert = state["experts"][expert_index]
    test_ds = state["test_ds"]
    val_n = min(2000, len(test_ds))
    comp = estimate_expert_competence(
        expert, test_ds.labels[:val_n], list(range(val_n)), N_CLASSES)
    system = DeferralSystem(state["clf"], expert, comp, tau=tau)
    res = evaluate_system(system, test_ds.texts, test_ds.labels,
                          list(range(len(test_ds))))
    res["tau"] = tau
    res["expert"] = expert.name
    res["competence_est"] = [round(float(x), 3) for x in comp]
    return res


def deferral_curve_plot(state, expert_index=0, filename="p3_defer_curve.png"):
    """System accuracy & coverage vs tau -> saved figure URL."""
    taus = np.linspace(-0.2, 0.4, 13)
    accs, covs = [], []
    for t in taus:
        r = run_deferral(state, float(t), expert_index)
        accs.append(r["system_accuracy"])
        covs.append(r["coverage"])

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(taus, accs, "o-", color="tab:blue", label="system accuracy")
    ax1.axhline(state["test_accuracy"], ls="--", color="gray",
                label="classifier alone")
    ax1.set_xlabel(r"deferral threshold $\tau$")
    ax1.set_ylabel("accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(taus, covs, "s-", color="tab:red", label="coverage")
    ax2.set_ylabel("coverage (kept)", color="tab:red")
    fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.15))
    fig.tight_layout()
    fig.savefig(_media_path(filename), dpi=110)
    plt.close(fig)
    return _media_url(filename)


# ---------- Task 4 ----------
def run_active(state, strategy, budget=400, batch=40, expert_index=0, seed=0):
    expert = state["experts"][expert_index]
    al = run_active_learning(state["clf"], expert, state["train_ds"],
                             N_CLASSES, strategy=strategy, budget=budget,
                             batch=batch, seed=seed)
    err = competence_error(al["competence"], expert, N_CLASSES)
    return {
        "strategy": strategy,
        "n_queries": al["n_queries"],
        "final_l1_error": round(err, 4),
        "history": al["history"],
        "final_competence": [round(float(x), 3)
                             for x in al["competence"].mean()],
        "true_competence": [round(expert.competence_profile()[c], 3)
                            for c in range(N_CLASSES)],
    }


def active_compare_plot(state, expert_index=0, budget=400,
                        filename="p3_active_curve.png"):
    """L1 competence error vs #queries, for all strategies."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"random": "gray", "clf_uncertain": "tab:green",
              "competence_gap": "tab:purple"}
    summary = {}
    for strat in ["random", "clf_uncertain", "competence_gap"]:
        expert = state["experts"][expert_index]
        al = run_active_learning(state["clf"], expert, state["train_ds"],
                                 N_CLASSES, strategy=strat, budget=budget,
                                 batch=40, seed=0)
        xs, ys = [], []
        for h in al["history"]:
            est = np.array(h["competence_mean"])
            true = np.array([expert.competence_profile()[c]
                             for c in range(N_CLASSES)])
            xs.append(h["n_queries"])
            ys.append(float(np.mean(np.abs(est - true))))
        ax.plot(xs, ys, "o-", color=colors[strat], label=strat)
        summary[strat] = round(ys[-1], 4)
    ax.set_xlabel("number of expert queries")
    ax.set_ylabel("competence L1 error")
    ax.set_title("Active learning: competence discovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_media_path(filename), dpi=110)
    plt.close(fig)
    return _media_url(filename), summary


# ---------- single-input demo ----------
def classify_text(state, text, expert_index=0):
    clf = state["clf"]
    proba = clf.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    expert = state["experts"][expert_index]
    comp = estimate_expert_competence(
        expert, state["test_ds"].labels[:2000], list(range(2000)), N_CLASSES)
    exp_expected = float(proba @ comp)
    clf_conf = float(proba.max())
    defer = exp_expected > clf_conf
    return {
        "pred_class": CLASS_NAMES[pred],
        "proba": {CLASS_NAMES[c]: round(float(proba[c]), 3)
                  for c in range(N_CLASSES)},
        "clf_confidence": round(clf_conf, 3),
        "expert_expected": round(exp_expected, 3),
        "decision": "DEFER to expert" if defer else "PREDICT with model",
    }
