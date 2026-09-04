"""
Active Learning for expert competence discovery (Task 4).

Setting: we have the full training set to learn the classifier, but NO
expert labels. We may query the expert on chosen examples to obtain a
label. The goal is to learn -- with as few queries as possible -- WHERE
the expert is competent, so we know when deferring is beneficial.

Key insight: deferral is useful on an input x exactly when the expert is
more likely correct than the classifier. So the value of querying x is
highest where we are most UNCERTAIN about that comparison. We estimate
the expert's per-class competence and actively pick queries to sharpen it.

We compare several strategies:

  random         -- pick query points uniformly (baseline).

  clf_uncertain  -- pick points where the classifier is least confident
                    (classic uncertainty sampling). These are the points
                    where the expert *might* help, so its competence there
                    matters most.

  competence_gap -- pick points whose predicted class currently has the
                    most uncertain expert-competence estimate (widest
                    Beta posterior), i.e. learn the competence profile
                    where we know least. This directly targets the
                    competence-discovery objective.

Competence is modelled per class with a Beta(alpha, beta) posterior over
"expert correct on this class". Each query updates the posterior of the
class the expert is asked about (we use the TRUE label of the queried
point, which is what the expert's correctness is defined against; in a
fully label-free setting one would use the classifier's predicted class
-- both variants are supported via `use_true_label`).
"""

import numpy as np


class CompetenceEstimator:
    """Per-class Beta posterior over P(expert correct | class)."""

    def __init__(self, n_classes, prior_a=1.0, prior_b=1.0):
        self.a = np.full(n_classes, prior_a)
        self.b = np.full(n_classes, prior_b)

    def update(self, class_id, correct):
        if correct:
            self.a[class_id] += 1
        else:
            self.b[class_id] += 1

    def mean(self):
        return self.a / (self.a + self.b)

    def variance(self):
        a, b = self.a, self.b
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


def pick_queries(strategy, pool_idx, proba_pool, comp, rng, batch=1):
    """Public entry point for `_acquire`, used by both the simulated-expert
    loop below and the human-in-the-loop session (Task 5)."""
    return _acquire(strategy, pool_idx, proba_pool, comp, rng, batch)


def _acquire(strategy, pool_idx, proba_pool, comp, rng, batch):
    if strategy == "random":
        return list(rng.choice(pool_idx, size=min(batch, len(pool_idx)),
                               replace=False))

    if strategy == "clf_uncertain":
        conf = proba_pool.max(axis=1)              # high = confident
        order = np.argsort(conf)                    # least confident first
        return [pool_idx[i] for i in order[:batch]]

    if strategy == "competence_gap":
        pred_class = np.argmax(proba_pool, axis=1)
        class_var = comp.variance()
        # primary: classes whose competence we are least sure about;
        # tie-break: points the classifier is least confident on, since
        # those are exactly where knowing the expert's skill matters.
        clf_unc = 1.0 - proba_pool.max(axis=1)
        score = class_var[pred_class] + 1e-3 * clf_unc
        order = np.argsort(-score)
        return [pool_idx[i] for i in order[:batch]]

    raise ValueError(f"Unknown strategy: {strategy}")


def run_active_learning(classifier, expert, train_ds, n_classes,
                        strategy="competence_gap", budget=400, batch=40,
                        seed=0, use_true_label=True):
    """
    Returns a history of competence estimates vs number of queries, plus
    the final CompetenceEstimator.

    classifier must already be fitted (it is fixed; we only learn the
    expert competence here, per the Task 4 setup).
    """
    rng = np.random.default_rng(seed)
    comp = CompetenceEstimator(n_classes)

    all_idx = np.arange(len(train_ds))
    queried = set()
    proba_all = classifier.predict_proba(train_ds.texts)

    history = []
    n_rounds = budget // batch
    for _ in range(n_rounds):
        pool_idx = np.array([i for i in all_idx if i not in queried])
        if len(pool_idx) == 0:
            break
        proba_pool = proba_all[pool_idx]
        picks = _acquire(strategy, pool_idx, proba_pool, comp, rng, batch)

        for i in picks:
            queried.add(int(i))
            y_true = train_ds.labels[i]
            exp_label = expert.query(y_true, int(i))
            correct = int(exp_label == y_true)
            # which class's competence do we credit this observation to?
            if use_true_label:
                cls = y_true
            else:
                cls = int(np.argmax(proba_all[i]))
            comp.update(cls, correct)

        history.append({
            "n_queries": len(queried),
            "competence_mean": comp.mean().tolist(),
            "competence_var": comp.variance().tolist(),
        })

    return {"history": history, "competence": comp,
            "n_queries": len(queried), "strategy": strategy}


def competence_error(comp, expert, n_classes):
    """L1 error between estimated and true per-class competence."""
    true = expert.competence_profile()
    est = comp.mean()
    return float(np.mean([abs(est[c] - true[c]) for c in range(n_classes)]))
