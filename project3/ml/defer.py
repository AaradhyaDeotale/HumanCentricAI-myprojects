"""
Learning to Defer (Task 3).

Setting: both the classifier and expert labels are available. For each
input the SYSTEM chooses to either (a) predict with the classifier or
(b) defer to the expert. We want the combined system to beat the
classifier-alone baseline.

We implement a competence-aware deferral rule. For input x:

    p_clf(x)    = classifier's probability of being correct, estimated
                  as its max softmax probability  max_k P(y=k | x).
    p_exp(x)    = expert's estimated probability of being correct on x.

We defer when the expert's expected correctness exceeds the classifier's
by a margin (a cost knob that lets the user trade off how often the
expert is bothered):

    defer  iff  p_exp(x) - p_clf(x) > tau

Because the expert's true competence is class-dependent and we do not
know the true label at decision time, p_exp(x) is estimated from the
classifier's predicted-class distribution and a learned per-class expert
competence vector `expert_acc_by_class` (estimated on the held-in
expert labels, exactly the kind of quantity Task 4 learns actively).

Evaluation reports:
    - system accuracy
    - classifier-alone accuracy (baseline)
    - expert-alone accuracy
    - coverage (fraction NOT deferred)
    - deferral quality: among deferred points, how often the expert was
      actually the better choice, and the same for kept points.
"""

import numpy as np


def estimate_expert_competence(expert, val_labels, val_ids, n_classes):
    """
    Per-class empirical accuracy of the expert, estimated from a set of
    expert labels. This is the only expert knowledge the deferral policy
    is allowed to use (no peeking at test labels).
    """
    preds = expert.query_batch(val_labels, val_ids)
    acc = np.full(n_classes, 0.5)  # prior if a class is unseen
    counts = np.zeros(n_classes)
    correct = np.zeros(n_classes)
    for p, y in zip(preds, val_labels):
        counts[y] += 1
        correct[y] += int(p == y)
    for c in range(n_classes):
        if counts[c] > 0:
            acc[c] = correct[c] / counts[c]
    return acc


class DeferralSystem:
    def __init__(self, classifier, expert, expert_acc_by_class, tau=0.0):
        self.clf = classifier
        self.expert = expert
        self.expert_acc = np.asarray(expert_acc_by_class, dtype=float)
        self.tau = tau

    def decide(self, texts):
        """Return (defer_mask, clf_preds, clf_conf, exp_expected)."""
        proba = self.clf.predict_proba(texts)
        clf_preds = np.argmax(proba, axis=1)
        clf_conf = proba.max(axis=1)
        # expected expert correctness = sum_k P(y=k|x) * expert_acc[k]
        exp_expected = proba @ self.expert_acc
        defer_mask = (exp_expected - clf_conf) > self.tau
        return defer_mask, clf_preds, clf_conf, exp_expected

    def predict(self, texts, true_labels, example_ids):
        """
        Produce the system prediction for each input. Deferred points get
        the expert's (queried) label; kept points get the classifier's.
        true_labels/example_ids are needed to *simulate* querying.
        """
        defer_mask, clf_preds, _, _ = self.decide(texts)
        preds = clf_preds.copy()
        for i in np.where(defer_mask)[0]:
            preds[i] = self.expert.query(true_labels[i], example_ids[i])
        return preds, defer_mask


def evaluate_system(system, texts, true_labels, example_ids):
    true_labels = np.asarray(true_labels)
    preds, defer_mask = system.predict(texts, true_labels, example_ids)

    proba = system.clf.predict_proba(texts)
    clf_preds = np.argmax(proba, axis=1)
    clf_correct = (clf_preds == true_labels)

    exp_preds = np.array(system.expert.query_batch(true_labels, example_ids))
    exp_correct = (exp_preds == true_labels)

    sys_acc = float(np.mean(preds == true_labels))
    coverage = float(np.mean(~defer_mask))

    # deferral quality: was deferring the right call?
    # "right to defer" = expert correct AND classifier wrong (or tie-correct)
    deferred = defer_mask
    kept = ~defer_mask
    good_defers = np.sum(deferred & exp_correct & ~clf_correct)
    bad_defers = np.sum(deferred & ~exp_correct & clf_correct)
    good_keeps = np.sum(kept & clf_correct)
    bad_keeps = np.sum(kept & ~clf_correct & exp_correct)

    n_def = max(int(deferred.sum()), 1)
    n_kept = max(int(kept.sum()), 1)

    return {
        "system_accuracy": sys_acc,
        "classifier_accuracy": float(np.mean(clf_correct)),
        "expert_accuracy": float(np.mean(exp_correct)),
        "coverage": coverage,
        "deferral_rate": float(np.mean(deferred)),
        "good_defer_rate": good_defers / n_def,
        "bad_defer_rate": bad_defers / n_def,
        "good_keep_rate": good_keeps / n_kept,
        "bad_keep_rate": bad_keeps / n_kept,
    }
