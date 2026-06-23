"""
Simulated experts (Task 2).

The brief requires experts that are NOT perfect and whose expertise is
localised to specific regions of the input space. We implement experts
whose competence is class-dependent: an expert is highly reliable on a
subset of "specialty" classes and unreliable elsewhere. This creates a
non-trivial deferral problem -- deferring is only worthwhile on inputs
that fall in the expert's specialty.

Each expert, when queried on example x with true label y, returns a
(possibly wrong) label. The behaviour is deterministic given the example
index (via a per-example seed) so that querying the same point twice
returns the same answer -- important for active learning where we must
not be able to "average out" expert noise by re-querying.

ExpertProfile records, per class, the probability the expert is correct.
"""

import numpy as np


class ClassSpecialistExpert:
    """
    Expert that is competent on `specialty` classes and weak elsewhere.

    On a specialty class: correct with prob `p_high`.
    On a non-specialty class: correct with prob `p_low`; when wrong,
    answers with a random other class.
    """

    def __init__(self, specialty, n_classes, p_high=0.95, p_low=0.30,
                 seed=0, name="expert"):
        self.specialty = set(specialty)
        self.n_classes = n_classes
        self.p_high = p_high
        self.p_low = p_low
        self.seed = seed
        self.name = name

    def _rng(self, example_id):
        # deterministic per (expert, example) so re-querying is stable
        return np.random.default_rng((self.seed * 1_000_003 + example_id)
                                     % (2**32))

    def query(self, true_label, example_id):
        rng = self._rng(example_id)
        p_correct = self.p_high if true_label in self.specialty else self.p_low
        if rng.random() < p_correct:
            return int(true_label)
        others = [c for c in range(self.n_classes) if c != true_label]
        return int(rng.choice(others))

    def query_batch(self, true_labels, example_ids):
        return [self.query(y, i) for y, i in zip(true_labels, example_ids)]

    def competence_profile(self):
        """Ground-truth per-class accuracy (for analysis/reporting)."""
        return {c: (self.p_high if c in self.specialty else self.p_low)
                for c in range(self.n_classes)}


def default_expert_team(n_classes=4, seed=0):
    """
    A small team of complementary experts. Together they cover all
    classes, but no single expert is good everywhere.
    """
    return [
        ClassSpecialistExpert(specialty={0, 1}, n_classes=n_classes,
                              p_high=0.95, p_low=0.25, seed=seed + 1,
                              name="expert_world_sports"),
        ClassSpecialistExpert(specialty={2, 3}, n_classes=n_classes,
                              p_high=0.93, p_low=0.25, seed=seed + 2,
                              name="expert_biz_tech"),
    ]


def evaluate_expert(expert, labels, example_ids=None):
    """Empirical accuracy of an expert on a labelled set."""
    if example_ids is None:
        example_ids = list(range(len(labels)))
    preds = expert.query_batch(labels, example_ids)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    # per-class breakdown
    per_class = {}
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        if idx:
            acc_c = np.mean([preds[i] == labels[i] for i in idx])
            per_class[c] = float(acc_c)
    return {"accuracy": correct / len(labels), "per_class": per_class}
