"""
Preference model for Project 4 (Task 2).

Standard Bradley-Terry gives the probability that item i beats item j from
their utilities U(x) = w^T x:

    P(i > j) = exp(U_i) / (exp(U_i) + exp(U_j))

Design 2 of the user study asks for a full ranking i_1 > i_2 > ... > i_n
instead of a single pairwise winner. We extend Bradley-Terry to rankings by
modelling the ranking as a sequence of "pick the best of what's left"
choices: first the winner is picked from all n items by a Bradley-Terry-like
contest among all of them, then the runner-up is picked the same way from
the remaining n-1, and so on down to the last two. This gives the
Plackett-Luce model:

    P(i_1 > i_2 > ... > i_n) = prod_{k=1}^{n-1} exp(U_{i_k}) / sum_{l=k}^{n} exp(U_{i_l})

Setting n=2 collapses this to exactly the Bradley-Terry formula above, so
pairwise comparisons and top-to-bottom rankings can be fit with the same
likelihood -- a comparison is just a ranking of length 2. This is what lets
`fit_preferences` below combine Design-1 and Design-2 responses into one
estimate of w.

Fitting: w is found by maximum a posteriori estimation -- maximizing the
Plackett-Luce log-likelihood of the observed rankings with a zero-mean
Gaussian prior on w (equivalently, L2-regularized negative log-likelihood).
The prior is necessary because with only a handful of comparisons and a
few dozen features, unregularized MLE can diverge (perfectly separable
choices push some weights to +/-infinity).
"""

import numpy as np
from scipy.optimize import minimize


def _neg_log_likelihood_and_grad(w, X, rankings, l2):
    """Negative log-posterior (NLL + L2 penalty) and its gradient w.r.t. w."""
    utilities = X @ w
    nll = 0.5 * l2 * np.dot(w, w)
    grad = l2 * w

    for ranking in rankings:
        ids = np.asarray(ranking)
        n = len(ids)
        for k in range(n - 1):
            remaining = ids[k:]
            u_remaining = utilities[remaining]
            m = u_remaining.max()
            logsumexp = m + np.log(np.sum(np.exp(u_remaining - m)))
            nll -= utilities[ids[k]] - logsumexp

            probs = np.exp(u_remaining - logsumexp)  # softmax over remaining items
            grad += -X[ids[k]] + probs @ X[remaining]

    return nll, grad


def fit_preferences(X, rankings, l2=5.0):
    """MAP estimate of the preference vector w from a list of rankings.

    Each entry of `rankings` is a sequence of row-indices into X, ordered
    from most to least preferred (a pairwise choice is a ranking of length
    2: [winner, loser]).
    """
    d = X.shape[1]
    if not rankings:
        return np.zeros(d)

    w0 = np.zeros(d)
    result = minimize(
        _neg_log_likelihood_and_grad, w0, args=(X, rankings, l2),
        jac=True, method="L-BFGS-B",
    )
    return result.x


def utilities(X, w):
    """U(x) = w^T x for every row of X."""
    return X @ w
