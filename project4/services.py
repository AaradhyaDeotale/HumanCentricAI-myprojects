"""
Service layer for Project 4. Bridges the Django views and the ML core in
`project4.ml`: loads/caches the featurized movie catalogue, formats movies
for display, and fits a participant's preference vector from their stored
responses.
"""

import json
import os
import threading

import numpy as np
import pandas as pd
from django.conf import settings

from .ml.data import load_movies, build_features
from .ml.preference import fit_preferences, utilities
from .models import PairwiseResponse, RankingResponse

_LOCK = threading.Lock()
_CACHE = {}  # csv_path -> (df, X, feature_names)

L2_REGULARIZATION = 5.0


def _data_path():
    """Where to find movie_metadata.csv. Override with IMDB5000_CSV."""
    return os.environ.get(
        "IMDB5000_CSV",
        os.path.join(settings.BASE_DIR, "data", "imdb5000", "movie_metadata.csv"))


def dataset():
    """Load (or fetch from cache) the cleaned dataframe + feature matrix."""
    path = _data_path()
    with _LOCK:
        if path in _CACHE:
            return _CACHE[path]

    df = load_movies(path)
    X, names = build_features(df)

    with _LOCK:
        _CACHE[path] = (df, X, names)
    return _CACHE[path]


def n_movies():
    df, _, _ = dataset()
    return len(df)


def _clean(value, default=""):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return value


def movie_brief(idx, df, reveal_score=False):
    """A participant-facing summary of one movie.

    IMDb score and vote count are withheld by default: showing a "quality"
    signal during elicitation would encourage participants to pick the more
    acclaimed/popular movie rather than the one they'd personally rather
    watch, contaminating the preference signal we are trying to measure
    (see the anchoring discussion in the Task 3 report). They are only
    revealed on the final results screen, after elicitation is complete.
    """
    row = df.iloc[idx]
    year = row["title_year"]
    genres = [g for g in str(row["genres"]).split("|") if g][:3]
    actors = [a for a in (row.get("actor_1_name"), row.get("actor_2_name"),
                          row.get("actor_3_name")) if isinstance(a, str)]
    keywords = [k for k in str(_clean(row.get("plot_keywords"), "")).split("|") if k][:4]

    brief = {
        "id": int(idx),
        "title": row["movie_title"],
        "year": int(year) if not pd.isna(year) else None,
        "genres": genres,
        "director": _clean(row.get("director_name"), "Unknown"),
        "actors": actors,
        "rating": _clean(row.get("content_rating"), "Not Rated"),
        "keywords": keywords,
    }
    if reveal_score:
        brief["imdb_score"] = round(float(row["imdb_score"]), 1)
    return brief


def _rankings(session_id, include_pairwise=True, include_ranking=True):
    """Collect this session's responses as Plackett-Luce rankings.

    Also returns the set of movie ids the participant has already seen,
    so recommendations don't just point back at what they were shown.
    """
    rankings, seen = [], set()

    if include_pairwise:
        qs = PairwiseResponse.objects.filter(session_id=session_id, phase="pairwise")
        for r in qs:
            other = r.movie_b if r.chosen == r.movie_a else r.movie_a
            rankings.append([r.chosen, other])
            seen.update([r.movie_a, r.movie_b])

    if include_ranking:
        qs = RankingResponse.objects.filter(session_id=session_id)
        for r in qs:
            ids = json.loads(r.movie_ids_json)
            rankings.append(ids)
            seen.update(ids)

    return rankings, seen


def _holdout_accuracy(X, w, holdout):
    """Fraction of held-out pairwise trials whose winner U(w) predicts correctly."""
    if not holdout or w is None:
        return None
    u = utilities(X, w)
    correct = sum(
        1 for r in holdout
        if (u[r.movie_a] >= u[r.movie_b]) == (r.chosen == r.movie_a)
    )
    return correct / len(holdout)


def session_summary(session_id, top_k=5, l2=L2_REGULARIZATION):
    """Everything shown on the debrief screen: personalised recommendations
    plus a peek at which interface's fitted model better predicts the
    participant's held-out choices."""
    df, X, _names = dataset()

    combined, seen = _rankings(session_id, include_pairwise=True, include_ranking=True)
    holdout = list(PairwiseResponse.objects.filter(session_id=session_id, phase="holdout"))

    recommendations = []
    if combined:
        w = fit_preferences(X, combined, l2=l2)
        u = utilities(X, w)
        for idx in np.argsort(-u):
            idx = int(idx)
            if idx in seen:
                continue
            recommendations.append(movie_brief(idx, df, reveal_score=True))
            if len(recommendations) >= top_k:
                break

    pairwise_only, _ = _rankings(session_id, include_pairwise=True, include_ranking=False)
    ranking_only, _ = _rankings(session_id, include_pairwise=False, include_ranking=True)
    w_pairwise = fit_preferences(X, pairwise_only, l2=l2) if pairwise_only else None
    w_ranking = fit_preferences(X, ranking_only, l2=l2) if ranking_only else None

    return {
        "recommendations": recommendations,
        "n_holdout": len(holdout),
        "pairwise_accuracy": _holdout_accuracy(X, w_pairwise, holdout),
        "ranking_accuracy": _holdout_accuracy(X, w_ranking, holdout),
    }
