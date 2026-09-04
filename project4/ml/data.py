"""
Dataset loading and feature extraction for Project 4 (Task 1).

The IMDB 5000 Movie Dataset has no user ratings at all -- it is pure movie
metadata. To turn a row into a feature vector x usable in a linear utility
U(x) = w^T x, we pick attributes that plausibly drive an individual's taste:

  - genres            what kind of story/content it is (multi-hot)
  - content rating    who it's aimed at / how graphic it is (one-hot, bucketed)
  - title_year        era / recency
  - duration          length
  - budget, gross     scale of production
  - cast/director fb likes   star power
  - num_voted_users   how mainstream/well-known the film is
  - imdb_score        critical consensus (a taste for acclaimed vs. "guilty
                       pleasure" films is itself a preference dimension)
  - language, country reduced to is_english / is_usa flags

We deliberately do NOT add an intercept ("always-on") feature. Under the
Bradley-Terry / Plackett-Luce likelihood, only utility *differences* within a
comparison set matter (see project4.ml.preference); a feature that is 1 for
every item contributes the same constant to every item's utility and cancels
out of every softmax, so its weight is unidentifiable. Leaving it out keeps
the parameter vector estimable from the small number of comparisons a
participant can realistically provide.
"""

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "Game-Show",
    "History", "Horror", "Music", "Musical", "Mystery", "News", "Reality-TV",
    "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "War", "Western",
]

RATING_BUCKETS = ["G", "PG", "PG-13", "R", "NC-17"]
RATING_CATEGORIES = RATING_BUCKETS + ["Other"]

# (column, apply log1p before z-scoring -- these are heavy-tailed counts/money)
NUMERIC_COLUMNS = [
    ("title_year", False),
    ("duration", False),
    ("num_voted_users", True),
    ("budget", True),
    ("gross", True),
    ("cast_total_facebook_likes", True),
    ("director_facebook_likes", True),
    ("imdb_score", False),
]


def load_movies(csv_path):
    """Load and clean the raw IMDB 5000 Movie Dataset CSV.

    Drops rows with no title/genres and exact (title, year) duplicates,
    which the dataset has ~240 of (e.g. re-releases scraped twice) -- left
    in, a participant could be asked to compare a movie against itself.
    """
    df = pd.read_csv(csv_path)
    df["movie_title"] = df["movie_title"].astype(str).str.strip()
    df = df.dropna(subset=["genres", "movie_title"])
    df = df.drop_duplicates(subset=["movie_title", "title_year"], keep="first")
    df = df.reset_index(drop=True)
    return df


def _rating_bucket(value):
    return value if value in RATING_BUCKETS else "Other"


def build_features(df):
    """Task 1: turn the cleaned dataframe into a dense feature matrix.

    Returns (X, feature_names) where X has shape (n_movies, n_features).
    Numeric columns are median-imputed then standardized; categorical
    columns are one-/multi-hot encoded.
    """
    n = len(df)
    blocks, names = [], []

    genre_mat = np.zeros((n, len(GENRES)), dtype=float)
    for i, g in enumerate(df["genres"].fillna("")):
        tokens = set(g.split("|"))
        for j, name in enumerate(GENRES):
            if name in tokens:
                genre_mat[i, j] = 1.0
    blocks.append(genre_mat)
    names += [f"genre:{g}" for g in GENRES]

    buckets = df["content_rating"].apply(_rating_bucket)
    rating_mat = np.zeros((n, len(RATING_CATEGORIES)), dtype=float)
    for i, b in enumerate(buckets):
        rating_mat[i, RATING_CATEGORIES.index(b)] = 1.0
    blocks.append(rating_mat)
    names += [f"rating:{c}" for c in RATING_CATEGORIES]

    numeric_mat = np.zeros((n, len(NUMERIC_COLUMNS)), dtype=float)
    for j, (col, log) in enumerate(NUMERIC_COLUMNS):
        vals = df[col].astype(float)
        vals = vals.fillna(vals.median())
        if log:
            vals = np.log1p(vals.clip(lower=0))
        mean, std = vals.mean(), vals.std()
        std = std if std > 1e-8 else 1.0
        numeric_mat[:, j] = (vals - mean) / std
    blocks.append(numeric_mat)
    names += [f"num:{col}" for col, _ in NUMERIC_COLUMNS]

    is_english = (df["language"] == "English").astype(float).to_numpy().reshape(-1, 1)
    is_usa = (df["country"] == "USA").astype(float).to_numpy().reshape(-1, 1)
    blocks.append(np.hstack([is_english, is_usa]))
    names += ["is_english", "is_usa"]

    X = np.hstack(blocks)
    return X, names
