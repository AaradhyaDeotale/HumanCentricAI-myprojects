# Project 2 — Explainability (HCAI)

A Django app (`project2`) implementing all five tasks of the Explainability
project on the **Palmer Penguins** dataset. The whole thing is one interactive,
single-page dashboard: pick a **model class** (decision tree or logistic
regression) and a **regularisation level λ**, and every panel — the model view,
counterfactuals and feature-effect plots — updates *live* (AJAX, no page reload)
from that one selected model.

## What's where

```
project2/
├── ml.py                       # ALL the machine-learning / explainability logic
│                               # (data, model families, λ-selection, counterfactuals,
│                               #  PDP and ALE). No Django imports — unit-testable.
├── views.py                    # thin JSON API + the single page view
├── urls.py                     # routes: page + 4 API endpoints
├── tests.py                    # `python manage.py test project2`
├── models.py / admin.py        # intentionally empty (no DB needed)
├── templates/project2/index.html
└── static/project2/
    ├── app.js                  # state, sliders, fetch calls, all Plotly rendering
    └── style.css               # project-specific styling (blue theme, restrained)
```

## Setup

1. **Drop the `project2/` folder** into the root of the HCAI-PBL project (next to
   `manage.py`, alongside `home/` and `demos/`).

2. **Install the one extra dependency** (everything else is already used by the
   skeleton):

   ```bash
   pip install palmerpenguins scikit-learn pandas numpy
   ```

3. **Three integration edits** — these were *already applied* to the project we
   sent back, but here they are for reference / if you re-clone the skeleton:

   * `pbl/settings.py` — add the app to `INSTALLED_APPS`:
     ```python
     INSTALLED_APPS = [
         ...
         "demos",
         "project2",
     ]
     ```
   * `pbl/urls.py` — include the app's URLs:
     ```python
     path("project2/", include("project2.urls")),
     ```
   * `home/views.py` — add a link on the home page (inside the `projects` list):
     ```python
     {"name": "Project 2: Explainability", "url_name": "project2:index"},
     ```

4. **Run it**:

   ```bash
   python manage.py runserver
   ```
   Then open <http://127.0.0.1:8000/project2/> (or click the link from the home
   page). The first request trains and caches the model families (a couple of
   seconds); everything is instant afterwards.

## How each task is covered

* **Task 1 — Decision tree.** A `DecisionTreeClassifier` is fitted; the page draws
  the resulting tree (split conditions in white nodes, leaves coloured by
  species with sample counts) and shows test accuracy + number of leaves in the
  summary chips. A "show tree as text" view is included too.

* **Task 2 — λ slider (trees).** We pre-train a family of trees across
  `max_leaf_nodes`, then the λ slider *selects* (never retrains) the model that
  optimises the accuracy/complexity trade-off, with Ω = number of leaves. An
  "accuracy vs. complexity" plot marks the currently selected model.

  **Sign convention (deliberate, see the long comment in `ml.py`):** the brief
  prints `acc_test + λ·Ω`, but also calls λ a *regularisation* parameter. Taken
  literally with accuracy *maximisation*, `+λ·Ω` would reward complexity — the
  opposite of regularisation. We therefore select the model maximising
  `acc_test − λ·Ω`, so λ = 0 gives the most accurate model and larger λ gives
  progressively simpler ones. This is documented in code.

* **Task 3 — Logistic regression.** Same λ machinery with an L1-penalised
  multinomial logistic regression; Ω = number of non-zero coefficients
  (sparsity). The tree view is replaced by a grouped coefficient bar chart.

* **Task 4 — Counterfactuals.** Implemented by hand exactly as described: sample
  N points locally around x, keep those predicted as the target class, rank by
  **MAD-weighted L1 distance**, return the best k. Numeric features get Gaussian
  noise scaled by their spread; **categorical / one-hot features** (island, sex)
  are noised by occasionally re-drawing the active category. If none are found,
  N is doubled and the variance widened, iteratively (up to 5 rounds). The panel
  is linked to the selected model class and λ.

* **Task 5 — PDP & ALE.** Both written from scratch (no explainability library),
  each producing **three curves (one per species)** and linked to the selected
  model / λ. ALE uses the **exact softmax derivative** for logistic regression
  (`dp_c/dx_j = p_c (w_{c,j} − Σ_k p_k w_{k,j}) / σ_j`, the 1/σ accounting for
  the StandardScaler) and a **finite-difference discretisation** for trees, which
  are piecewise-constant and non-differentiable — exactly the distinction the
  brief asks you to reason about.

## Notes

* Plots use **Plotly.js** via CDN. If you must run fully offline, download
  `plotly-2.32.0.min.js` into `static/` and point the one `<script>` tag in
  `templates/project2/index.html` at it.
* No database is used — models live in an in-memory cache in `ml.py`. You still
  need to have run `migrate` once for Django's session/CSRF machinery (the
  skeleton already does).
* `python manage.py test project2` runs five quick sanity checks (data loads,
  Ω is monotone, λ simplifies the model, PDP probabilities sum to 1,
  counterfactuals hit the target class).
```
