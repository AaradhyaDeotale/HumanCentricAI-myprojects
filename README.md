# Group 32: Human-Centric Artificial Intelligence - Project Portfolio

A Django-based web application developed as part of Group 32's submission for the *Human-Centric Artificial Intelligence* course at TUHH (SoSe 2026). The project bundles multiple interactive machine learning apps under a single launch page, each focusing on human-facing aspects of ML systems.

---

## Group

**Group 32** is a single-member group. This repository is the complete submission of:

| Name | Student ID |
|------|------------|
| Aaradhya Deotale | 680318 |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/AaradhyaDeotale/HumanCentricAI-myprojects.git
cd HumanCentricAI-myprojects
pip install -r requirements.txt
```

### Running the server

```bash
python manage.py runserver
```

Then open your browser at [http://127.0.0.1:8000/home/](http://127.0.0.1:8000/home/)

---

## Project Structure

```
HCAI-PBL/
├── home/               # Launch page with group info and project links
├── project1/           # Supervised Learning Interface
├── project2/           # Explainability
├── project3/           # Active Learning for Learning-to-Defer
│   └── ml/             # Framework-agnostic ML core
├── project4/           # Preference Elicitation user study
│   └── ml/             # Feature extraction + Plackett-Luce preference model
├── data/               # AG News CSVs + IMDB 5000 Movie Dataset CSV
├── pbl/                # Django project settings and root URLs
├── static/             # Global CSS
├── manage.py
└── requirements.txt
```

---

## Project 1 - Supervised Learning Interface

**URL:** `/project1/`

An end-to-end interface for supervised machine learning on tabular CSV data.

### Features

- **Data Upload** - Upload any CSV file where the first row contains feature names and the last column is the target label; the dataset persists in the session so it survives page reloads
- **Classification & Regression** - the problem type is auto-detected from the label column (continuous numeric -> regression, otherwise classification)
- **Data Visualization** - scatter plot of any two numeric features (class-colored for classification, colorbar-shaded by target for regression), plus a feature-correlation heatmap
- **Model Training Pipeline:**
  - Choose a model - K-Nearest Neighbours, Decision Tree, or SVM (classifier/regressor variant picked automatically)
  - Configure the train/test split ratio
  - Tune the model's hyperparameter (k / max depth / C), with a score-vs-hyperparameter curve showing the sweep
  - Evaluated with accuracy + classification report + confusion matrix (classification) or R²/RMSE (regression)
- **Training History** - every training run is persisted to the database (`TrainingRun`) with the best run so far highlighted; inspectable in the Django admin
- Built with **scikit-learn**

### Example Dataset

No dataset is bundled - upload any CSV with a header row and a trailing target column. The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (classification) or any tabular regression dataset work well as a quick test.

---

## Project 2 - Explainability

**URL:** `/project2/`

An interactive explainability dashboard built on the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/). Target variable: `species` (Adelie, Gentoo, Chinstrap).

### Features

#### Model Interpretability
- Fit a **Decision Tree** classifier and visualize the full tree structure
- Display test accuracy and number of leaves
- **λ slider** - interactively trade off accuracy vs. model complexity; the interface always shows the model minimizing `acc_test + λ · Ω(f)`
- Same interface for **Logistic Regression** with an appropriate complexity measure

#### Counterfactual Explanations (`/project2/api/counterfactuals/`)
- Select any data point and a desired target class
- Generates counterfactual examples via local random sampling + MAD-weighted L¹ distance ranking
- Handles numerical, binary, and categorical features appropriately
- Linked to the currently selected model type and λ value

#### Feature Effect Plots (`/project2/api/feature-effect/`)
- Select any of the four numerical features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`
- Displays both a **PDP** (Partial Dependence Plot) and an **ALE** (Accumulated Local Effects) plot
- Each plot shows three curves - one per species
- PDP and ALE computation implemented from scratch (no external library)
- Linked to the currently selected model type and λ value

#### Recorded Models (`/project2/api/record/`, `/project2/api/history/`)
- **"Record this model"** button snapshots the current model type, λ, complexity Ω and train/test accuracy to the database (`ModelSelectionRun` via the Django ORM)
- History table lists recorded models and highlights the best (highest test accuracy); inspectable in the Django admin

---

## Project 3 - Active Learning for Learning-to-Defer

**URL:** `/project3/`

A topic classifier for the **AG News** dataset that collaborates with a simulated human expert via a learning-to-defer policy, and discovers expert competence through active learning.

### Features

#### Task 1 - Baseline Classifier
- **TF-IDF + Logistic Regression** pipeline (fast, CPU-only, default)
- **DistilBERT** drop-in alternative (requires `torch` + `transformers`)
- Reports test accuracy on up to 20k training samples

#### Task 2 - Simulated Expert Team
- Multiple experts with localised, class-dependent competence profiles
- Per-class and overall accuracy reported for each expert

#### Task 3 - Deferral Policy
- Competence-aware threshold τ: defer to expert when expected expert accuracy exceeds classifier confidence
- System accuracy and coverage plotted against τ; curve downloadable as PNG

#### Task 4 - Active Learning
- Three query strategies: **random**, **uncertainty sampling**, **competence-gap**
- Discovers the expert's competence profile from a small query budget
- Convergence curves (L1 competence error vs. #queries) compared across strategies

#### Task 5 (optional) - Active Learning with a Human Expert
- "Be the expert" panel runs the same Task-4 acquisition loop, but *you* answer the queries instead of the simulated expert
- The chosen strategy picks the next training article; you read it, guess its topic, and get immediate right/wrong feedback plus your running accuracy
- Answers are graded against the true label to update a per-class Beta-posterior competence estimate - the same estimator used for the simulated experts, now fed by a real person
- Finished sessions are recorded to the run history as `human:<strategy>` entries alongside the simulated Task-4 runs

#### Live Demo
- Enter any text snippet to see the classifier's predicted class, confidence, and defer/predict decision
- Downloadable PDF report summarising all four tasks (plus a note on the optional Task 5 interface)

#### Run History (`/project3/history/`)
- Every baseline training (Task 1), deferral run (Task 3), active-learning run (Task 4) and human-in-the-loop session (Task 5) is persisted to the database (`TrainedModel`, `DeferralResult`, `ActiveLearningRun`)
- On-page history panel refreshes after each run and flags the best result in each category; also inspectable in the Django admin

### ML Core (`project3/ml/`)

| Module | Purpose |
|--------|---------|
| `data.py` | Loads AG News from local CSVs or HuggingFace |
| `classifiers.py` | TfidfLogReg and DistilBertClassifier |
| `expert.py` | Simulated experts with configurable competence |
| `defer.py` | Competence estimation and deferral system |
| `active.py` | Active learning loop and strategy implementations (also used by the Task 5 human-in-the-loop session) |

---

## Project 4 - Preference Elicitation

**URL:** `/project4/`

A user study comparing two interfaces for eliciting a new user's movie preferences from the **IMDB 5000 Movie Dataset**: repeated pairwise choices versus ranking lists of ten. The landing page offers (1) a downloadable PDF explaining the method and study design, and (2) a link into the participant-facing study itself, which can be run with real participants without further modification.

### Features

#### Task 1 - Feature Representation
- Each movie is vectorised into a 42-dimensional feature vector: multi-hot genres, bucketed content rating, log-scaled production/popularity stats (budget, gross, votes, cast/director Facebook likes), era, duration, and language/country flags
- No intercept term - utility differences are all that matter under a comparison-based likelihood, so a constant feature would be unidentifiable

#### Task 2 - Preference Model
- Standard **Bradley-Terry** model for pairwise comparisons, extended to full rankings via **Plackett-Luce** (sequential "pick the best of what's left" factorization); pairwise comparisons are just length-2 rankings, so both designs share one likelihood
- Preference vector `w` fit by maximum a posteriori estimation (L-BFGS with an analytic gradient, L2/Gaussian-prior regularization)

#### Task 3 - User Study Design
- Within-subjects, counterbalanced design: every participant does both interfaces, in randomized order, over exposure-matched movie budgets (30 movies either way)
- A held-out block of comparisons (never used for fitting) lets the study score each interface's fitted model on genuinely unseen choices
- Full protocol - hypotheses, recruitment, sample size, quality control, ethics, analysis plan - documented in the downloadable PDF report

#### Task 4 - Study Interface
- Consent screen -> randomized-order instructions -> pairwise-comparison trials / click-to-rank trials -> held-out comparisons -> post-study Likert questionnaire -> debrief screen
- Debrief screen shows personalised top-5 recommendations computed from the participant's own responses, a live comparison of how well the pairwise-only vs. ranking-only fitted model predicts held-out choices, and a completion code
- Every response is persisted to the database as it happens (`StudySession`, `PairwiseResponse`, `RankingResponse`, `Questionnaire`), so partial/abandoned sessions are never lost; inspectable in the Django admin

### ML Core (`project4/ml/`)

| Module | Purpose |
|--------|---------|
| `data.py` | Loads and cleans the IMDB 5000 Movie Dataset; builds the feature matrix |
| `preference.py` | Plackett-Luce log-likelihood, gradient, and MAP fitting of the preference vector |

---

## API Endpoints

### Project 2

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/project2/api/model/` | Returns model info for given `model_type` and `lambda` |
| GET | `/project2/api/feature-effect/` | Returns PDP + ALE data for a given feature |
| POST | `/project2/api/counterfactuals/` | Returns counterfactual examples for a given input point |
| POST | `/project2/api/record/` | Persists the current model selection and returns the updated history |
| GET | `/project2/api/history/` | Returns the recorded-model history |

### Project 3

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/project3/` | Main interface (classifier selection, task results, live demo) |
| GET | `/project3/train/` | Train the baseline classifier (Task 1); records a `TrainedModel` |
| GET | `/project3/defer/` | Deferral result for given τ and expert (Task 3); records a `DeferralResult` |
| GET | `/project3/active/` | Active learning results for given strategy (Task 4); records an `ActiveLearningRun` |
| GET | `/project3/classify/` | Classify a text snippet and return the defer decision |
| GET | `/project3/human/start/` | Task 5 (optional): start a human-in-the-loop session for the given strategy, returns the first query |
| GET | `/project3/human/answer/` | Task 5: grade the visitor's answer, update competence, and return the next query |
| GET | `/project3/human/finish/` | Task 5: end the session, record it as a `human:<strategy>` `ActiveLearningRun`, and return the summary |
| GET | `/project3/history/` | Returns the run history for all tasks |
| GET | `/project3/report/` | Download the PDF report |

### Project 4

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/project4/` | Landing page (download report, start study) |
| GET | `/project4/report/` | Download the PDF report (Tasks 1-3) |
| GET/POST | `/project4/study/` | Consent screen; POST creates a `StudySession` and begins the study |
| GET/POST | `/project4/study/task/` | Current instructions / pairwise / ranking trial; POST records a response and advances |
| GET/POST | `/project4/study/questionnaire/` | Post-study questionnaire; POST records a `Questionnaire` and generates a completion code |
| GET | `/project4/study/done/` | Debrief screen: personalised recommendations, model comparison, completion code |
| GET | `/project4/study/restart/` | Clears the in-progress study state and returns to consent |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Django |
| ML | scikit-learn, scipy, palmerpenguins |
| Data | pandas, numpy |
| Plots | matplotlib / custom JS |
| Reports | reportlab |
| Frontend | HTML, CSS, JavaScript |
| Optional | torch, transformers (DistilBERT classifier) |

---

## Course

**Human-Centric Artificial Intelligence**
Hamburg University of Technology (TUHH) - SoSe 2026

Submitted by **Group 32** - Aaradhya Deotale (student ID 680318), sole member.
