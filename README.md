# Human-Centric Artificial Intelligence — Project Portfolio

A Django-based web application developed for the *Human-Centric Artificial Intelligence* course at TUHH (SoSe 2026). The project bundles multiple interactive machine learning apps under a single launch page, each focusing on human-facing aspects of ML systems.

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
├── pbl/                # Django project settings and root URLs
├── static/             # Global CSS
├── manage.py
└── requirements.txt
```

---

## Project 1 — Supervised Learning Interface

**URL:** `/project1/`

An end-to-end interface for supervised machine learning on tabular CSV data.

### Features

- **Data Upload** — Upload any CSV file where the first row contains feature names and the last column is the target label
- **Data Visualization** — Scatter plots of selected feature pairs, color-coded by class label
- **Model Training Pipeline:**
  - Choose a ML algorithm (e.g. Decision Tree, K-Nearest Neighbours, etc.)
  - Configure train/test split ratio
  - Tune hyperparameters
  - Evaluate model performance with a chosen scoring metric
- Built with **scikit-learn**

### Example Dataset

The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is provided (`iris.csv`) — 150 samples, 4 features, 3 species classes.

---

## Project 2 — Explainability

**URL:** `/project2/`

An interactive explainability dashboard built on the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/). Target variable: `species` (Adelie, Gentoo, Chinstrap).

### Features

#### Model Interpretability
- Fit a **Decision Tree** classifier and visualize the full tree structure
- Display test accuracy and number of leaves
- **λ slider** — interactively trade off accuracy vs. model complexity; the interface always shows the model minimizing `acc_test + λ · Ω(f)`
- Same interface for **Logistic Regression** with an appropriate complexity measure

#### Counterfactual Explanations (`/project2/api/counterfactuals/`)
- Select any data point and a desired target class
- Generates counterfactual examples via local random sampling + MAD-weighted L¹ distance ranking
- Handles numerical, binary, and categorical features appropriately
- Linked to the currently selected model type and λ value

#### Feature Effect Plots (`/project2/api/feature-effect/`)
- Select any of the four numerical features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`
- Displays both a **PDP** (Partial Dependence Plot) and an **ALE** (Accumulated Local Effects) plot
- Each plot shows three curves — one per species
- PDP and ALE computation implemented from scratch (no external library)
- Linked to the currently selected model type and λ value

---

## API Endpoints (Project 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/project2/api/model/` | Returns model info for given `model_type` and `lambda` |
| GET | `/project2/api/feature-effect/` | Returns PDP + ALE data for a given feature |
| POST | `/project2/api/counterfactuals/` | Returns counterfactual examples for a given input point |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Django |
| ML | scikit-learn, palmerpenguins |
| Data | pandas, numpy |
| Plots | matplotlib / custom JS |
| Frontend | HTML, CSS, JavaScript |

---

## Course

**Human-Centric Artificial Intelligence**
Hamburg University of Technology (TUHH) — SoSe 2026
