# Project 3 — Active Learning for Learning-to-Defer

Topic classifier for **AG News** that collaborates with a simulated human
expert via learning-to-defer, and discovers expert competence through
active learning.

## Tasks implemented
- **Task 1** — baseline classifier (TF-IDF + LogReg; DistilBERT drop-in)
- **Task 2** — simulated experts with localised, class-dependent competence
- **Task 3** — competence-aware deferral policy + deferral-quality metrics
- **Task 4** — active learning (random / uncertainty / competence-gap) to
  learn the expert's competence profile from a small query budget
- PDF report, downloadable from the interface

## Setup
1. Install deps:
   ```
   pip install django scikit-learn numpy matplotlib reportlab
   # optional, for the DistilBERT classifier:
   pip install torch transformers datasets
   ```
2. Provide the AG News dataset as two CSVs in `data/agnews/`:
   `train.csv`, `test.csv` in the classic format `label,title,description`
   (label in 1..4). A copy is already included.
   Alternatively set `AGNEWS_DIR` to point elsewhere, or switch
   `data.load_agnews(source="hf")` to pull `fancyzhx/ag_news` from
   HuggingFace.
3. Migrate and run:
   ```
   python manage.py migrate
   python manage.py runserver
   ```
4. Open http://127.0.0.1:8000/project3/

## Structure
- `ml/` — framework-agnostic ML core (data, classifiers, expert, defer, active)
- `services.py` — trains/caches the model, runs tasks, renders plots
- `views.py`, `urls.py` — Django endpoints
- `report.py` — generates the downloadable PDF report
- `templates/project3/index.html` — interactive interface
