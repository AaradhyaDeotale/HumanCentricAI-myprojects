import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render, redirect
from .models import TrainingRun
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error,
)


def fig_to_base64(fig):
    """Render a matplotlib figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def detect_problem_type(y):
    """Decide classification vs regression from the label column.

    A continuous numeric column with many distinct values -> regression.
    Otherwise (text labels, or few distinct integers) -> classification.
    """
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        # Heuristic: lots of distinct numeric values means a continuous target.
        if n_unique > 20 and n_unique > 0.2 * len(y):
            return 'regression'
    return 'classification'


def index(request):
    context = {}

    # --- 1. Load data: uploaded file, else session, else empty page ---
    if request.method == 'POST':
        if request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']
            if not csv_file.name.lower().endswith('.csv'):
                context['error'] = "Please upload a file with a .csv extension."
                return render(request, 'project1/index.html', context)
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                context['error'] = f"Could not read the CSV file: {e}"
                return render(request, 'project1/index.html', context)
            if df.shape[1] < 2:
                context['error'] = "The CSV needs at least two columns (features + label)."
                return render(request, 'project1/index.html', context)
            request.session['csv_data'] = df.to_json()
            request.session['dataset_name'] = csv_file.name
        elif request.session.get('csv_data'):
            df = pd.read_json(io.StringIO(request.session['csv_data']))
        else:
            return render(request, 'project1/index.html', context)
    else:
        if request.session.get('csv_data'):
            df = pd.read_json(io.StringIO(request.session['csv_data']))
        else:
            return render(request, 'project1/index.html', context)

    # --- 2. Basic dataset info ---
    label_col = df.columns[-1]
    feature_cols = list(df.columns[:-1])
    numeric_features = [c for c in feature_cols
                        if pd.api.types.is_numeric_dtype(df[c])]

    y_all = df[label_col]
    problem_type = detect_problem_type(y_all)

    context['columns'] = feature_cols          # only features are selectable axes
    context['numeric_features'] = numeric_features
    context['label_col'] = label_col
    context['problem_type'] = problem_type
    context['table'] = df.head(10).to_html(classes='data-table', index=False)
    context['shape'] = df.shape

    if not numeric_features:
        context['error'] = "No numeric feature columns were found to plot or train on."
        return render(request, 'project1/index.html', context)

    # --- 3. Scatter plot (axes restricted to numeric features) ---
    default_x = numeric_features[0]
    default_y = numeric_features[1] if len(numeric_features) > 1 else numeric_features[0]
    x_col = request.POST.get('x_col', default_x)
    y_col = request.POST.get('y_col', default_y)
    # Guard against a stale/invalid selection.
    if x_col not in numeric_features:
        x_col = default_x
    if y_col not in numeric_features:
        y_col = default_y

    fig, ax = plt.subplots(figsize=(8, 5))
    if problem_type == 'classification':
        for label in y_all.unique():
            subset = df[df[label_col] == label]
            ax.scatter(subset[x_col], subset[y_col], label=str(label), s=20)
        ax.legend(title=label_col)
    else:
        # Regression: color points by the continuous target value.
        sc = ax.scatter(df[x_col], df[y_col], c=y_all, cmap='viridis', s=20)
        fig.colorbar(sc, ax=ax, label=label_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    context['plot'] = fig_to_base64(fig)
    context['x_col'] = x_col
    context['y_col'] = y_col

    # --- 3b. Correlation heatmap of numeric features (extra visualization) ---
    if len(numeric_features) >= 2:
        corr = df[numeric_features].corr()
        figh, axh = plt.subplots(figsize=(6, 5))
        im = axh.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        axh.set_xticks(range(len(numeric_features)))
        axh.set_yticks(range(len(numeric_features)))
        axh.set_xticklabels(numeric_features, rotation=45, ha='right')
        axh.set_yticklabels(numeric_features)
        for i in range(len(numeric_features)):
            for j in range(len(numeric_features)):
                axh.text(j, i, f'{corr.iloc[i, j]:.2f}',
                         ha='center', va='center', fontsize=8)
        figh.colorbar(im, ax=axh)
        axh.set_title('Feature Correlation')
        context['heatmap'] = fig_to_base64(figh)

    # --- 4. Model training ---
    model_name = request.POST.get('model')
    try:
        test_size = int(request.POST.get('test_size', 20)) / 100
        hyperparameter = int(request.POST.get('hyperparameter', 5))
    except (TypeError, ValueError):
        test_size, hyperparameter = 0.20, 5
    test_size = min(max(test_size, 0.10), 0.50)
    hyperparameter = max(hyperparameter, 1)
    context['test_size'] = int(test_size * 100)
    context['model_name'] = model_name
    context['hyperparameter'] = hyperparameter

    if model_name and model_name in ['knn', 'decision_tree', 'svm']:
        # Train only on numeric features; drop rows with missing values.
        data = df[numeric_features + [label_col]].dropna()
        X = data[numeric_features]
        y = data[label_col]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)

            def make_model(name, p, problem):
                if problem == 'classification':
                    if name == 'knn':
                        return KNeighborsClassifier(n_neighbors=p), 'K (neighbors)'
                    if name == 'decision_tree':
                        return DecisionTreeClassifier(max_depth=p, random_state=42), 'Max Depth'
                    return SVC(C=p), 'C (regularization)'
                else:
                    if name == 'knn':
                        return KNeighborsRegressor(n_neighbors=p), 'K (neighbors)'
                    if name == 'decision_tree':
                        return DecisionTreeRegressor(max_depth=p, random_state=42), 'Max Depth'
                    return SVR(C=p), 'C (regularization)'

            model, hyperparam_label = make_model(model_name, hyperparameter, problem_type)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            context['hyperparam_label'] = hyperparam_label

            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                context['accuracy'] = round(accuracy * 100, 2)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                context['report'] = pd.DataFrame(report).transpose().round(2).to_html(classes='data-table')

                # Confusion matrix (extra)
                labels_sorted = sorted(y.unique().tolist(), key=str)
                cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
                figc, axc = plt.subplots(figsize=(5, 4))
                im = axc.imshow(cm, cmap='Blues')
                axc.set_xticks(range(len(labels_sorted)))
                axc.set_yticks(range(len(labels_sorted)))
                axc.set_xticklabels([str(l) for l in labels_sorted], rotation=45, ha='right')
                axc.set_yticklabels([str(l) for l in labels_sorted])
                for i in range(len(labels_sorted)):
                    for j in range(len(labels_sorted)):
                        axc.text(j, i, cm[i, j], ha='center', va='center',
                                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
                axc.set_xlabel('Predicted')
                axc.set_ylabel('Actual')
                axc.set_title('Confusion Matrix')
                context['confusion'] = fig_to_base64(figc)
                score_label = 'Accuracy'
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                context['accuracy'] = round(r2, 3)
                context['rmse'] = round(rmse, 3)
                score_label = 'R\u00b2 score'
            context['score_label'] = score_label

            # --- Score vs hyperparameter curve ---
            param_range = list(range(1, 21))
            scores = []
            for p in param_range:
                m, _ = make_model(model_name, p, problem_type)
                m.fit(X_train, y_train)
                pred = m.predict(X_test)
                if problem_type == 'classification':
                    scores.append(accuracy_score(y_test, pred))
                else:
                    scores.append(r2_score(y_test, pred))

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(param_range, scores, marker='o')
            ax2.axvline(x=hyperparameter, color='red', linestyle='--', label='Selected')
            ax2.set_xlabel(hyperparam_label)
            ax2.set_ylabel(score_label)
            ax2.set_title(f'{score_label} vs Hyperparameter')
            ax2.legend()
            context['acc_plot'] = fig_to_base64(fig2)

            # --- Persist this run to the database (Django ORM) ---
            TrainingRun.objects.create(
                dataset_name=request.session.get('dataset_name', 'uploaded dataset'),
                model_name=model_name,
                problem_type=problem_type,
                hyperparameter=hyperparameter,
                hyperparameter_label=hyperparam_label,
                test_size=int(test_size * 100),
                score=context['accuracy'],
                score_label=score_label,
            )

        except Exception as e:
            context['train_error'] = f"Training failed: {e}"

    # --- Training history from the database (most recent first) ---
    history = TrainingRun.objects.all()[:15]
    context['history'] = history
    if history:
        # Best run = highest score among classification runs, or highest R2
        # among regression runs. We keep it simple: highest score overall,
        # since both accuracy and R2 are "higher is better".
        context['best_run'] = max(history, key=lambda r: r.score)
        context['total_runs'] = TrainingRun.objects.count()

    return render(request, 'project1/index.html', context)


def clear(request):
    """Remove the uploaded dataset from the session and return to a blank page."""
    request.session.pop('csv_data', None)
    return redirect('project1:index')
