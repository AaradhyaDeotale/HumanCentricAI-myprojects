import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def index(request):
    context = {}

    if request.method == 'POST':
        # Use uploaded file if provided, otherwise fall back to session
        if request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']
            df = pd.read_csv(csv_file)
            request.session['csv_data'] = df.to_json()
        elif request.session.get('csv_data'):
            df = pd.read_json(request.session['csv_data'])
        else:
            # No file and no session - go back to empty page
            return render(request, 'project1/index.html', context)
    else:
        # GET request - check if we have session data to restore
        if request.session.get('csv_data'):
            df = pd.read_json(request.session['csv_data'])
        else:
            return render(request, 'project1/index.html', context)

    # Dataset info
    context['columns'] = list(df.columns)
    context['table'] = df.head(10).to_html(classes='data-table', index=False)
    context['shape'] = df.shape

    # Scatter plot
    x_col = request.POST.get('x_col', df.columns[0])
    y_col = request.POST.get('y_col', df.columns[1])
    label_col = df.columns[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        ax.scatter(subset[x_col], subset[y_col], label=str(label))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.set_title(f'{x_col} vs {y_col}')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    context['plot'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    context['x_col'] = x_col
    context['y_col'] = y_col

# ML Training
    model_name = request.POST.get('model')
    test_size = int(request.POST.get('test_size', 20)) / 100
    hyperparameter = int(request.POST.get('hyperparameter', 5))
    context['test_size'] = int(test_size * 100)
    context['model_name'] = model_name
    context['hyperparameter'] = hyperparameter

    if model_name and model_name in ['knn', 'decision_tree', 'svm']:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=hyperparameter)
            hyperparam_label = 'K (neighbors)'
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=hyperparameter)
            hyperparam_label = 'Max Depth'
        else:
            model = SVC(C=hyperparameter)
            hyperparam_label = 'C (regularization)'

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)

        context['accuracy'] = round(accuracy * 100, 2)
        context['report'] = report_df.to_html(classes='data-table')
        context['hyperparam_label'] = hyperparam_label

        # Plot accuracy vs hyperparameter
        param_range = range(1, 21)
        accuracies = []
        for p in param_range:
            if model_name == 'knn':
                m = KNeighborsClassifier(n_neighbors=p)
            elif model_name == 'decision_tree':
                m = DecisionTreeClassifier(max_depth=p)
            else:
                m = SVC(C=p)
            m.fit(X_train, y_train)
            accuracies.append(accuracy_score(y_test, m.predict(X_test)))

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(list(param_range), accuracies, marker='o')
        ax2.axvline(x=hyperparameter, color='red', linestyle='--', label='Selected')
        ax2.set_xlabel(hyperparam_label)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Hyperparameter')
        ax2.legend()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        context['acc_plot'] = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close()

    return render(request, 'project1/index.html', context)