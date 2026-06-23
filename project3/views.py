import json
import os

from django.conf import settings
from django.http import JsonResponse, FileResponse, Http404
from django.shortcuts import render

from . import services
from .report import build_report


def index(request):
    """Landing page for Project 3."""
    return render(request, "project3/index.html",
                  {"class_names": services.CLASS_NAMES})


def train(request):
    """Task 1: train baseline classifier, return its test accuracy."""
    kind = request.GET.get("kind", "tfidf")
    cap = request.GET.get("cap", "20000")
    cap = None if cap in ("", "all", "0") else int(cap)
    state = services.get_state(clf_kind=kind, train_cap=cap)
    return JsonResponse({
        "kind": state["clf_kind"],
        "test_accuracy": round(state["test_accuracy"], 4),
        "n_train": state["n_train"],
        "experts": services.expert_report(state),
    })


def defer(request):
    """Task 3: run deferral at a chosen tau + return the tau-sweep plot."""
    state = services.get_state()
    tau = float(request.GET.get("tau", "0.1"))
    expert_index = int(request.GET.get("expert", "0"))
    res = services.run_deferral(state, tau, expert_index)
    plot_url = services.deferral_curve_plot(state, expert_index)
    res["plot_url"] = plot_url
    res = {k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in res.items()}
    return JsonResponse(res)


def active(request):
    """Task 4: run an active-learning strategy + comparison plot."""
    state = services.get_state()
    strategy = request.GET.get("strategy", "competence_gap")
    budget = int(request.GET.get("budget", "400"))
    expert_index = int(request.GET.get("expert", "0"))
    single = services.run_active(state, strategy, budget=budget,
                                 expert_index=expert_index)
    plot_url, summary = services.active_compare_plot(
        state, expert_index, budget=budget)
    single.pop("history", None)
    return JsonResponse({"run": single, "plot_url": plot_url,
                         "summary": summary})


def classify(request):
    """Single-input demo: classify text and show the defer decision."""
    state = services.get_state()
    text = request.GET.get("text", "")
    if not text:
        return JsonResponse({"error": "empty text"}, status=400)
    return JsonResponse(services.classify_text(state, text))


def report(request):
    """Generate the PDF report on demand and serve it for download."""
    state = services.get_state()
    path = build_report(state, services)
    if not os.path.exists(path):
        raise Http404("report not generated")
    return FileResponse(open(path, "rb"), as_attachment=True,
                        filename="project3_report.pdf")
