import json
import os

from django.conf import settings
from django.http import JsonResponse, FileResponse, Http404
from django.shortcuts import render

from . import services
from .models import TrainedModel, DeferralResult, ActiveLearningRun
from .report import build_report


# How many rows of each kind to show in the history panel.
HISTORY_LIMIT = 10


# --------------------------------------------------------------------------- #
# Run history (persisted with the Django ORM)
# --------------------------------------------------------------------------- #


def _latest_trained_model(state=None):
    """The most recent TrainedModel row, creating one from `state` if none exist.

    Task 3/4 need a model to attach their results to. Normally the user trains
    in Task 1 first (which writes a row); this is the fallback for when they
    jump straight to a later task.
    """
    tm = TrainedModel.objects.order_by("-created").first()
    if tm is None and state is not None:
        tm = TrainedModel.objects.create(
            kind=state["clf_kind"],
            test_accuracy=state["test_accuracy"],
            n_train=state["n_train"],
        )
    return tm


def _history_payload():
    """Recent trainings / deferrals / active-learning runs + 'best' markers."""
    trainings = list(TrainedModel.objects.order_by("-created")[:HISTORY_LIMIT])
    deferrals = list(DeferralResult.objects.order_by("-created")[:HISTORY_LIMIT])
    active_runs = list(ActiveLearningRun.objects.order_by("-created")[:HISTORY_LIMIT])

    best_train = max(trainings, key=lambda t: t.test_accuracy, default=None)
    best_defer = max(deferrals, key=lambda d: d.system_accuracy, default=None)
    best_active = min(active_runs, key=lambda a: a.final_l1_error, default=None)

    return {
        "trainings": [
            {
                "id": t.id,
                "when": t.created.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": t.kind,
                "test_accuracy": round(t.test_accuracy, 4),
                "n_train": t.n_train,
                "best": bool(best_train and t.id == best_train.id),
            }
            for t in trainings
        ],
        "deferrals": [
            {
                "id": d.id,
                "when": d.created.strftime("%Y-%m-%d %H:%M:%S"),
                "tau": round(d.tau, 3),
                "system_accuracy": round(d.system_accuracy, 4),
                "classifier_accuracy": round(d.classifier_accuracy, 4),
                "expert_accuracy": round(d.expert_accuracy, 4),
                "coverage": round(d.coverage, 4),
                "best": bool(best_defer and d.id == best_defer.id),
            }
            for d in deferrals
        ],
        "active_runs": [
            {
                "id": a.id,
                "when": a.created.strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": a.strategy,
                "budget": a.budget,
                "final_l1_error": round(a.final_l1_error, 4),
                "best": bool(best_active and a.id == best_active.id),
            }
            for a in active_runs
        ],
        "counts": {
            "trainings": TrainedModel.objects.count(),
            "deferrals": DeferralResult.objects.count(),
            "active_runs": ActiveLearningRun.objects.count(),
        },
    }


# --------------------------------------------------------------------------- #
# Pages / API
# --------------------------------------------------------------------------- #


def index(request):
    """Landing page for Project 3. The history panel is filled client-side
    from the `history/` endpoint (and refreshed after every task)."""
    return render(request, "project3/index.html",
                  {"class_names": services.CLASS_NAMES})


def history(request):
    """Return the run history as JSON (used to refresh the panel live)."""
    return JsonResponse(_history_payload())


def train(request):
    """Task 1: train baseline classifier, return its test accuracy."""
    kind = request.GET.get("kind", "tfidf")
    cap = request.GET.get("cap", "20000")
    cap = None if cap in ("", "all", "0") else int(cap)
    state = services.get_state(clf_kind=kind, train_cap=cap)

    # One recorded training per explicit "Train" click (mirrors Project 1).
    TrainedModel.objects.create(
        kind=state["clf_kind"],
        test_accuracy=state["test_accuracy"],
        n_train=state["n_train"],
    )

    return JsonResponse({
        "kind": state["clf_kind"],
        "test_accuracy": round(state["test_accuracy"], 4),
        "n_train": state["n_train"],
        "experts": services.expert_report(state),
        "history": _history_payload(),
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

    DeferralResult.objects.create(
        model=_latest_trained_model(state),
        tau=tau,
        system_accuracy=res["system_accuracy"],
        classifier_accuracy=res["classifier_accuracy"],
        expert_accuracy=res["expert_accuracy"],
        coverage=res["coverage"],
    )
    res["history"] = _history_payload()
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
    run_history = single.pop("history", None)

    ActiveLearningRun.objects.create(
        model=_latest_trained_model(state),
        strategy=strategy,
        budget=budget,
        final_l1_error=single["final_l1_error"],
        history_json=json.dumps(run_history) if run_history else "",
    )

    return JsonResponse({"run": single, "plot_url": plot_url,
                         "summary": summary, "history": _history_payload()})


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
