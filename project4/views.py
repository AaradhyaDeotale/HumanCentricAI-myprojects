"""
Views implementing the Project 4 interface.

Two audiences, per the assignment brief:
  - `index` is the landing page: download the PDF (Tasks 1-3) or start the
    study (Task 4).
  - `consent` / `task` / `questionnaire` / `done` are the participant-facing
    user study itself -- the interface a real participant would use.

The participant's progress through the study (which phase, which trial) is
kept in the Django session as a plain dict (`p4_plan`), so a page refresh or
a closed tab resumes exactly where they left off. Each individual response
is written to the database as soon as it's given, so partial runs are not
lost either.
"""

import json
import os
import random
import string

from django.http import FileResponse, Http404
from django.shortcuts import render, redirect

from . import services
from .models import StudySession, PairwiseResponse, RankingResponse, Questionnaire
from .report import build_report

PAIRWISE_TRIALS = 15
RANKING_TRIALS = 3
RANKING_SIZE = 10
HOLDOUT_TRIALS = 8


# --------------------------------------------------------------------------- #
# Landing page / report download
# --------------------------------------------------------------------------- #


def index(request):
    return render(request, "project4/index.html", {
        "n_movies": services.n_movies(),
        "n_sessions": StudySession.objects.filter(finished=True).count(),
    })


def report(request):
    path = build_report()
    if not os.path.exists(path):
        raise Http404("report not generated")
    return FileResponse(open(path, "rb"), as_attachment=True,
                        filename="project4_report.pdf")


# --------------------------------------------------------------------------- #
# Study plan helpers
# --------------------------------------------------------------------------- #


def _ensure_session_key(request):
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def _new_plan():
    n = services.n_movies()
    order = random.choice(["pairwise_first", "ranking_first"])

    pairwise_phase = {
        "type": "pairwise", "label": "pairwise",
        "trials": [random.sample(range(n), 2) for _ in range(PAIRWISE_TRIALS)],
    }
    ranking_phase = {
        "type": "ranking", "label": "ranking",
        "trials": [random.sample(range(n), RANKING_SIZE) for _ in range(RANKING_TRIALS)],
    }
    holdout_phase = {
        "type": "pairwise", "label": "holdout",
        "trials": [random.sample(range(n), 2) for _ in range(HOLDOUT_TRIALS)],
    }

    phases = ([pairwise_phase, ranking_phase] if order == "pairwise_first"
             else [ranking_phase, pairwise_phase])
    phases.append(holdout_phase)

    return {"order": order, "phases": phases, "phase_idx": 0,
            "trial_idx": 0, "stage": "instructions"}


def _advance(plan):
    plan["phase_idx"] += 1
    plan["trial_idx"] = 0
    if plan["phase_idx"] >= len(plan["phases"]):
        plan["stage"] = "questionnaire"
    else:
        plan["stage"] = "instructions"


# --------------------------------------------------------------------------- #
# Consent -> task loop -> questionnaire -> done
# --------------------------------------------------------------------------- #


def consent(request):
    if request.method == "POST":
        session_key = _ensure_session_key(request)
        plan = _new_plan()
        study_session = StudySession.objects.create(
            session_key=session_key, order=plan["order"])
        plan["session_id"] = study_session.id
        request.session["p4_plan"] = plan
        return redirect("project4:task")
    return render(request, "project4/consent.html")


def restart(request):
    request.session.pop("p4_plan", None)
    return redirect("project4:consent")


def task(request):
    plan = request.session.get("p4_plan")
    if not plan:
        return redirect("project4:consent")

    stage = plan["stage"]
    if stage == "instructions":
        return _instructions(request, plan)
    if stage == "pairwise":
        return _pairwise_trial(request, plan)
    if stage == "ranking":
        return _ranking_trial(request, plan)
    if stage == "questionnaire":
        return redirect("project4:questionnaire")
    if stage == "done":
        return redirect("project4:done")
    return redirect("project4:consent")


def _instructions(request, plan):
    phase = plan["phases"][plan["phase_idx"]]
    if request.method == "POST":
        plan["stage"] = phase["type"]
        request.session["p4_plan"] = plan
        return redirect("project4:task")

    return render(request, "project4/instructions.html", {
        "phase_label": phase["label"],
        "phase_number": plan["phase_idx"] + 1,
        "total_phases": len(plan["phases"]),
        "n_trials": len(phase["trials"]),
    })


def _pairwise_trial(request, plan):
    phase = plan["phases"][plan["phase_idx"]]
    idx = plan["trial_idx"]
    a, b = phase["trials"][idx]

    if request.method == "POST":
        try:
            chosen = int(request.POST.get("chosen", ""))
        except ValueError:
            chosen = None
        if chosen not in (a, b):
            df, _X, _names = services.dataset()
            return render(request, "project4/pairwise.html", {
                "movie_a": services.movie_brief(a, df),
                "movie_b": services.movie_brief(b, df),
                "trial_number": idx + 1, "total_trials": len(phase["trials"]),
                "is_holdout": phase["label"] == "holdout",
                "error": "Please choose one of the two movies.",
            })

        rt = request.POST.get("rt_ms")
        PairwiseResponse.objects.create(
            session_id=plan["session_id"], phase=phase["label"], trial_index=idx,
            movie_a=a, movie_b=b, chosen=chosen,
            response_ms=int(rt) if rt and rt.isdigit() else None,
        )
        plan["trial_idx"] += 1
        if plan["trial_idx"] >= len(phase["trials"]):
            _advance(plan)
        request.session["p4_plan"] = plan
        return redirect("project4:task")

    df, _X, _names = services.dataset()
    return render(request, "project4/pairwise.html", {
        "movie_a": services.movie_brief(a, df),
        "movie_b": services.movie_brief(b, df),
        "trial_number": idx + 1, "total_trials": len(phase["trials"]),
        "is_holdout": phase["label"] == "holdout",
    })


def _ranking_trial(request, plan):
    phase = plan["phases"][plan["phase_idx"]]
    idx = plan["trial_idx"]
    movie_ids = phase["trials"][idx]

    if request.method == "POST":
        order_str = request.POST.get("order", "")
        try:
            ids = [int(x) for x in order_str.split(",") if x.strip() != ""]
        except ValueError:
            ids = []

        if sorted(ids) != sorted(movie_ids):
            df, _X, _names = services.dataset()
            return render(request, "project4/ranking.html", {
                "movies": [services.movie_brief(m, df) for m in movie_ids],
                "trial_number": idx + 1, "total_trials": len(phase["trials"]),
                "error": "Please rank all the movies shown before submitting.",
            })

        rt = request.POST.get("rt_ms")
        RankingResponse.objects.create(
            session_id=plan["session_id"], trial_index=idx,
            movie_ids_json=json.dumps(ids),
            response_ms=int(rt) if rt and rt.isdigit() else None,
        )
        plan["trial_idx"] += 1
        if plan["trial_idx"] >= len(phase["trials"]):
            _advance(plan)
        request.session["p4_plan"] = plan
        return redirect("project4:task")

    df, _X, _names = services.dataset()
    return render(request, "project4/ranking.html", {
        "movies": [services.movie_brief(m, df) for m in movie_ids],
        "trial_number": idx + 1, "total_trials": len(phase["trials"]),
    })


def questionnaire(request):
    plan = request.session.get("p4_plan")
    if not plan or plan.get("stage") not in ("questionnaire", "done"):
        return redirect("project4:consent")
    if plan["stage"] == "done":
        return redirect("project4:done")

    if request.method == "POST":
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        Questionnaire.objects.create(
            session_id=plan["session_id"],
            pairwise_difficulty=int(request.POST["pairwise_difficulty"]),
            ranking_difficulty=int(request.POST["ranking_difficulty"]),
            pairwise_confidence=int(request.POST["pairwise_confidence"]),
            ranking_confidence=int(request.POST["ranking_confidence"]),
            preferred_method=request.POST.get("preferred_method", "no_preference"),
            comments=request.POST.get("comments", "")[:2000],
        )
        StudySession.objects.filter(id=plan["session_id"]).update(
            finished=True, completion_code=code)
        plan["stage"] = "done"
        request.session["p4_plan"] = plan
        return redirect("project4:done")

    return render(request, "project4/questionnaire.html")


def done(request):
    plan = request.session.get("p4_plan")
    if not plan or plan.get("stage") != "done":
        return redirect("project4:consent")

    study_session = StudySession.objects.get(id=plan["session_id"])
    summary = services.session_summary(study_session.id)
    return render(request, "project4/done.html", {
        "completion_code": study_session.completion_code,
        **summary,
    })
