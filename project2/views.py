"""
Django views for Project 2 (Explainability).

The views are deliberately thin: all the heavy lifting lives in ``ml.py``. Each
endpoint returns JSON so the front-end can update the Plotly charts live without
a page reload. The single HTML page is served by :func:`index`.
"""

import json

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from . import ml


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def index(request):
    """Render the single-page explainability dashboard."""
    data = ml.ml_safe_get_data()
    context = {
        "classes": data["classes"],
        "numeric_features": [
            {"value": f, "label": ml.FEATURE_LABELS[f]} for f in ml.NUMERIC_FEATURES
        ],
        "n_samples": len(data["df"]),
    }
    return render(request, "project2/index.html", context)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _model_type(request, default="tree"):
    mt = request.GET.get("model_type", default)
    return mt if mt in ("tree", "logistic") else default


def _lambda(request, model_type):
    """Parse lambda from the request, clamped to the model's valid range."""
    lo, hi = ml.lambda_range(model_type)
    try:
        lam = float(request.GET.get("lambda", 0.0))
    except (TypeError, ValueError):
        lam = 0.0
    return max(lo, min(hi, lam))


# --------------------------------------------------------------------------- #
# API: model metadata (ranges, slider config, dataset rows)
# --------------------------------------------------------------------------- #


@require_GET
def api_meta(request):
    """Return slider ranges + the full model family for both model types.

    Also returns the list of dataset instances so the counterfactual selector
    can be populated client-side.
    """
    payload = {}
    for mt in ("tree", "logistic"):
        family = ml.get_model_family(mt)
        lo, hi = ml.lambda_range(mt)
        payload[mt] = {
            "lambda_min": lo,
            "lambda_max": hi,
            "omega_label": family[0]["omega_label"] if family else "",
            "family": [
                {
                    "omega": m["omega"],
                    "acc_test": round(m["acc_test"], 4),
                    "acc_train": round(m["acc_train"], 4),
                }
                for m in family
            ],
        }

    data = ml.ml_safe_get_data()
    instances = []
    for i in range(len(data["df"])):
        row = data["df"].iloc[i]
        instances.append(
            {
                "index": i,
                "label": f"#{i} — {row[ml.TARGET]} "
                f"({row['island']}, bill {row['bill_length_mm']:.0f}mm)",
                "species": str(row[ml.TARGET]),
            }
        )
    payload["instances"] = instances
    payload["classes"] = data["classes"]
    return JsonResponse(payload)


# --------------------------------------------------------------------------- #
# API: model summary for the current (model_type, lambda)
# --------------------------------------------------------------------------- #


@require_GET
def api_model(request):
    """Return the selected model's summary + a tree structure if applicable."""
    model_type = _model_type(request)
    lam = _lambda(request, model_type)
    best, scored = ml.select_by_lambda(model_type, lam)

    data = ml.ml_safe_get_data()
    response = {
        "model_type": model_type,
        "lambda": lam,
        "omega": best["omega"],
        "omega_label": best["omega_label"],
        "acc_test": round(best["acc_test"], 4),
        "acc_train": round(best["acc_train"], 4),
        "hyperparam": best["hyperparam"],
        "selection_curve": [
            {
                "omega": m["omega"],
                "acc_test": round(m["acc_test"], 4),
                "score": round(m["selection_score"], 4),
                "selected": m["omega"] == best["omega"],
            }
            for m in scored
        ],
    }

    if model_type == "tree":
        response["tree"] = ml.tree_to_dict(
            best["model"], data["feature_names"], best["model"].classes_
        )
        response["tree_text"] = ml.tree_text(best["model"], data["feature_names"])
    else:
        # expose the (non-zero) coefficients for transparency
        model = best["model"]
        coefs = {}
        for ci, cls in enumerate(model.classes_):
            coefs[str(cls)] = {
                fn: round(float(c), 4)
                for fn, c in zip(data["feature_names"], model.coef_[ci])
            }
        response["coefficients"] = coefs

    return JsonResponse(response)


# --------------------------------------------------------------------------- #
# API: counterfactuals
# --------------------------------------------------------------------------- #


@require_POST
def api_counterfactuals(request):
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return HttpResponseBadRequest("Invalid JSON body")

    model_type = body.get("model_type", "tree")
    if model_type not in ("tree", "logistic"):
        return HttpResponseBadRequest("Bad model_type")

    lo, hi = ml.lambda_range(model_type)
    try:
        lam = max(lo, min(hi, float(body.get("lambda", 0.0))))
        index = int(body["instance_index"])
        k = int(body.get("k", 3))
    except (KeyError, TypeError, ValueError):
        return HttpResponseBadRequest("Missing or invalid parameters")

    target = body.get("target_label")
    data = ml.ml_safe_get_data()
    if target not in data["classes"]:
        return HttpResponseBadRequest("Bad target_label")

    best, _ = ml.select_by_lambda(model_type, lam)
    result = ml.generate_counterfactuals(
        best, instance_index=index, target_label=target, k=k
    )
    result["model_type"] = model_type
    result["lambda"] = lam
    return JsonResponse(result)


# --------------------------------------------------------------------------- #
# API: PDP and ALE
# --------------------------------------------------------------------------- #


@require_GET
def api_feature_effect(request):
    """Return both PDP and ALE for one numeric feature under the current model."""
    model_type = _model_type(request)
    lam = _lambda(request, model_type)
    feature = request.GET.get("feature", ml.NUMERIC_FEATURES[0])
    if feature not in ml.NUMERIC_FEATURES:
        return HttpResponseBadRequest("Bad feature")

    best, _ = ml.select_by_lambda(model_type, lam)
    pdp = ml.compute_pdp(best, feature)
    ale = ml.compute_ale(best, feature)

    return JsonResponse(
        {
            "model_type": model_type,
            "lambda": lam,
            "feature": feature,
            "feature_label": ml.FEATURE_LABELS.get(feature, feature),
            "pdp": pdp,
            "ale": ale,
        }
    )
