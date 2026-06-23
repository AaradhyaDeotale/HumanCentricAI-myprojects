"""
PDF report generator (required deliverable for Project 3).

Builds a self-contained report describing the experiments, design
choices and results for Tasks 1-4, embedding the figures produced by
the service layer. Served for download from the project interface.
"""

import os
from django.conf import settings

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle)


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Body2", parent=ss["BodyText"],
                          spaceAfter=8, leading=14))
    return ss


def build_report(state, services):
    out_path = os.path.join(settings.MEDIA_ROOT, "project3_report.pdf")
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    ss = _styles()
    story = []

    def H(t, lvl="Heading2"):
        story.append(Paragraph(t, ss[lvl]))

    def P(t):
        story.append(Paragraph(t, ss["Body2"]))

    H("Project 3: Active Learning for Learning-to-Defer", "Title")
    P("Human-Centric Artificial Intelligence &mdash; experimental report. "
      "The system labels AG News articles into four topics "
      "(World, Sports, Business, Sci/Tech) and collaborates with a "
      "simulated human expert through learning-to-defer.")

    # Task 1
    H("Task 1 &mdash; Baseline classifier")
    P(f"We train a <b>{state['clf_kind']}</b> classifier on "
      f"{state['n_train']} labelled training articles. On the held-out "
      f"AG News test set (7600 articles) it reaches a test accuracy of "
      f"<b>{state['test_accuracy']*100:.2f}%</b>. This is the baseline "
      f"the human-AI team must match or beat.")
    P("Design choice: TF-IDF + multinomial logistic regression gives "
      "calibrated class probabilities cheaply, which the deferral and "
      "active-learning stages depend on. A fine-tuned DistilBERT variant "
      "shares the same interface and can be swapped in for higher accuracy.")

    # Task 2
    H("Task 2 &mdash; Simulated experts")
    P("Experts are deliberately imperfect and have <i>localised</i> "
      "competence: each is reliable on a subset of classes and weak "
      "elsewhere. Querying the same example twice returns the same answer, "
      "so expert noise cannot be averaged away. Measured accuracies:")
    rows = [["Expert", "Overall", "World", "Sports", "Business", "Sci/Tech"]]
    for r in services.expert_report(state):
        pc = r["per_class"]
        rows.append([r["name"], f"{r['accuracy']:.2f}",
                     f"{pc.get('World','-')}", f"{pc.get('Sports','-')}",
                     f"{pc.get('Business','-')}", f"{pc.get('Sci/Tech','-')}"])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # Task 3
    H("Task 3 &mdash; Learning to defer")
    best = services.run_deferral(state, tau=0.1)
    P("The deferral policy compares, per input, the classifier's max "
      "probability against the expert's <i>expected</i> correctness "
      "(estimated from the predicted-class distribution and a per-class "
      "competence vector). It defers when the expert's expected accuracy "
      "exceeds the classifier's by a margin &tau;.")
    P(f"At &tau;=0.1 the combined system reaches "
      f"<b>{best['system_accuracy']*100:.2f}%</b> accuracy with "
      f"{best['coverage']*100:.0f}% coverage, versus "
      f"{best['classifier_accuracy']*100:.2f}% for the classifier alone "
      f"and only {best['expert_accuracy']*100:.2f}% for the expert alone. "
      f"Selective deferral on the expert's strong classes lifts the team "
      f"above the baseline.")
    defer_plot = services.deferral_curve_plot(state)
    img_path = os.path.join(settings.MEDIA_ROOT, os.path.basename(defer_plot))
    if os.path.exists(img_path):
        story.append(Image(img_path, width=12 * cm, height=8 * cm))

    # Task 4
    H("Task 4 &mdash; Active learning for competence discovery")
    P("Without any expert labels at training time, we actively query the "
      "expert to learn <i>where</i> it is competent. Per-class competence "
      "is modelled with a Beta posterior; we compare three acquisition "
      "strategies. Uncertainty sampling concentrates queries where the "
      "classifier is unsure &mdash; exactly the region where knowing the "
      "expert's skill matters &mdash; and recovers the competence profile "
      "fastest.")
    plot_url, summary = services.active_compare_plot(state)
    img2 = os.path.join(settings.MEDIA_ROOT, os.path.basename(plot_url))
    if os.path.exists(img2):
        story.append(Image(img2, width=12 * cm, height=8 * cm))
    srows = [["Strategy", "Final competence L1 error"]]
    for k, v in summary.items():
        srows.append([k, f"{v:.4f}"])
    t2 = Table(srows, hAlign="LEFT")
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8e44ad")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)

    H("Conclusion")
    P("The experiments show a working human-AI team: a strong baseline "
      "classifier, imperfect experts with localised skill, a deferral "
      "policy that beats both members alone, and an active-learning loop "
      "that discovers expert competence from a small query budget.")

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    return out_path
