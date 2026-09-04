"""
PDF report generator (required deliverable for Project 4).

Describes the feature representation and Plackett-Luce preference model
(Tasks 1-2) and the full user-study protocol (Task 3). Served for download
from the project landing page.
"""

import os
from django.conf import settings

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, ListFlowable, ListItem)

from . import services


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Body2", parent=ss["BodyText"], spaceAfter=8, leading=14))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8.5, leading=11,
                          textColor=colors.HexColor("#555555")))
    ss.add(ParagraphStyle("CenterTitle", parent=ss["Title"], alignment=TA_CENTER))
    return ss


def build_report():
    out_path = os.path.join(settings.MEDIA_ROOT, "project4_report.pdf")
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    ss = _styles()
    story = []

    def H(t, lvl="Heading2"):
        story.append(Paragraph(t, ss[lvl]))

    def P(t):
        story.append(Paragraph(t, ss["Body2"]))

    def small(t):
        story.append(Paragraph(t, ss["Small"]))

    def bullets(items):
        story.append(ListFlowable(
            [ListItem(Paragraph(i, ss["Body2"])) for i in items],
            bulletType="bullet", leftIndent=14))

    df, X, feature_names = services.dataset()

    # ------------------------------------------------------------------ #
    H("Project 4: Preference Elicitation for Movie Recommendation", "CenterTitle")
    P("Human-Centric Artificial Intelligence &mdash; design report. "
      "We design and implement a user study comparing two interfaces for "
      "eliciting a new user's movie preferences: repeated pairwise choices "
      "(Design&nbsp;1) and ranking lists of ten movies (Design&nbsp;2). Both "
      "feed a linear utility model fit with a Plackett-Luce likelihood over "
      f"the {len(df)} movies of the IMDB&nbsp;5000 Movie Dataset. The "
      "interactive interface implementing Task&nbsp;4 is linked from the "
      "project landing page.")

    # ================================================================== #
    H("Task 1 &mdash; Feature representation")
    P("The dataset contains only movie metadata, no ratings, so the "
      "feature vector x has to be built entirely from a movie's own "
      "attributes. We use five groups of features, chosen to cover the "
      "dimensions along which people typically describe liking or "
      "disliking a film:")
    bullets([
        "<b>Genre</b> (multi-hot over 26 genres) &mdash; the most direct "
        "signal of content type; a movie can carry several genre tags.",
        "<b>Content rating</b> (one-hot, bucketed to G / PG / PG-13 / R / "
        "NC-17 / Other) &mdash; tone and intended audience.",
        "<b>Production scale</b> &mdash; log-budget, log-gross, log-cast "
        "and log-director Facebook likes (star/production power).",
        "<b>Era and length</b> &mdash; release year and running time, "
        "z-scored.",
        "<b>Reach and reception</b> &mdash; log-vote-count (how mainstream "
        "the film is), IMDb score (critical consensus), and binary "
        "is-English / is-USA flags.",
    ])
    P(f"Concatenating these blocks gives a <b>{X.shape[1]}-dimensional</b> "
      "feature vector per movie. Skewed counts (votes, budget, gross, "
      "Facebook likes) are log<sub>1p</sub>-transformed before "
      "z-scoring so that a handful of blockbusters don't dominate the "
      "scale; missing numeric values are median-imputed and missing "
      "categorical values fall into an &ldquo;Other&rdquo;/&ldquo;Not "
      "Rated&rdquo; bucket rather than being dropped, so the full catalogue "
      "stays available for sampling. Exact duplicate (title, year) rows "
      "(&asymp;5% of the raw file, mostly re-scraped re-releases) are "
      "removed so a participant is never asked to compare a movie against "
      "itself.")
    P("<b>Deliberately excluded:</b> an intercept/bias feature. The "
      "Bradley-Terry and Plackett-Luce likelihoods (Task&nbsp;2) depend "
      "only on <i>differences</i> of utilities within a comparison set; a "
      "feature equal to 1 for every movie contributes an identical additive "
      "constant to every item's utility and cancels out of every "
      "comparison, so its weight cannot be estimated from choice data and "
      "including it would only waste one dimension.")
    P("Extraction is implemented in <font face='Courier'>project4/ml/data.py</font> "
      "(<font face='Courier'>load_movies</font> for cleaning, "
      "<font face='Courier'>build_features</font> for the vectors above) and "
      "cached once per process in <font face='Courier'>project4/services.py</font>.")

    # ================================================================== #
    H("Task 2 &mdash; From Bradley-Terry to rankings (Plackett-Luce)")
    P("The standard Bradley-Terry model gives the probability that movie "
      "i beats movie j from their utilities U(x)&nbsp;=&nbsp;w<super>T</super>x:")
    P("<i>P(i &#8227; j) = exp(U<sub>i</sub>) / (exp(U<sub>i</sub>) + exp(U<sub>j</sub>))</i>",)
    P("Design&nbsp;2 asks for a full ranking i<sub>1</sub>&nbsp;&#8227;&nbsp;"
      "i<sub>2</sub>&nbsp;&#8227;&nbsp;&hellip;&nbsp;&#8227;&nbsp;i<sub>n</sub> "
      "of n&nbsp;=&nbsp;10 movies rather than a single winner. We extend "
      "Bradley-Terry by treating the ranking as a sequence of "
      "&ldquo;pick the best of what remains&rdquo; choices: the top movie "
      "is chosen from all n candidates by a Bradley-Terry-style contest "
      "among all of them; the runner-up is then chosen the same way among "
      "the n&minus;1 that remain; and so on down to the last pair. "
      "Multiplying the probabilities of these n&minus;1 sequential choices "
      "gives the <b>Plackett-Luce model</b>:")
    P("<i>P(i<sub>1</sub> &#8227; &hellip; &#8227; i<sub>n</sub>) = "
      "&prod;<sub>k=1</sub><sup>n&minus;1</sup> "
      "exp(U<sub>i<sub>k</sub></sub>) / "
      "&sum;<sub>l=k</sub><sup>n</sup> exp(U<sub>i<sub>l</sub></sub>)</i>")
    P("Setting n&nbsp;=&nbsp;2 collapses this to exactly the Bradley-Terry "
      "formula above, so a pairwise comparison is simply a ranking of "
      "length&nbsp;2. This means Design-1 and Design-2 responses can be "
      "combined into a single likelihood and fit jointly, which is what "
      "the live interface does when it computes a participant's final "
      "recommendations from all of their answers.")
    P("<b>Fitting.</b> We find w by maximizing this log-likelihood plus an "
      "L2 penalty &lambda;&#8214;w&#8214;<super>2</super> (equivalently, a "
      "zero-mean Gaussian prior on w), using L-BFGS with an analytic "
      "gradient (<font face='Courier'>project4/ml/preference.py</font>). "
      "The penalty matters in practice: with only a few dozen comparisons "
      "and 34 features, unregularized maximum likelihood can diverge "
      "whenever the observed choices happen to be perfectly separable by "
      "some direction in feature space.")
    P("<b>Assumption and limitation.</b> Plackett-Luce inherits Luce's "
      "independence-of-irrelevant-alternatives property: the relative odds "
      "of i beating j never depend on which other items are present in the "
      "comparison set. This is a strong assumption &mdash; e.g. a "
      "participant's relative preference between two thrillers might shift "
      "once a comedy is added to the set as a &ldquo;none of these "
      "either&rdquo; option &mdash; but it is what makes pairwise and "
      "10-way comparisons commensurable under one model, which the "
      "study needs in order to compare the two designs on equal footing.")

    # ================================================================== #
    H("Task 3 &mdash; User study design")

    H("Research question and hypotheses", "Heading3")
    P("<b>RQ:</b> For eliciting a new user's movie preferences under a "
      "fixed number of movies viewed, is a ranking-based interface or a "
      "pairwise-comparison interface more effective, and how do users "
      "experience each?")
    bullets([
        "<b>H1 (accuracy):</b> a preference vector fit from ranking "
        "responses predicts a participant's held-out choices at least as "
        "well as one fit from the same number of movie exposures worth of "
        "pairwise responses. Rationale: one ranking of 10 items logically "
        "implies up to C(10,2)&nbsp;=&nbsp;45 pairwise orderings, so "
        "rankings should be more information-dense per movie shown.",
        "<b>H2 (subjective cost):</b> participants report ranking as more "
        "cognitively demanding than pairwise choice, even if it is more "
        "informative &mdash; i.e. we expect an accuracy/effort trade-off "
        "rather than one design dominating the other outright.",
    ])

    H("Design", "Heading3")
    P("<b>Within-subjects, counterbalanced.</b> Every participant "
      "completes both interfaces, in a randomly assigned order (pairwise "
      "&#8594; ranking, or ranking &#8594; pairwise), so each person acts "
      "as their own control for individual differences in taste and in "
      "general willingness to engage with the task. Counterbalancing order "
      "controls for practice and fatigue effects. The main threat specific "
      "to a within-subjects design is carry-over (the first task shapes how "
      "the participant approaches the second); we mitigate this by drawing "
      "an entirely fresh, non-overlapping set of movies for every phase, so "
      "there is nothing for the participant to consciously stay consistent "
      "with across phases.")
    P("<b>Exposure-matched budget.</b> The pairwise phase is 15 trials "
      "(30 movie views) and the ranking phase is 3 rankings of 10 movies "
      "(also 30 movie views), so both interfaces show the participant the "
      "same number of movies. This keeps the comparison about "
      "&ldquo;information per movie shown&rdquo; rather than "
      "&ldquo;information per minute&rdquo; or an arbitrarily chosen trial "
      "count for one condition.")
    P("<b>Held-out evaluation block.</b> After both elicitation phases, "
      "every participant completes 8 additional pairwise comparisons on "
      "freshly drawn movies, framed neutrally as &ldquo;a few final "
      "comparisons&rdquo; so they are answered in the same spirit as the "
      "rest of the study rather than as a test. These trials are never "
      "used to fit a preference vector &mdash; only to score one. All "
      "movies are drawn uniformly at random from the catalogue, as "
      "specified in the assignment brief; an adaptive/informative "
      "selection strategy is noted as an interesting extension but is out "
      "of scope here.")

    H("Independent and dependent variables", "Heading3")
    tbl = [
        ["Variable", "Type", "Levels / operationalisation"],
        ["Elicitation interface", "Within-subject IV", "Pairwise (Design 1) vs. Ranking (Design 2)"],
        ["Presentation order", "Between-subject IV", "Pairwise-first vs. Ranking-first (counterbalanced)"],
        ["Held-out predictive accuracy", "DV (primary)", "% of the 8 held-out pairwise choices correctly predicted by w fit on that phase's data alone"],
        ["Held-out log-likelihood", "DV (primary)", "Plackett-Luce log-likelihood of held-out choices under each phase's w"],
        ["Perceived difficulty", "DV (subjective)", "5-point Likert, one item per interface"],
        ["Perceived confidence", "DV (subjective)", "5-point Likert: 'this reflected my real taste', per interface"],
        ["Overall preference", "DV (subjective)", "Forced choice: pairwise / ranking / no preference"],
        ["Completion time per phase", "DV (behavioral)", "Client-side timestamp, ms, recorded per trial"],
    ]
    t = Table(tbl, colWidths=[4.2*cm, 3.0*cm, 8.3*cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#275cb2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    H("Materials", "Heading3")
    P("Each movie is shown as a card with title, year, up to 3 genres, up "
      "to 4 plot keywords, director, and top-3 billed cast &mdash; enough "
      "content information to form a genuine preference. <b>IMDb score and "
      "vote count are deliberately withheld</b> during both elicitation "
      "phases and only revealed on the final results screen: showing a "
      "visible &ldquo;quality&rdquo; number would anchor choices toward "
      "the more acclaimed/popular movie rather than the one the "
      "participant would personally rather watch, contaminating exactly "
      "the signal the study is trying to measure.")

    H("Procedure", "Heading3")
    bullets([
        "Landing page &#8594; consent screen (purpose, duration &asymp;8-10 "
        "min, voluntary participation, no personal data collected beyond "
        "an anonymous session id).",
        "Randomly assigned phase order; on-screen instructions before each "
        "phase.",
        "Phase A (15 pairwise trials, or 3 rankings of 10) &#8594; "
        "interstitial instructions &#8594; Phase B (the other interface).",
        "8-trial held-out comparison block.",
        "Post-study questionnaire (difficulty, confidence, overall "
        "preference, free-text comments).",
        "Debrief screen: personalised top-5 recommendations computed from "
        "the participant's own responses (a concrete, motivating payoff "
        "for genuine engagement) and a completion code for crowdsourcing "
        "platforms.",
    ])

    H("Recruitment and sample size", "Heading3")
    P("Recruit via a platform such as Prolific, screening for fluent "
      "English and self-reported regular movie-watching (at least one "
      "movie per month), to avoid participants with too little basis for a "
      "preference. A rough power calculation for a paired comparison "
      "(e.g. Wilcoxon signed-rank on perceived difficulty, or a paired "
      "t-test on held-out accuracy) at a medium effect size "
      "(d&nbsp;&asymp;&nbsp;0.5), &alpha;&nbsp;=&nbsp;.05, power&nbsp;=&nbsp;.8 "
      "calls for roughly 34 participants; we would recruit "
      "<b>N&nbsp;=&nbsp;45</b> to allow for exclusions. Compensation "
      "targeting a fair hourly rate (e.g. &pound;1.50 for a ~9-minute "
      "study).")

    H("Quality control", "Heading3")
    P("One of the 15 pairwise trials is silently repeated later in the "
      "same phase with the identical pair; a participant who answers it "
      "inconsistently, combined with an implausibly fast median "
      "response time (&lt;1.5s), is a candidate for exclusion before "
      "analysis. Session state and every individual response are "
      "persisted to the database as they happen, so partial/abandoned "
      "sessions remain available rather than being silently lost.")

    H("Ethics", "Heading3")
    P("No personal data is collected: participants are identified only by "
      "an anonymous, randomly generated session key. Participation is "
      "voluntary and participants may stop at any time with no way for us "
      "to trace an abandoned session back to them. A real deployment of "
      "this study would be submitted for institutional ethics review "
      "before recruiting; the consent screen implemented here reflects "
      "what that review would typically require (purpose, duration, "
      "voluntariness, data handling) but does not substitute for it.")

    H("Analysis plan", "Heading3")
    P("For each participant, fit two <i>independent</i> preference "
      "vectors &mdash; w<sub>pairwise</sub> from only their pairwise-phase "
      "responses, w<sub>ranking</sub> from only their ranking-phase "
      "responses &mdash; and score both against that participant's own "
      "held-out block (never used for fitting). Compare the two "
      "interfaces with a paired test (Wilcoxon signed-rank, participant as "
      "the unit of analysis) on: (1) held-out predictive accuracy, (2) "
      "held-out log-likelihood, (3) Likert difficulty/confidence ratings, "
      "and (4) time on task. Presentation order enters as a between-"
      "subjects check for carry-over (an order&nbsp;&times;&nbsp;interface "
      "interaction test); if no interaction is found, order is dropped and "
      "the two orders are pooled.")

    small("Interactive implementation of Task 4 (this protocol's actual "
         "participant-facing interface) is available from the Project 4 "
         "landing page.")

    doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    doc.build(story)
    return out_path
