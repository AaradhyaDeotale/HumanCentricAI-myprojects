from django.db import models


class StudySession(models.Model):
    """One participant's run through the user study (Task 3 design)."""

    ORDER_CHOICES = [
        ("pairwise_first", "Pairwise first, then ranking"),
        ("ranking_first", "Ranking first, then pairwise"),
    ]

    session_key = models.CharField(max_length=64)
    order = models.CharField(max_length=20, choices=ORDER_CHOICES)
    started = models.DateTimeField(auto_now_add=True)
    finished = models.BooleanField(default=False)
    completion_code = models.CharField(max_length=12, blank=True)

    def __str__(self):
        return f"Session {self.id} ({self.order})"


class PairwiseResponse(models.Model):
    """A single Design-1 trial: a choice between two movies.

    `phase` distinguishes the elicitation trials ("pairwise") used to fit a
    participant's preference vector from the "holdout" trials collected
    after both elicitation phases and reserved purely for evaluating how
    well each interface's fitted model predicts unseen choices.
    """

    PHASE_CHOICES = [("pairwise", "Elicitation"), ("holdout", "Holdout")]

    session = models.ForeignKey(StudySession, on_delete=models.CASCADE,
                                related_name="pairwise_responses")
    phase = models.CharField(max_length=10, choices=PHASE_CHOICES, default="pairwise")
    trial_index = models.IntegerField()
    movie_a = models.IntegerField()
    movie_b = models.IntegerField()
    chosen = models.IntegerField()
    response_ms = models.IntegerField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)


class RankingResponse(models.Model):
    """A single Design-2 trial: a full ranking of ten movies."""

    session = models.ForeignKey(StudySession, on_delete=models.CASCADE,
                                related_name="ranking_responses")
    trial_index = models.IntegerField()
    movie_ids_json = models.TextField()  # JSON list, most to least preferred
    response_ms = models.IntegerField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)


class Questionnaire(models.Model):
    """Post-study questionnaire comparing the two interfaces (Task 3)."""

    PREFERENCE_CHOICES = [
        ("pairwise", "Pairwise comparisons"),
        ("ranking", "Ranking lists"),
        ("no_preference", "No preference"),
    ]

    session = models.OneToOneField(StudySession, on_delete=models.CASCADE,
                                   related_name="questionnaire")
    pairwise_difficulty = models.IntegerField()
    ranking_difficulty = models.IntegerField()
    pairwise_confidence = models.IntegerField()
    ranking_confidence = models.IntegerField()
    preferred_method = models.CharField(max_length=20, choices=PREFERENCE_CHOICES)
    comments = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
