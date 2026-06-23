from django.db import models


class TrainedModel(models.Model):
    """A trained baseline classifier (Task 1)."""
    kind = models.CharField(max_length=32, default="tfidf")
    test_accuracy = models.FloatField(default=0.0)
    n_train = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    artifact_path = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.kind} ({self.test_accuracy:.3f})"


class DeferralResult(models.Model):
    """A learning-to-defer evaluation (Task 3) at a given tau."""
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE,
                              related_name="deferrals")
    tau = models.FloatField(default=0.0)
    system_accuracy = models.FloatField(default=0.0)
    classifier_accuracy = models.FloatField(default=0.0)
    expert_accuracy = models.FloatField(default=0.0)
    coverage = models.FloatField(default=0.0)
    created = models.DateTimeField(auto_now_add=True)


class ActiveLearningRun(models.Model):
    """An active-learning competence-discovery run (Task 4)."""
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE,
                              related_name="al_runs")
    strategy = models.CharField(max_length=32, default="competence_gap")
    budget = models.IntegerField(default=400)
    final_l1_error = models.FloatField(default=0.0)
    history_json = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
