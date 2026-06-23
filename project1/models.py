from django.db import models


class TrainingRun(models.Model):
    """A single model-training attempt, persisted to the database.

    Every time the user trains a model in the interface, one row is written
    here. This lets the app show a history of past runs and identify the best
    one, demonstrating use of the Django ORM rather than only the session.
    """

    PROBLEM_CHOICES = [
        ('classification', 'Classification'),
        ('regression', 'Regression'),
    ]

    # What was trained
    dataset_name = models.CharField(max_length=200, default='uploaded dataset')
    model_name = models.CharField(max_length=50)
    problem_type = models.CharField(max_length=20, choices=PROBLEM_CHOICES)

    # How it was configured
    hyperparameter = models.IntegerField()
    hyperparameter_label = models.CharField(max_length=50, blank=True)
    test_size = models.IntegerField(help_text='Test set size in percent')

    # The result. For classification this is accuracy (0-100); for
    # regression it is the R-squared score. score_label records which.
    score = models.FloatField()
    score_label = models.CharField(max_length=30, default='Accuracy')

    # When
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return (f"{self.model_name} ({self.hyperparameter_label}="
                f"{self.hyperparameter}) -> {self.score} {self.score_label}")
