from django.db import models


class ModelSelectionRun(models.Model):
    """A model the user chose to keep, persisted to the database.

    Project 2 pre-trains a whole family of models and the lambda slider only
    *selects* among them, so there is no "train" click to hook. Instead the user
    presses "Record this model" when they have a selection worth keeping; one
    row is written here per press. This gives the page a history of recorded
    selections and a "best so far" marker, mirroring Project 1's TrainingRun.
    """

    MODEL_CHOICES = [
        ('tree', 'Decision tree'),
        ('logistic', 'Logistic regression'),
    ]

    # What was selected
    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)
    lam = models.FloatField(help_text='Regularisation lambda at selection time')

    # The concrete model behind the selection
    omega = models.FloatField(help_text='Complexity (leaves / non-zero coefs)')
    omega_label = models.CharField(max_length=40, blank=True)
    hyperparam = models.CharField(max_length=60, blank=True,
                                  help_text='e.g. "max_leaf_nodes=6" or "C=0.5"')

    # The result
    acc_test = models.FloatField(help_text='Test accuracy (0-1)')
    acc_train = models.FloatField(help_text='Train accuracy (0-1)')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return (f"{self.get_model_type_display()} "
                f"(lambda={self.lam:.4f}, omega={self.omega:g}) "
                f"-> {self.acc_test:.3f} test acc")
