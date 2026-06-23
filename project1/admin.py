from django.contrib import admin
from .models import TrainingRun


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'dataset_name', 'model_name',
                    'problem_type', 'hyperparameter', 'score', 'score_label')
    list_filter = ('problem_type', 'model_name')
    ordering = ('-created_at',)
