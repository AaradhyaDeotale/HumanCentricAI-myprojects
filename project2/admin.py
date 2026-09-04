from django.contrib import admin

from .models import ModelSelectionRun


@admin.register(ModelSelectionRun)
class ModelSelectionRunAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'model_type', 'lam', 'omega',
                    'hyperparam', 'acc_test', 'acc_train')
    list_filter = ('model_type',)
    ordering = ('-created_at',)
