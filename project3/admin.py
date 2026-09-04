from django.contrib import admin

from .models import TrainedModel, DeferralResult, ActiveLearningRun


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('created', 'kind', 'test_accuracy', 'n_train')
    list_filter = ('kind',)
    ordering = ('-created',)


@admin.register(DeferralResult)
class DeferralResultAdmin(admin.ModelAdmin):
    list_display = ('created', 'model', 'tau', 'system_accuracy',
                    'classifier_accuracy', 'expert_accuracy', 'coverage')
    ordering = ('-created',)


@admin.register(ActiveLearningRun)
class ActiveLearningRunAdmin(admin.ModelAdmin):
    list_display = ('created', 'model', 'strategy', 'budget', 'final_l1_error')
    list_filter = ('strategy',)
    ordering = ('-created',)
