from django.contrib import admin

from .models import StudySession, PairwiseResponse, RankingResponse, Questionnaire


@admin.register(StudySession)
class StudySessionAdmin(admin.ModelAdmin):
    list_display = ("id", "order", "started", "finished", "completion_code")
    list_filter = ("order", "finished")


@admin.register(PairwiseResponse)
class PairwiseResponseAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "phase", "trial_index", "movie_a", "movie_b",
                    "chosen", "response_ms", "created")
    list_filter = ("phase",)


@admin.register(RankingResponse)
class RankingResponseAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "trial_index", "response_ms", "created")


@admin.register(Questionnaire)
class QuestionnaireAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "preferred_method", "pairwise_difficulty",
                    "ranking_difficulty", "pairwise_confidence", "ranking_confidence")
