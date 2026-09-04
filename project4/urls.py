from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report/", views.report, name="report"),
    path("study/", views.consent, name="consent"),
    path("study/task/", views.task, name="task"),
    path("study/questionnaire/", views.questionnaire, name="questionnaire"),
    path("study/done/", views.done, name="done"),
    path("study/restart/", views.restart, name="restart"),
]
