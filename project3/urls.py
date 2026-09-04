from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("history/", views.history, name="history"),
    path("train/", views.train, name="train"),
    path("defer/", views.defer, name="defer"),
    path("active/", views.active, name="active"),
    path("classify/", views.classify, name="classify"),
    path("human/start/", views.human_start, name="human_start"),
    path("human/answer/", views.human_answer, name="human_answer"),
    path("human/finish/", views.human_finish, name="human_finish"),
    path("report/", views.report, name="report"),
]
