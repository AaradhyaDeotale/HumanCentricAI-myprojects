from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("train/", views.train, name="train"),
    path("defer/", views.defer, name="defer"),
    path("active/", views.active, name="active"),
    path("classify/", views.classify, name="classify"),
    path("report/", views.report, name="report"),
]
