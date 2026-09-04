from django.urls import path

from . import views

app_name = "project2"

urlpatterns = [
    path("", views.index, name="index"),
    # JSON API endpoints consumed by the front-end
    path("api/meta/", views.api_meta, name="api_meta"),
    path("api/model/", views.api_model, name="api_model"),
    path("api/counterfactuals/", views.api_counterfactuals, name="api_counterfactuals"),
    path("api/feature-effect/", views.api_feature_effect, name="api_feature_effect"),
    path("api/record/", views.api_record, name="api_record"),
    path("api/history/", views.api_history, name="api_history"),
]
