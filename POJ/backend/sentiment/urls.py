from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health),
    path("runs/", views.runs),
    path("runs/<uuid:run_id>/", views.run_detail),
    path("runs/<uuid:run_id>/results/", views.run_results),
    path("runs/<uuid:run_id>/mcnemar/", views.run_mcnemar),
    path("runs/<uuid:run_id>/ablation/", views.run_ablation),
    path("runs/<uuid:run_id>/confusion/", views.run_confusion),
    path("runs/<uuid:run_id>/timings/", views.run_timings),
    path("runs/<uuid:run_id>/controlled/", views.run_controlled),
    path("runs/<uuid:run_id>/activate/", views.run_activate),
    path("predict/", views.predict),
    path("preprocess/", views.preprocess_view),
    path("samples/", views.sample_text),
]
