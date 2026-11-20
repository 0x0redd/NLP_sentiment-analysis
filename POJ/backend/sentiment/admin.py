from django.contrib import admin

from .models import ExperimentRun


@admin.register(ExperimentRun)
class ExperimentRunAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "created_at", "finished_at")
    list_filter = ("status",)
    readonly_fields = ("id", "created_at", "started_at", "finished_at", "summary", "error", "progress")
