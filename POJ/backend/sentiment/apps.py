import os
import sys

from django.apps import AppConfig


class SentimentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sentiment"

    def ready(self) -> None:
        # Skip Django reloader parent process
        if os.environ.get("RUN_MAIN") == "false":
            return
        if any(cmd in sys.argv for cmd in ("makemigrations", "migrate", "collectstatic", "test")):
            return
        try:
            from django.conf import settings

            from .pipeline.registry import registry

            if registry.loaded_models:
                return
            bootstrap = getattr(settings, "BOOTSTRAP_ARTIFACTS_DIR", None)
            if bootstrap and Path_exists(bootstrap):
                registry.load_bootstrap(bootstrap)
        except Exception:
            pass


def Path_exists(p) -> bool:
    from pathlib import Path

    return Path(p).exists()
