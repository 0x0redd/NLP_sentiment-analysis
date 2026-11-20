from rest_framework import serializers

from .models import ExperimentRun


class ExperimentRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentRun
        fields = [
            "id",
            "status",
            "options",
            "progress",
            "error",
            "summary",
            "created_at",
            "started_at",
            "finished_at",
        ]
        read_only_fields = fields


class CreateRunSerializer(serializers.Serializer):
    fast_mode = serializers.BooleanField(default=True)
    run_distilbert = serializers.BooleanField(default=False)
    run_ablation = serializers.BooleanField(default=True)
    run_negation = serializers.BooleanField(default=True)


class PredictSerializer(serializers.Serializer):
    text = serializers.CharField(allow_blank=False, max_length=10000)
    models = serializers.ListField(
        child=serializers.CharField(), required=False, allow_empty=True
    )
    preprocess = serializers.ChoiceField(choices=["article", "negation"], default="article")


class PreprocessSerializer(serializers.Serializer):
    text = serializers.CharField(allow_blank=False, max_length=10000)
