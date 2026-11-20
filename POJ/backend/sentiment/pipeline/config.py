from __future__ import annotations

from dataclasses import dataclass, asdict

SEED = 42


@dataclass
class ExperimentOptions:
    fast_mode: bool = True
    run_distilbert: bool = False
    run_ablation: bool = True
    run_negation: bool = True

    @property
    def max_train(self) -> int:
        return 8_000 if self.fast_mode else 50_000

    @property
    def max_sst_test(self) -> int:
        return 2_000 if self.fast_mode else 17_349

    @property
    def max_imdb_test(self) -> int:
        return 2_000 if self.fast_mode else 10_000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExperimentOptions":
        data = data or {}
        return cls(
            fast_mode=bool(data.get("fast_mode", True)),
            run_distilbert=bool(data.get("run_distilbert", False)),
            run_ablation=bool(data.get("run_ablation", True)),
            run_negation=bool(data.get("run_negation", True)),
        )
