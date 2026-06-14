"""Injection schedule module."""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Injection:
    source_idx: int
    sink_idx: int
    current: float


class InjectionSchedule:
    """Schedule of active injections for training or measurement."""

    def __init__(self, injections: list[Injection]):
        self.injections = injections

    def __len__(self) -> int:
        return len(self.injections)

    def __getitem__(self, idx: int) -> Injection:
        return self.injections[idx]

    @classmethod
    def from_config(
        cls,
        acquisition_config: Optional[dict],
        fallback_source_model_config: Optional[dict]
    ) -> InjectionSchedule:
        """
        Builds the schedule from acquisition config.
        Falls back to legacy inverse.source_model config if acquisition config is missing.
        """
        if acquisition_config and "injections" in acquisition_config:
            injections = []
            for inj in acquisition_config["injections"]:
                injections.append(
                    Injection(
                        source_idx=int(inj["source"]),
                        sink_idx=int(inj["sink"]),
                        current=float(inj["current"]),
                    )
                )
            return cls(injections)

        # Fallback to legacy
        if fallback_source_model_config is None:
            raise ValueError("No injection schedule provided and no fallback source model found.")

        fixed_pair = fallback_source_model_config.get("fixed_electrode_pair", [0, 1])
        current = float(fallback_source_model_config.get("current", 1.0))
        
        if not isinstance(fixed_pair, (list, tuple)) or len(fixed_pair) != 2:
            raise ValueError("source_model.fixed_electrode_pair must be [source_idx, sink_idx]")

        return cls([
            Injection(
                source_idx=int(fixed_pair[0]),
                sink_idx=int(fixed_pair[1]),
                current=current,
            )
        ])
