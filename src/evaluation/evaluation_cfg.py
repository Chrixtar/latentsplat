from dataclasses import dataclass
from pathlib import Path


@dataclass
class MethodCfg:
    name: str
    key: str
    path: Path


@dataclass
class ModalityCfg:
    name: str
    key: str


@dataclass
class SceneCfg:
    scene: str
    context_index: list[int]
    target_index: int | list[int]


@dataclass
class EvaluationCfg:
    methods: list[MethodCfg] | MethodCfg
    side_by_side_path: Path | None
    animate_side_by_side: bool
    highlighted: list[SceneCfg]
    modalities: list[ModalityCfg] | None = None
