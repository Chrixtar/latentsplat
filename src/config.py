from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossGroupCfg
from .model.autoencoder import AutoencoderCfg
from .model.decoder import DecoderCfg
from .model.discriminator import DiscriminatorCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import FreezeCfg, OptimizerCfg, TestCfg, TrainCfg
from .model.types import VariationalMode


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    resume: bool = False


@dataclass
class ModelCfg:
    autoencoder: AutoencoderCfg
    decoder: DecoderCfg
    discriminator: DiscriminatorCfg | None
    encoder: EncoderCfg
    encode_latents: bool
    supersampling_factor: int
    variational: VariationalMode = "none"


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None


@dataclass
class TargetRenderLossCfg:
    latent: LossGroupCfg | None
    image: LossGroupCfg | None


@dataclass
class TargetLossCfg:
    autoencoder: LossGroupCfg | None
    render: TargetRenderLossCfg
    combined: LossGroupCfg | None


@dataclass
class LossCfg:
    gaussian: LossGroupCfg | None
    context: LossGroupCfg | None
    target: TargetLossCfg


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "val", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: LossCfg
    test: TestCfg
    train: TrainCfg
    freeze: FreezeCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {},
    )
