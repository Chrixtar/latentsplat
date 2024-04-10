import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfg
    from src.evaluation.gt_saver import GTSaver
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    seed: int
    output_path: Path
    save_context: bool = False


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="generate_gt_image_directory",
)
def generate(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    trainer = Trainer(max_epochs=-1, accelerator="gpu")
    computer = GTSaver(cfg.output_path, cfg.save_context)
    data_module = DataModule(cfg.dataset, cfg.data_loader)
    trainer.test(computer, datamodule=data_module)

if __name__ == "__main__":
    generate()
