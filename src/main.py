from fractions import Fraction
import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.autoencoder import get_autoencoder
    from src.model.decoder import get_decoder
    from src.model.discriminator import get_discriminator
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.activated:
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            entity=cfg_dict.wandb.entity
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_last=True
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker(cfg.train.step_offset)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
    )

    autoencoder = get_autoencoder(cfg.model.autoencoder)    
    encoder, encoder_visualizer = get_encoder(
        cfg.model.encoder,
        d_in=autoencoder.d_latent if cfg.model.encode_latents else 3,
        n_feature_channels=autoencoder.d_latent,
        scale_factor=Fraction(cfg.model.supersampling_factor, 1 if cfg.model.encode_latents else autoencoder.downscale_factor),
        variational=cfg.model.variational != "none"
    )
    decoder = get_decoder(cfg.model.decoder, cfg.dataset.background_color, cfg.model.variational == "latents")
    
    kwargs = dict(
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        freeze_cfg=cfg.freeze,
        autoencoder=autoencoder,
        encoder=encoder,
        encode_latents=cfg.model.encode_latents,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        supersampling_factor=cfg.model.supersampling_factor,
        variational=cfg.model.variational,
        discriminator=get_discriminator(cfg.model.discriminator) if cfg.model.discriminator is not None else None,
        gaussian_loss_cfg=cfg.loss.gaussian,
        context_loss_cfg=cfg.loss.context,
        target_autoencoder_loss_cfg=cfg.loss.target.autoencoder,
        target_render_latent_loss_cfg=cfg.loss.target.render.latent,
        target_render_image_loss_cfg=cfg.loss.target.render.image,
        target_combined_loss_cfg=cfg.loss.target.combined,
        step_tracker=step_tracker
    )
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights but no optimizer state
        model_wrapper = ModelWrapper.load_from_checkpoint(checkpoint_path, **kwargs, strict=False)
    else:
        model_wrapper = ModelWrapper(**kwargs)
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path if cfg.checkpointing.resume else None)
    elif cfg.mode == "val":
        trainer.validate(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    elif cfg.mode == "test":
        trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    train()
