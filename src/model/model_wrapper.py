from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from math import prod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Protocol, runtime_checkable, Tuple
from warnings import warn

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, optim
from torch.nn import Module, Parameter
from torchvision.transforms.functional import resize

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import LossGroupCfg, LossGroup, get_loss_group
from ..misc.benchmarker import Benchmarker
from ..misc.fraction_utils import get_integer, get_inv
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_depth_color_map
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .autoencoder.autoencoder import Autoencoder
from .decoder.decoder import Decoder, DepthRenderingMode
from .discriminator.discriminator import Discriminator
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .types import Prediction, GroundTruth, VariationalGaussians, VariationalMode


@dataclass
class LRSchedulerCfg:
    # Assumes every step as frequency
    name: str
    kwargs: Dict[str, Any] | None = None


def freeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


def unfreeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = True
    m.train()


@dataclass
class FreezeCfg:
    autoencoder: bool = False
    encoder: bool = False
    decoder: bool = False
    discrimininator: bool = False


@dataclass
class GeneratorOptimizerCfg:
    name: str
    autoencoder_lr: float
    scale_autoencoder_lr: bool
    lr: float
    scale_lr: bool
    autoencoder_kwargs: Dict[str, Any] | None = None
    kwargs: Dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
    gradient_clip_val: float | int | None = None 
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"

@dataclass
class DiscriminatorOptimizerCfg:
    name: str
    lr: float
    scale_lr: bool
    kwargs: Dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
    gradient_clip_val: float | int | None = None 
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"

@dataclass
class OptimizerCfg:
    generator: GeneratorOptimizerCfg
    discriminator: DiscriminatorOptimizerCfg | None = None

@dataclass
class TestCfg:
    output_path: Path


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    step_offset: int
    video_interpolation: bool = False
    video_wobble: bool = False


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    autoencoder: Autoencoder
    encoder: Encoder
    encode_latents: bool
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    supersampling_factor: int
    discriminator: Discriminator | None
    gaussian_losses: LossGroup
    context_losses: LossGroup
    target_autoencoder_losses: LossGroup
    target_render_latent_losses: LossGroup
    target_render_image_losses: LossGroup
    target_combined_losses: LossGroup
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    freeze_cfg: FreezeCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        freeze_cfg: FreezeCfg,
        autoencoder: Autoencoder,
        encoder: Encoder,
        encode_latents: bool,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        supersampling_factor: int = 1,
        variational: VariationalMode = "none",
        discriminator: Discriminator | None = None,
        gaussian_loss_cfg: LossGroupCfg | None = None,
        context_loss_cfg: LossGroupCfg | None = None,
        target_autoencoder_loss_cfg: LossGroupCfg | None = None,
        target_render_latent_loss_cfg: LossGroupCfg | None = None,
        target_render_image_loss_cfg: LossGroupCfg | None = None,
        target_combined_loss_cfg: LossGroupCfg | None = None,
        step_tracker: StepTracker | None = None
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        # self.strict_loading = False

        self.optimizer_cfg = optimizer_cfg

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.freeze_cfg = freeze_cfg
        self.step_tracker = step_tracker

        self.supersampling_factor = supersampling_factor
        self.variational = variational

        # Set up the model.
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.encode_latents = encode_latents
        self.encoder_visualizer = encoder_visualizer
        self.data_shim = get_data_shim(self.encoder)
        self.decoder = decoder
        self.discriminator = discriminator

        self.gaussian_losses = get_loss_group("gaussian", gaussian_loss_cfg)
        self.context_losses = get_loss_group("context", context_loss_cfg)
        self.target_autoencoder_losses = get_loss_group("target/autoencoder/loss", target_autoencoder_loss_cfg)
        self.target_render_latent_losses = get_loss_group("target/render/latent", target_render_latent_loss_cfg)
        self.target_render_image_losses = get_loss_group("target/render/image", target_render_image_loss_cfg)
        self.target_combined_losses = get_loss_group("target/combined", target_combined_loss_cfg)

        assert not self.target_render_latent_losses.has_generator_loss and not self.target_render_latent_losses.has_discriminator_loss, \
            "Cannot apply GAN losses in latent space"
        assert not self.target_render_image_losses.has_generator_loss and not self.target_render_image_losses.has_discriminator_loss, \
            "Cannot apply GAN losses on low resolution RGB space"
        
        if any(loss_group.has_generator_loss or loss_group.has_discriminator_loss \
               for loss_group in [self.context_losses, self.target_autoencoder_losses, self.target_combined_losses]):
            assert self.discriminator is not None, "Found GAN loss but no discriminator!"
            assert self.optimizer_cfg.discriminator is not None, "Found GAN loss but no discriminator optimizer config!"
            
        # Optionally freeze components
        if self.freeze_cfg.autoencoder:
            freeze(self.autoencoder)
        if self.freeze_cfg.encoder:
            freeze(self.encoder)
        if self.freeze_cfg.decoder:
            freeze(self.decoder)
        if self.freeze_cfg.discrimininator:
            freeze(self.discriminator)

        # This is used for testing.
        self.benchmarker = Benchmarker()

    @property
    def scale_factor(self) -> Fraction:
        return Fraction(self.supersampling_factor, self.autoencoder.downscale_factor)

    @property
    def last_layer_weight(self) -> Tensor:
        res = self.autoencoder.last_layer_weights
        if res is None:
            res = self.decoder.last_layer_weights
            if res is None:
                res = self.encoder.last_layer_weights
                if res is None:
                    raise ValueError("Could not find last layer weights in autoencoder, decoder, or encoder")
        return res

    @staticmethod
    def get_scaled_size(scale: Fraction, size: Iterable[int]) -> Tuple[int, ...]:
        return tuple(get_integer(scale * s) for s in size)

    def setup(self, stage: str) -> None:
        # Scale base learning rates to effective batch size
        if stage == "fit":
            assert self.trainer.accumulate_grad_batches == 1, "Gradient accumulation currently not supported because of manual optimization!"
            # assumes one fixed batch_size for all train dataloaders!
            effective_batch_size = self.trainer.accumulate_grad_batches \
                * self.trainer.num_devices \
                * self.trainer.num_nodes \
                * self.trainer.datamodule.data_loader_cfg.train.batch_size
            
            self.generator_lr = effective_batch_size * self.optimizer_cfg.generator.lr \
                if self.optimizer_cfg.generator.scale_lr else self.optimizer_cfg.generator.lr
            self.autoencoder_lr = effective_batch_size * self.optimizer_cfg.generator.autoencoder_lr \
                if self.optimizer_cfg.generator.scale_autoencoder_lr else self.optimizer_cfg.generator.autoencoder_lr
            if self.optimizer_cfg.discriminator is not None:
                self.discriminator_lr = effective_batch_size * self.optimizer_cfg.discriminator.lr \
                    if self.optimizer_cfg.discriminator.scale_lr else self.optimizer_cfg.discriminator.lr
        return super().setup(stage)

    @staticmethod
    def rescale(
        x: Float[Tensor, "... height width"], 
        scale_factor: Fraction
    ) -> Float[Tensor, "... downscaled_height downscaled_width"]:
        batch_dims = x.shape[:-2]
        spatial = x.shape[-2:]
        size = ModelWrapper.get_scaled_size(scale_factor, spatial)
        return resize(x.view(-1, *spatial), size=size, antialias=True).view(*batch_dims, *size)

    def get_active_loss_groups(self):
        return {
            "gaussian": self.gaussian_losses.is_active(self.step_tracker.get_step()),
            "context": self.context_losses.is_active(self.step_tracker.get_step()),
            "target_autoencoder": self.target_autoencoder_losses.is_active(self.step_tracker.get_step()),
            "target_render_latent": self.target_render_latent_losses.is_active(self.step_tracker.get_step()),
            "target_render_image": self.target_render_image_losses.is_active(self.step_tracker.get_step()),
            "target_combined": self.target_combined_losses.is_active(self.step_tracker.get_step()),
        }

    def training_step(self, batch, batch_idx):
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log(f"step_tracker/step", self.step_tracker.get_step())

        # Get optimizers
        opt = self.optimizers()
        if isinstance(opt, list):
            g_opt, d_opt = opt
            # Do not increment global step for discriminator step
            d_opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
            d_opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        else:
            g_opt = opt

        # Data shim, shape...
        batch: BatchedExample = self.data_shim(batch)
        v_c = batch["context"]["image"].shape[1]
        b, v_t = batch["target"]["image"].shape[:2]
        size = self.get_scaled_size(self.scale_factor, batch["target"]["image"].shape[-2:])

        # Find out active losses
        is_active_loss = self.get_active_loss_groups()

        # Prepare predictions and ground truth
        gaussian_pred = Prediction()
        context_pred = Prediction()
        context_gt = GroundTruth(batch["context"]["image"])
        target_autoencoder_pred = Prediction()
        target_autoencoder_gt = GroundTruth(batch["target"]["image"])
        target_render_latent_pred = Prediction()
        target_render_latent_gt = GroundTruth(near=batch["target"]["near"], far=batch["target"]["far"])
        target_render_image_pred = Prediction()
        target_render_image_gt = GroundTruth(
            image=self.rescale(batch["target"]["image"], self.scale_factor) if is_active_loss["target_render_image"] else None,
            near=batch["target"]["near"], 
            far=batch["target"]["far"]
        )
        target_combined_pred = Prediction()
        target_combined_gt = GroundTruth(batch["target"]["image"], near=batch["target"]["near"], far=batch["target"]["far"])

        # Run the model.
        # First generator pass
        self.toggle_optimizer(g_opt)

        # Apply Autoencoder encoder to ...
        # ... context views
        latents_to_decode = {}
        if is_active_loss["context"] or \
            (self.encode_latents and \
                (is_active_loss["target_render_latent"] or is_active_loss["target_render_image"] or is_active_loss["target_combined"])):
            context_pred.posterior = self.autoencoder.encode(batch["context"]["image"])
            context_latents = context_pred.posterior.sample()
            if is_active_loss["context"]:
                latents_to_decode["context"] = context_latents

        # ... target views
        if is_active_loss["target_autoencoder"] or is_active_loss["target_render_latent"]:
            target_autoencoder_pred.posterior = self.autoencoder.encode(batch["target"]["image"])
            target_latents = target_autoencoder_pred.posterior.sample()
            if is_active_loss["target_autoencoder"]:
                latents_to_decode["target"] = target_latents
            if is_active_loss["target_render_latent"]:
                target_render_latent_gt.image = target_latents
        
        if is_active_loss["gaussian"] or is_active_loss["target_render_latent"] or is_active_loss["target_render_image"] or is_active_loss["target_combined"]:
            gaussians: VariationalGaussians = self.encoder(
                batch["context"], 
                self.step_tracker.get_step(),
                features=context_latents if self.encode_latents else None,
                deterministic=False
            )
            if is_active_loss["gaussian"]:
                gaussian_pred.posterior = gaussians.feature_harmonics
            output = self.decoder.forward(
                gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten(),     # Sample from variational Gaussians
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                size,
                depth_mode=self.train_cfg.depth_mode,
                return_colors=is_active_loss["target_render_image"],
                return_features=is_active_loss["target_render_latent"] or is_active_loss["target_combined"]
            )
            target_render_image_pred.image = output.color
            target_render_latent_pred.posterior = output.feature_posterior
            latent_sample = output.feature_posterior.sample()
            # Invert supersampling
            z = self.rescale(latent_sample, Fraction(1, self.supersampling_factor))
            target_render_latent_pred.image = z   # TODO use kl divergence between latents instead of image loss between samples?

            if is_active_loss["target_combined"]:
                # Decode not batched with autoencoder branch because of skip connections
                if self.autoencoder.expects_skip:
                    skip_z = torch.cat((output.color.detach(), latent_sample), dim=-3) if self.autoencoder.expects_skip_extra else latent_sample
                else:
                    skip_z = None
                target_combined_pred.image = self.autoencoder.decode(z, skip_z)
        
        # Apply Autoencoder decoder batched
        if latents_to_decode:
            split_sizes = [prod(l.shape[:-3]) for l in latents_to_decode.values()]
            latents = torch.cat([l.flatten(0, -4) for l in latents_to_decode.values()])
            images = self.autoencoder.decode(latents)
            pred_images = dict(zip(latents_to_decode.keys(), images.split(split_sizes)))
            if is_active_loss["context"]:
                context_pred.image = rearrange(pred_images["context"], "(b v) c h w -> b v c h w", b=b, v=v_c)
            if is_active_loss["target_autoencoder"]:
                target_autoencoder_pred.image = rearrange(pred_images["target"], "(b v) c h w -> b v c h w", b=b, v=v_t)

        # Compute and log image metrics
        for view, pred, gt in zip(
            ("context", "target_autoencoder", "target_render", "target_combined"), 
            (context_pred, target_autoencoder_pred, target_render_image_pred, target_combined_pred),
            (context_gt, target_autoencoder_gt, target_render_image_gt, target_combined_gt)
        ):
            if gt.image is not None and pred.image is not None:
                psnr = compute_psnr(
                    rearrange(gt.image, "b v c h w -> (b v) c h w"),
                    rearrange(pred.image, "b v c h w -> (b v) c h w"),
                )
                self.log(f"train/{view}/psnr", psnr.mean())

        # Compute discriminator logits for generator losses
        for loss_group, pred in zip(
            (self.context_losses, self.target_autoencoder_losses, self.target_combined_losses),
            (context_pred, target_autoencoder_pred, target_combined_pred),
        ):
            if loss_group.is_generator_loss_active(self.step_tracker.get_step()):
                b, v = pred.image.shape[:2]
                logits_fake = self.discriminator(rearrange(pred.image, "b v c h w -> (b v) c h w"))     # TODO allow any number of batch dimensions in discriminator
                pred.logits_fake = rearrange(logits_fake, "(b v) c h w -> b v c h w", b=b, v=v)
        
        # Compute and log loss for generator
        generator_loss = 0
        for loss_group, pred, gt in zip(
            (self.gaussian_losses, self.context_losses, self.target_autoencoder_losses, self.target_render_image_losses, self.target_render_latent_losses, self.target_combined_losses),
            (gaussian_pred, context_pred, target_autoencoder_pred, target_render_image_pred, target_render_latent_pred, target_combined_pred),
            (None, context_gt, target_autoencoder_gt, target_render_image_gt, target_render_latent_gt, target_combined_gt)
        ):
            group_loss, loss_dict = loss_group.forward_generator(pred, gt, self.step_tracker.get_step(), self.last_layer_weight)
            for loss_name, loss in loss_dict.items():
                self.log(f"loss/generator/{loss_name}", loss.unweighted)
            self.log(f"loss/generator/{loss_group.name}/total", group_loss)
            generator_loss = generator_loss + group_loss

        self.log(f"loss/generator/total", generator_loss)

        if isinstance(generator_loss, Tensor):
            if not generator_loss.isnan().any():
                # Generator optimization step
                g_opt.zero_grad()
                self.manual_backward(generator_loss)
                # Clip gradients manually (manual optimization)
                self.clip_gradients(
                    g_opt, 
                    gradient_clip_val=self.optimizer_cfg.generator.gradient_clip_val, 
                    gradient_clip_algorithm=self.optimizer_cfg.generator.gradient_clip_algorithm
                )
                g_opt.step()
            else:
                warn(f"Encountered nan generator loss in iteration {self.step_tracker.get_step()}")
        
        self.untoggle_optimizer(g_opt)

        if self.discriminator is not None:
            # Second discriminator optimization
            self.toggle_optimizer(d_opt)

            discriminator_loss = 0
            for loss_group, pred, gt in zip(
                (self.context_losses, self.target_autoencoder_losses, self.target_combined_losses),
                (context_pred, target_autoencoder_pred, target_combined_pred),
                (context_gt, target_autoencoder_gt, target_combined_gt)
            ):  
                if loss_group.is_discriminator_loss_active(self.step_tracker.get_step()):
                    # TODO allow any number of batch dimensions in discriminator
                    b, v = pred.image.shape[:2]
                    logits_fake = self.discriminator(rearrange(pred.image.detach(), "b v c h w -> (b v) c h w"))    # NOTE detach here
                    logits_real = self.discriminator(rearrange(gt.image, "b v c h w -> (b v) c h w"))
                    pred.logits_fake = rearrange(logits_fake, "(b v) c h w -> b v c h w", b=b, v=v)
                    pred.logits_real = rearrange(logits_real, "(b v) c h w -> b v c h w", b=b, v=v)
                    group_loss, loss_dict = loss_group.forward_discriminator(pred, gt, self.step_tracker.get_step())
                    for loss_name, loss in loss_dict.items():
                        self.log(f"loss/discriminator/{loss_name}", loss.unweighted)
                    self.log(f"loss/discriminator/{loss_group.name}/total", group_loss)
                    discriminator_loss = discriminator_loss + group_loss

            self.log(f"loss/discriminator/total", discriminator_loss)

            if isinstance(discriminator_loss, Tensor):
                if not discriminator_loss.isnan().any():
                    # Discriminator optimization step
                    d_opt.zero_grad()
                    self.manual_backward(discriminator_loss)
                    # Clip gradients manually (manual optimization)
                    self.clip_gradients(
                        d_opt, 
                        gradient_clip_val=self.optimizer_cfg.discriminator.gradient_clip_val, 
                        gradient_clip_algorithm=self.optimizer_cfg.discriminator.gradient_clip_algorithm
                    )
                    d_opt.step()
                else:
                    warn(f"Encountered nan discriminator loss in iteration {self.step_tracker.get_step()}")
            
            self.untoggle_optimizer(d_opt)
        else:
            discriminator_loss = None

        # Print progress
        if self.global_rank == 0:
            progress = f"train step {self.step_tracker.get_step()}; " \
                f"scene = {batch['scene']}; " \
                f"context = {batch['context']['index'].tolist()}; " \
                f"generator loss = {generator_loss:.6f}; "
            if discriminator_loss is not None:
                progress += f"discriminator loss = {discriminator_loss:.6f}"
            print(progress)

        # Do all scheduler steps
        schedulers = self.lr_schedulers()
        if schedulers:  # schedulers is not None and not empty list
            if isinstance(schedulers, list):
                for scheduler in schedulers:
                    scheduler.step()
            else:
                schedulers.step()
            

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v = batch["target"]["image"].shape[:2]
        size = self.get_scaled_size(self.scale_factor, batch["target"]["image"].shape[-2:])

        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        if self.encode_latents:
            with self.benchmarker.time("autoencoder_encoder", num_calls=batch["context"]["image"].shape[1]):
                posterior = self.autoencoder.encode(batch["context"]["image"])
                context_latents = posterior.sample()
        else:
            context_latents = None

        with self.benchmarker.time("encoder"):
            gaussians: VariationalGaussians = self.encoder(
                batch["context"],
                self.step_tracker.get_step(),
                features=context_latents,
                deterministic=False,
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten(),     # Sample from variational Gaussians
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                size,
            )
        
        with self.benchmarker.time("autoencoder_decoder", num_calls=v):
            latent_sample = output.feature_posterior.sample()
            # Invert supersampling
            z = self.rescale(latent_sample, Fraction(1, self.supersampling_factor))
            if self.autoencoder.expects_skip:
                skip_z = torch.cat((output.color.detach(), latent_sample), dim=-3) if self.autoencoder.expects_skip_extra else latent_sample
            else:
                skip_z = None
            target_pred_image = self.autoencoder.decode(z, skip_z)

        # Save images.
        (scene,) = batch["scene"]
        context_index_str = "_".join(map(str, sorted(batch["context"]["index"][0].tolist())))
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        for index, color in zip(batch["target"]["index"][0], target_pred_image[0]):
            save_image(color, path / scene / context_index_str / f"color/{index:0>6}.png")

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.step_tracker.get_step()}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        b, v = batch["target"]["image"].shape[:2]
        assert b == 1

        size = self.get_scaled_size(self.scale_factor, batch["target"]["image"].shape[-2:])

        pred = {
            "low": {},
            "high": {}
        }
        # Render Gaussians.
        # Probabilistic pass
        if self.encode_latents:
            posterior = self.autoencoder.encode(batch["context"]["image"])
            context_latents = posterior.sample()
        else:
            context_latents = None

        gaussians_probabilistic: VariationalGaussians = self.encoder(
            batch["context"],
            self.step_tracker.get_step(),
            features=context_latents,
            deterministic=False,
        )
        output_probabilistic = self.decoder.forward(
            gaussians_probabilistic.sample() if self.variational in ("gaussians", "none") else gaussians_probabilistic.flatten(),
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            size,
        )
        pred["low"]["probabilistic"] = output_probabilistic.color[0]
        latent_probabilistic = output_probabilistic.feature_posterior.sample()[0]

        # Deterministic pass
        gaussians_deterministic: VariationalGaussians = self.encoder(
            batch["context"],
            self.step_tracker.get_step(),
            features=posterior.mode() if self.encode_latents else None,
            deterministic=True,
        )
        output_deterministic = self.decoder.forward(
            gaussians_deterministic.mode() if self.variational in ("gaussians", "none") else gaussians_probabilistic.flatten(),
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            size,
        )
        pred["low"]["deterministic"] = output_deterministic.color[0]
        latent_deterministic = output_deterministic.feature_posterior.mode()[0]

        latents = torch.cat((latent_probabilistic, latent_deterministic))
        # Invert supersampling
        z = self.rescale(latents, Fraction(1, self.supersampling_factor))
        if self.autoencoder.expects_skip:
            if self.autoencoder.expects_skip_extra:
                colors = torch.cat((pred["low"]["probabilistic"], pred["low"]["deterministic"]))
                skip_z = torch.cat((colors, latents), dim=-3)   # -3 = channel dimension
            else:
                skip_z = latents
        else:
            skip_z = None
        dec = self.autoencoder.decode(z, skip_z)
        pred["high"]["probabilistic"], pred["high"]["deterministic"] = dec.tensor_split(2)

        # Compute validation metrics.
        rgb_high_res_gt = batch["target"]["image"][0]
        rgb_low_res_gt = self.rescale(rgb_high_res_gt, self.scale_factor)

        for mode in ("deterministic", "probabilistic"):
            # Skip lpips and ssim for low resolution, because too small
            score = compute_psnr(rgb_low_res_gt, pred["low"][mode]).mean()
            self.log(f"val/{mode}/low/psnr", score, rank_zero_only=True)
            for metric_name, metric in zip(
                ("psnr", "lpips", "ssim"),
                (compute_psnr, compute_lpips, compute_ssim)
            ):
                score = metric(rgb_high_res_gt, pred["high"][mode]).mean()
                self.log(f"val/{mode}/high/{metric_name}", score, rank_zero_only=True)

        # Construct comparison image.
        comparison = {}
        # Skip high resolution context images and labels for low resolution comparison
        comparison["low"] = add_border(
            hcat(
                vcat(*rgb_low_res_gt, gap=1),
                vcat(*pred["low"]["probabilistic"], gap=1),
                vcat(*pred["low"]["deterministic"], gap=1),
                gap=1
            ),
            border=1
        )
        comparison["high"] = add_border(hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_high_res_gt), "Target (Ground Truth)"),
            add_label(vcat(*pred["high"]["probabilistic"]), "Target (Probabilistic)"),
            add_label(vcat(*pred["high"]["deterministic"]), "Target (Deterministic)"),
        ))
        for res, comp in comparison.items():
            self.logger.log_image(
                f"comparison_{res}",
                [prep_image(comp)],
                step=self.step_tracker.get_step(),
                caption=batch["scene"],
            )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        # TODO can we adapt this for latent space?
        """
        projections = vcat(
            hcat(
                *render_projections(
                    gaussians_probabilistic,
                    256,
                    extra_label="(Probabilistic)",
                )[0]
            ),
            hcat(
                *render_projections(
                    gaussians_deterministic, 256, extra_label="(Deterministic)"
                )[0]
            ),
            align="left",
        )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.step_tracker.get_step(),
        )
        """

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.step_tracker.get_step()
        )

        # TODO adapt encoder_visualizer
        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], 
                self.step_tracker.get_step(),
                features=posterior.mode() if self.encode_latents else None
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.step_tracker.get_step())

        # Run video validation step.
        if self.train_cfg.video_interpolation:
            self.render_video_interpolation(batch)
        if self.train_cfg.video_wobble:
            self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        if self.encode_latents:
            posterior = self.autoencoder.encode(batch["context"]["image"])

        gaussians_prob: VariationalGaussians = self.encoder(
            batch["context"], 
            self.step_tracker.get_step(),
            features=posterior.sample() if self.encode_latents else None,
            deterministic=False
        )
        gaussians_det: VariationalGaussians = self.encoder(
            batch["context"], 
            self.step_tracker.get_step(),
            features=posterior.mode() if self.encode_latents else None,
            deterministic=True
        )

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)
        size = self.get_scaled_size(self.scale_factor, batch["context"]["image"].shape[-2:])

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)

        output_prob = self.decoder.forward(
            gaussians_prob.sample() if self.variational in ("gaussians", "none") else gaussians_prob.flatten(),
            extrinsics, intrinsics, near, far, size, "depth"
        )
        output_det = self.decoder.forward(
            gaussians_det.mode() if self.variational in ("gaussians", "none") else gaussians_prob.flatten(),
            extrinsics, intrinsics, near, far, size, "depth"
        )
        latent_prob = output_prob.feature_posterior.sample()
        latent_det = output_prob.feature_posterior.mode()


        latents = torch.cat((latent_prob, latent_det))
        # Invert supersampling
        z = self.rescale(latents, Fraction(1, self.supersampling_factor))
        if self.autoencoder.expects_skip:
            if self.autoencoder.expects_skip_extra:
                colors = torch.cat((output_prob.color, output_det.color))
                skip_z = torch.cat((colors, latents), dim=-3)   # -3 = channel dimension
            else:
                skip_z = latents
        else:
            skip_z = None
        dec = self.autoencoder.decode(z, skip_z)

        image_prob, image_det = dec.tensor_split(2)
        pred = {}
        for mode, output, images in zip(("probabilistic", "deterministic"), (output_prob, output_det), (image_prob, image_det)):
            masks = repeat(self.rescale(output.mask, get_inv(self.scale_factor))[0].unsqueeze(1), "v () h w -> v c h w", c=3)
            depths = apply_depth_color_map(self.rescale(output.depth, get_inv(self.scale_factor))[0])
            pred[mode] = [vcat(image, mask, depth) for image, mask, depth in zip(images[0], masks, depths)]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(pred["probabilistic"], pred["deterministic"])
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.step_tracker.get_step():0>6}.mp4"), logger=None
                )

    @staticmethod
    def get_optimizer(
        optimizer_cfg: GeneratorOptimizerCfg | DiscriminatorOptimizerCfg,
        params: Iterator[Parameter] | list[Dict[str, Any]],
        lr: float
    ) -> optim.Optimizer:
        return getattr(optim, optimizer_cfg.name)(
            params,
            lr=lr,
            **(optimizer_cfg.kwargs if optimizer_cfg.kwargs is not None else {})       
        )

    @staticmethod
    def get_lr_scheduler(
        opt: optim.Optimizer, 
        lr_scheduler_cfg: LRSchedulerCfg
    ) -> optim.lr_scheduler.LRScheduler:
        return getattr(optim.lr_scheduler, lr_scheduler_cfg.name)(
            opt,
            **(lr_scheduler_cfg.kwargs if lr_scheduler_cfg.kwargs is not None else {})     
        )

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        # Generator optimizer
        g_opt = self.get_optimizer(
            self.optimizer_cfg.generator,
            [
                {"params": chain(self.encoder.parameters(), self.decoder.parameters())},
                {"params": self.autoencoder.parameters(), "lr": self.autoencoder_lr} \
                    | (self.optimizer_cfg.generator.autoencoder_kwargs if self.optimizer_cfg.generator.autoencoder_kwargs is not None else {})
            ],
            self.generator_lr
        )
        optimizers.append(g_opt)
        # Generator scheduler
        if self.optimizer_cfg.generator.scheduler is not None:
            schedulers.append(self.get_lr_scheduler(g_opt, self.optimizer_cfg.generator.scheduler))
        
        # Discriminator optimizer
        if self.discriminator is not None:
            d_opt = self.get_optimizer(self.optimizer_cfg.discriminator, self.discriminator.parameters(), self.discriminator_lr)
            optimizers.append(d_opt)
            # Discriminator scheduler
            if self.optimizer_cfg.discriminator.scheduler is not None:
                schedulers.append(self.get_lr_scheduler(d_opt, self.optimizer_cfg.discriminator.scheduler))
        
        return optimizers, schedulers
