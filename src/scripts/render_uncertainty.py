from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config, RootCfg as MainCfg
    from src.dataset.data_module import DataModule
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.autoencoder import get_autoencoder
    from src.model.decoder import get_decoder
    from src.model.discriminator import get_discriminator
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.types import Gaussians, VariationalGaussians
    from src.visualization.color_map import apply_depth_color_map


# NUM_EXAMPLES = 10
# LEFT = 45
# NUM_TARGETS = 4
# CONTEXT_DIST = 10
# MAX_VIEW_INDEX = 90
INF = 1_000_000_000
FEATURE_PERCENTILES = (0.05, 0.95)
STD_PERCENTILES = (0.05, 0.95)
SH_DEG = 0
MAX_TARGET_CHUNK_SIZE = 10


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(torch.nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def feature_map_to_rgb(feat):
    """
    feat: [..., d, h, w]
    """
    batch_dims, (h, w) = feat.shape[:-3], feat.shape[-2:]
    feat = feat.flatten(0, -4).flatten(-2)  # [n, d, h*w]
    feat = feat.transpose(1, 2).flatten(0, 1)   # [n*h*w, d]
    pca = PCA(3)
    pca.fit(feat)
    rgb = pca.transform(feat)
    min, max = torch.quantile(rgb, FEATURE_PERCENTILES[0], dim=0), torch.quantile(rgb, FEATURE_PERCENTILES[1], dim=0)
    rgb = (rgb - min) / (max - min)
    
    rgb = rgb.view(-1, h*w, 3)
    rgb = rgb.transpose(1, 2)
    rgb = rgb.reshape(*batch_dims, 3, h, w)
    return rgb


def feat_std_to_rgb(feat_std: torch.Tensor):
    """
    feat: [..., d, h, w]
    """
    std = feat_std.mean(dim=-3, keepdim=True)
    # min, max = std.min(), std.max()
    min, max = torch.quantile(std, STD_PERCENTILES[0]), torch.quantile(std, STD_PERCENTILES[1])
    std = (std - min) / (max - min)
    std = torch.clamp(std, 0.0, 1.0)
    std = 1 - std   # NOTE flip std to visualize high uncertainty as black
    rgb = std.expand(*std.shape[:-3], 3, -1, -1)
    return rgb

@dataclass
class RootCfg(MainCfg):
    output_path: Path


# Run with something like:
# python -m src.scripts.render_uncertainty +output_path=outputs/render_uncertainty/co3d_hydrant_360 checkpointing.load=outputs/checkpoints_200k/co3d_hydrant_kl.ckpt +experiment=co3d_hydrant_f8_256_vg_gan_skip_kl0.1 mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index/co3d_hydrant_360.json


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)

@torch.no_grad()
def render(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    assert checkpoint_path is not None

    autoencoder = get_autoencoder(cfg.model.autoencoder)    
    encoder, encoder_visualizer = get_encoder(
        cfg.model.encoder,
        d_in=autoencoder.d_latent if cfg.model.encode_latents else 3,
        n_feature_channels=autoencoder.d_latent,
        scale_factor=Fraction(cfg.model.supersampling_factor, 1 if cfg.model.encode_latents else autoencoder.downscale_factor),
        variational=cfg.model.variational != "none"
    )
    decoder = get_decoder(cfg.model.decoder, cfg.dataset.background_color, cfg.model.variational == "latents")
    
    model = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
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
        strict=False
    )

    assert model.variational in ("gaussians", "none")

    model = model.cuda()

    cfg.data_loader.test.batch_size = 1
    """
    target_views = [0, 50, 75]
    # target_views = torch.linspace(LEFT + CONTEXT_DIST / 2, MAX_VIEW_INDEX, NUM_TARGETS, dtype=torch.int64).tolist()
    cfg.dataset.view_sampler = ViewSamplerArbitraryCfg(
        "arbitrary",
        2,
        len(target_views),
        context_views=[LEFT, LEFT+CONTEXT_DIST],
        target_views=target_views,
    )
    """
    step_tracker = StepTracker(INF)
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker=step_tracker)
    dataset = iter(data_module.test_dataloader())
    
    # Keep all latents and stds for later combined processing
    latent_dict = {}
    std_dict = {}

    for batch in tqdm(dataset):
        batch = data_module.transfer_batch_to_device(batch, device=model.device, dataloader_idx=0)

        # Re-create test step without autoencoder application
        batch = model.data_shim(batch)
        b, v = batch["target"]["image"].shape[:2]
        assert b == 1
        size = model.get_scaled_size(model.scale_factor, batch["target"]["image"].shape[-2:])

        # Render Gaussians.
        if model.encode_latents:
            posterior = model.autoencoder.encode(batch["context"]["image"])
            context_latents = posterior.sample()
        else:
            context_latents = None

        variational_gaussians: VariationalGaussians = model.encoder(
            batch["context"],
            INF,                            # NOTE expects training to be finished!
            features=context_latents,
            deterministic=False,
        )
        # Sample from variational Gaussians
        gaussians = variational_gaussians.sample() if model.variational in ("gaussians", "none") else variational_gaussians.flatten()
        feature_harmonics = gaussians.feature_harmonics
        std_harmonics = variational_gaussians.feature_harmonics.std[..., :(SH_DEG+1)**2]

        target_index_list = batch["target"]["index"].split(MAX_TARGET_CHUNK_SIZE, dim=1)
        target_extr_list = batch["target"]["extrinsics"].split(MAX_TARGET_CHUNK_SIZE, dim=1)
        target_intr_list = batch["target"]["intrinsics"].split(MAX_TARGET_CHUNK_SIZE, dim=1)
        target_near_list = batch["target"]["near"].split(MAX_TARGET_CHUNK_SIZE, dim=1)
        target_far_list = batch["target"]["far"].split(MAX_TARGET_CHUNK_SIZE, dim=1)
        target_image_list = batch["target"]["image"].split(MAX_TARGET_CHUNK_SIZE, dim=1)

        (scene,) = batch["scene"]
        context_index_str = "_".join(map(str, sorted(batch["context"]["index"][0].tolist())))
        output_path = cfg.output_path / scene / context_index_str
        if output_path not in latent_dict:
            latent_dict[output_path] = []
        if output_path not in std_dict:
            std_dict[output_path] = []

        for index, extr, intr, near, far, image in zip(
            target_index_list,
            target_extr_list,
            target_intr_list,
            target_near_list,
            target_far_list,
            target_image_list
        ):
            gaussians.feature_harmonics = feature_harmonics
            output = model.decoder.forward(gaussians, extr, intr, near, far, size)
        
            latent_sample = output.feature_posterior.sample()
            # Invert supersampling
            z = model.rescale(latent_sample, Fraction(1, model.supersampling_factor))
            if model.autoencoder.expects_skip:
                skip_z = torch.cat((output.color.detach(), latent_sample), dim=-3) if model.autoencoder.expects_skip_extra else latent_sample
            else:
                skip_z = None

            aux_pred_image = output.color
            target_pred_image = model.autoencoder.decode(z, skip_z)
            depths = apply_depth_color_map(output.depth)

            # Convert variational gaussians to std gaussians
            gaussians.feature_harmonics = std_harmonics
            output = model.decoder.forward(gaussians, extr, intr, near, far, size)
            
            std = output.feature_posterior.mode()
            # Handle background uncertainty!
            std = std + 1 - output.mask.unsqueeze(-3)

            # std_image = feat_std_to_rgb(std)
            # latent_pca_image = feature_map_to_rgb(latent_sample)

            # Save images.
            for i, target_index in enumerate(index[0]):
                for context_index, context_image in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                    save_image(context_image, output_path / f"context/{context_index:0>6}.png")
                save_image(aux_pred_image[0, i], output_path/ f"aux/{target_index:0>6}.png")
                save_image(target_pred_image[0, i], output_path / f"pred/{target_index:0>6}.png")
                save_image(output.mask[0, i], output_path / f"alpha/{target_index:0>6}.png")
                save_image(depths[0, i], output_path / f"depth/{target_index:0>6}.png")
                save_image(image[0, i], output_path / f"gt/{target_index:0>6}.png")
                latent_dict[output_path].append((target_index.item(), latent_sample[0, i].cpu()))
                std_dict[output_path].append((target_index.item(), std[0, i].cpu()))

    for output_path in latent_dict.keys():
        target_indices, latent = zip(*latent_dict[output_path])
        _, std = zip(*std_dict[output_path])
        latent = torch.stack(latent)
        std = torch.stack(std)
        latent_pca_image = feature_map_to_rgb(latent)
        std_image = feat_std_to_rgb(std)
        for i, target_index in enumerate(target_indices):
            save_image(latent_pca_image[i], output_path / f"latent/{target_index:0>6}.png")
            save_image(std_image[i], output_path / f"std/{target_index:0>6}.png")

if __name__ == "__main__":
    render()
