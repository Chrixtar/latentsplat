import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from tqdm import tqdm

from ..misc.image_io import save_image
from .types import IndexEntry
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat


@dataclass
class CO3DEvaluationIndexGeneratorCfg:
    num_context_pairs_per_scene: int
    num_target_views: int
    min_context_distance: int
    max_context_distance: int
    output_path: Path
    save_previews: bool
    seed: int
    intra_context: bool


class CO3DEvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: CO3DEvaluationIndexGeneratorCfg
    index: dict[str, list[IndexEntry]]

    def __init__(self, cfg: CO3DEvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}
    
    def save_preview(self, batch, delta, context_left, context_right):
        preview_path = self.cfg.output_path / "previews"
        preview_path.mkdir(exist_ok=True, parents=True)
        a = batch["target"]["image"][0, context_left]
        b = batch["target"]["image"][0, context_right]
        vis = add_border(add_border(hcat(a, b)), 1, 0)
        vis = add_label(vis, f"Distance: {delta} frames")
        save_image(add_border(vis), preview_path / f"{batch['scene'][0]}.png")

    def test_step(self, batch, batch_idx):
        b, v = batch["target"]["image"].shape[:2]
        assert b == 1
        scene = batch["scene"][0]
        views = []

        context_indices = torch.randperm(v, generator=self.generator)[:self.cfg.num_context_pairs_per_scene]
        n_context_indices = context_indices.shape[0]
        partner_indices = torch.arange(self.cfg.min_context_distance, self.cfg.max_context_distance)
        partner_indices = torch.cat((-partner_indices, partner_indices))

        partner_indices = context_indices.view(-1, 1) + partner_indices.view(1, -1)   # [v, valid_partners]
        chosen = torch.randint(0, partner_indices.shape[1], size=(n_context_indices,), generator=self.generator)
        partner_indices = partner_indices[torch.arange(n_context_indices), chosen]  # [v]
        # partner_indices = partner_indices % v # TODO Later!

        context_pairs = torch.stack((context_indices, partner_indices), dim=1)  # [v, 2]
        context_pairs = torch.sort(context_pairs, dim=1).values

        delta = context_pairs[:, 1] - context_pairs[:, 0]

        for i in tqdm(range(n_context_indices), "Computing target views"):
            context_left, context_right = context_pairs[i]
            # Pick non-repeated random target views.
            if self.cfg.intra_context:
                # Pick non-repeated random target views.
                target_views = torch.arange(context_left, context_right + 1)
            else:
                if context_left < 0 and context_right < v:
                    target_views = torch.arange(context_right + 1, context_left % v)
                elif context_left >= 0 and context_right < v:
                    target_views = torch.cat((torch.arange(0, context_left), torch.arange(context_right, v)))
                elif context_left >= 0 and context_right >= v:
                    target_views = torch.arange(context_right % v + 1, context_left)
                else:
                    raise ValueError("Impossible")
                
            if len(target_views) < self.cfg.num_target_views:
                continue
                
            rand_idx = torch.randperm(target_views.shape[0], generator=self.generator)
            target_views = target_views[rand_idx][:self.cfg.num_target_views]
            target_views = torch.sort(target_views).values
            # Convert negative indices to positive ones because of circular cameras
            # NOTE do not swap again, because the order is already correct perceptually
            context_left %= v
            context_right %= v
            target_views = target_views % v

            context_left = context_left.item()
            context_right = context_right.item()
            target = tuple(target_views.tolist())
            views.append(IndexEntry(
                context=(context_left, context_right),
                target=target,
            ))
            # Optionally, save a preview.
            if self.cfg.save_previews:
                self.save_preview(batch, delta[i], context_left, context_right)

        self.index[scene] = views
        

    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        with (self.cfg.output_path / "evaluation_index.json").open("w") as f:
            json.dump(
                {k: [asdict(v) for v in views] for k, views in self.index.items()}, f
            )
