import os
from pathlib import Path

from pytorch_lightning import LightningModule

from ..misc.image_io import save_image


class GTSaver(LightningModule):

    def __init__(self, output_path: Path, save_context: bool = False) -> None:
        super(GTSaver, self).__init__()
        self.output_path = output_path
        self.save_context = save_context

    def test_step(self, batch, batch_idx):
        scene = batch["scene"][0]
        b, cv, _, _, _ = batch["context"]["image"].shape
        assert b == 1 and cv == 2
        _, v, _, _, _ = batch["target"]["image"].shape

        context_index_str = "_".join(map(str, sorted(batch["context"]["index"][0].tolist())))
        target_dir_path = self.output_path / scene / context_index_str / "color"
        context_dir_path = self.output_path / scene / context_index_str / "context"

        for i in range(v):
            true_index = batch["target"]["index"][0, i]
            gt_image = batch["target"]["image"][0, i]
            save_image(
                gt_image,
                target_dir_path / f"{true_index:0>6}.png"
            )

        if self.save_context:
            for i in range(cv):
                true_index = batch["context"]["index"][0, i]
                gt_image = batch["context"]["image"][0, i]
                save_image(
                    gt_image,
                    context_dir_path / f"{true_index:0>6}.png"
                )
