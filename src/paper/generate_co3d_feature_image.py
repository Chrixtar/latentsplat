import json
from pathlib import Path
from itertools import product
import svg
import torch
from hydra import compose, initialize
from jaxtyping import Float, install_import_hook
from torch import Tensor
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.global_cfg import set_cfg
    from src.misc.image_io import load_image
    from src.paper.common import encode_image, save_svg
    from src.scripts.compute_metrics import RootCfg

    from .common import MARGIN

FIGURE_WIDTH = 500
LINE_WIDTH = 0.5
TEXT_SIZE = 9


def generate_image_grid(
    images: list[list[Float[Tensor, "3 height width"] | None]],
    method_names: list[str],
    out_file_name: str
):
    num_rows = len(images)
    n = len(images[0])

    # There are two extra margins for the vertical lines.
    image_width = (FIGURE_WIDTH - (n - 2) * MARGIN) / (n - 1.5)
    figure_height = num_rows * image_width + (num_rows - 1) * MARGIN + TEXT_SIZE
    mini_image_width = (image_width - MARGIN) / 2

    # Setting width and height seems to be broken, so we manually set them here.
    fig = svg.SVG(
        width=FIGURE_WIDTH,
        height=figure_height,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, FIGURE_WIDTH, figure_height),
    )

    # Add the first context image.
    for row, (image, *_) in enumerate(images):
        if image is None:
            image = torch.ones((3, 128, 128), dtype=torch.float32) * 0.5

        # For now, assume square images.
        _, h, w = image.shape
        assert h == w

        # Compute values needed to determine the image's position.
        offset = image_width + MARGIN

        image = svg.Image(
            width=mini_image_width,
            height=mini_image_width,
            href=encode_image(image, "jpeg"),
            x=0,
            y=offset * row + TEXT_SIZE,
        )
        fig.elements.append(image)

    # Add the second context image.
    for row, (_, image, *_) in enumerate(images):
        if image is None:
            image = torch.ones((3, 128, 128), dtype=torch.float32) * 0.5

        # For now, assume square images.
        _, h, w = image.shape
        assert h == w

        # Compute values needed to determine the image's position.
        offset = image_width + MARGIN

        image = svg.Image(
            width=mini_image_width,
            height=mini_image_width,
            href=encode_image(image, "jpeg"),
            x=0,
            y=offset * row + TEXT_SIZE + MARGIN + mini_image_width,
        )
        fig.elements.append(image)

    for row, row_images in enumerate(images):
        for col, image in enumerate(row_images[2:]):
            if image is None:
                image = torch.ones((3, 128, 128), dtype=torch.float32) * 0.5

            # For now, assume square images.
            _, h, w = image.shape
            assert h == w

            # Compute values needed to determine the image's position.
            offset = image_width + MARGIN

            image = svg.Image(
                width=image_width,
                height=image_width,
                href=encode_image(image, "jpeg"),
                x=offset * col + mini_image_width + MARGIN,
                y=offset * row + TEXT_SIZE,
            )
            fig.elements.append(image)

    # Draw the context image label.
    text = svg.Text(
        x=mini_image_width / 2,
        y=TEXT_SIZE * 0.65,
        elements=["Ref."],
        font_size=TEXT_SIZE,
        text_anchor="middle",
    )
    fig.elements.append(text)

    # Draw the target image label.
    text = svg.Text(
        x=mini_image_width + image_width * 0.5 + MARGIN,
        y=TEXT_SIZE * 0.65,
        elements=["Target View"],
        font_size=TEXT_SIZE,
        text_anchor="middle",
    )
    fig.elements.append(text)

    # Draw the method names.
    for i, method_name in enumerate(method_names):
        text = svg.Text(
            x=mini_image_width + image_width * (1.5 + i) + MARGIN * (2 + i),
            y=TEXT_SIZE * 0.65,
            elements=[method_name],
            font_size=TEXT_SIZE,
            text_anchor="middle",
        )
        fig.elements.append(text)

    save_svg(fig, Path(f"{out_file_name}.svg"))


def generate_image_comparison():
    rows = []
    for dataset, evaluation in product(("co3d_teddybear",), ("360_features",)):    # co3d_teddybear # co3d_hydrant
        cfg_dict = compose(
            config_name="compute_metrics",
            # overrides=[f"+experiment={experiment}", f"+evaluation={experiment}_{evaluation}"],
            overrides=[f"dataset={dataset}", f"+evaluation={dataset}_{evaluation}"],
        )
        set_cfg(cfg_dict)
        cfg = load_typed_config(cfg_dict, RootCfg)

        # Load the evaluation index.
        assert cfg.dataset.view_sampler.name == "evaluation"

        for highlighted in tqdm(cfg.evaluation.highlighted, "Loading data"):
            scene = highlighted.scene
            context_indices = highlighted.context_index
            target_index = highlighted.target_index

            # Create a dataset to load the image.
            cfg.dataset.overfit_to_scene = scene
            cfg.dataset.view_sampler = ViewSamplerArbitraryCfg(
                "arbitrary",
                2,
                1,
                context_views=context_indices,
                target_views=[target_index],
            )
            dataset = get_dataset(cfg.dataset, "test", None)    # test

            # Load the ground truth with any dataset transformations applied.
            example = next(iter(dataset))

            # Add the context views and ground-truth target view to the grid.
            context_a, context_b = example["context"]["image"]
            gt = example["target"]["image"][0]
            row = [context_a, context_b, gt]

            context_index_str = "_".join(map(str, sorted(context_indices)))

            # Add the rendered frames to the grid.
            assert not isinstance(cfg.evaluation.methods, list), "Expected only one method for feature image"
            method = cfg.evaluation.methods
            for modality in cfg.evaluation.modalities:
                try:
                    image = load_image(
                        method.path / scene / context_index_str / f"{modality.key}/{target_index:0>6}.png"
                    )
                except FileNotFoundError:
                    image = None

                row.append(image)
            rows.append(row)

    generate_image_grid(
        rows, 
        [modality.name for modality in cfg.evaluation.modalities],
        "co3d_features"
    )


if __name__ == "__main__":
    initialize(config_path="../../config", job_name="generate_image_comparison")
    generate_image_comparison()
