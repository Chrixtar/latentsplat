import logging
from typing import List
import os
import io
import json
import numpy as np
from PIL import Image
from einops import repeat
from .utils import (
    _get_pytorch3d_camera,
    _opencv_from_cameras_projection,
)
from torch.utils.data import IterableDataset
from .dataset import DatasetCfgCommon
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from builtins import str
import torch
import torchvision.transforms as tf

from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerEvaluation


@dataclass
class DatasetCO3DCfg(DatasetCfgCommon):
    name: Literal["co3d"]
    scene: Literal["hydrant", "teddybear"]
    roots: list[Path]
    max_fov: float
    augment: bool
    planes: list[float] | None
    train_split_json: Path
    eval_split_json: Path


class DatasetCO3D(IterableDataset):
    cfg: DatasetCO3DCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor

    def __init__(
        self,
        cfg: DatasetCO3DCfg,
        stage,
        view_sampler,
        force_shuffle: bool = False
    ):
        super().__init__()
        assert cfg.image_shape is not None
        self._installed = False
        self.install()
        self.cfg = cfg
        self.categories = [cfg.scene]
        self.stage = stage
        self.path = cfg.roots[0]
        self.image_size = cfg.image_shape
        self.to_tensor = tf.ToTensor()
        self.view_sampler = view_sampler
        self.force_shuffle = force_shuffle

        self.dataset = self.get_dataset()
        self.sequence_names = list(self.dataset.keys())


    def install(self):
        if not self._installed:
            if not os.path.exists(os.path.expanduser("~/.cache/latentsplat/co3dv2")):
                import urllib.request
                import zipfile
                import shutil

                os.makedirs(os.path.expanduser("~/.cache/latentsplat"), exist_ok=True)
                with urllib.request.urlopen(
                    "https://github.com/facebookresearch/co3d/archive/c5c6c8ab1b39c70c4661581b84e0b2a5dfab1f64.zip"
                ) as f:
                    with io.BytesIO(f.read()) as bytes_io:
                        f.close()
                        with zipfile.ZipFile(bytes_io, "r") as archive:
                            archive.extractall(
                                os.path.expanduser("~/.cache/latentsplat")
                            )
                shutil.move(
                    os.path.expanduser(
                        "~/.cache/latentsplat/co3d-c5c6c8ab1b39c70c4661581b84e0b2a5dfab1f64"
                    ),
                    os.path.expanduser("~/.cache/latentsplat/co3dv2"),
                )
                logging.info(
                    f'CO3Dv2 installed to "{os.path.expanduser("~/.cache/latentsplat/co3dv2")}"'
                )

            def use_co3d_data_types():
                class ctx:
                    def __enter__(self):
                        import sys

                        sys.path.insert(
                            0, os.path.expanduser("~/.cache/latentsplat/co3dv2")
                        )
                        from co3d.dataset import data_types

                        return data_types

                    def __exit__(self, *args, **kwargs):
                        import sys

                        sys.path.remove(
                            os.path.expanduser("~/.cache/latentsplat/co3dv2")
                        )

                return ctx()

            self.use_co3d_data_types = use_co3d_data_types
            self._installed = True

    def get_dataset(self):
        sequence_to_frame_annotations = {}
        with self.use_co3d_data_types() as data_types:
            for i, c in enumerate(self.categories):
                print(f"Loading CO3D category {c} [{i+1}/{len(self.categories)}].")
                _path = f"{self.path}/{c}/frame_annotations.jgz"
                print(f"loading from this {_path}")
                category_frame_annotations = data_types.load_dataclass_jgzip(
                    _path,
                    List[data_types.FrameAnnotation],
                )

                frame_annotation_map = {
                    (x.sequence_name, x.frame_number): x
                    for x in category_frame_annotations
                }

                if self.stage in ["test", "val"] or self.cfg.overfit_to_scene:
                    json_path = self.cfg.eval_split_json
                else:
                    json_path = self.cfg.train_split_json

                with open(json_path, "r") as f:
                    try:
                        data_list = json.load(f)
                    except Exception as e:
                        print(f"Invalid file {json_path}")
                        raise e

                for seq_name, frame_num, _ in data_list:
                    if self.cfg.overfit_to_scene is None or self.cfg.overfit_to_scene == seq_name:
                        if seq_name not in sequence_to_frame_annotations:
                            sequence_to_frame_annotations[seq_name] = []
                        sequence_to_frame_annotations[seq_name].append(frame_annotation_map[(seq_name, frame_num)])
                
                for seq_name in sequence_to_frame_annotations.keys():
                    sequence_to_frame_annotations[seq_name] = list(sorted(sequence_to_frame_annotations[seq_name], key=lambda frame_annotation: frame_annotation.frame_number))

        return sequence_to_frame_annotations


    def _load_image(self, image_path: str):
        image_path = os.path.join(self.path, image_path)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Failed to load image {image_path}")
            raise e
        return image

    @staticmethod
    def _process_extrinsic(x):
        pycamera = _get_pytorch3d_camera(x)
        h, w = x.image.size
        R, T, _ = _opencv_from_cameras_projection(pycamera, torch.tensor(((h, w),)))
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        # returns in camera coords
        return w2c.inverse()

    @staticmethod
    def _check_rotation(x):
        R = x[..., :3, :3]
        return torch.allclose(torch.det(R), R.new_tensor(1))

    @staticmethod
    def _process_intrinsic(x):
        pycamera = _get_pytorch3d_camera(x)
        h, w = x.image.size
        _, _, K = _opencv_from_cameras_projection(pycamera, torch.tensor(((h, w),)))
        K = K.squeeze(0)
        # K in normalized based on W, h as per repo
        K[0, :] /= w
        K[1, :] /= h
        return K

    def _process_images(self, images):
        # track size of images
        min_h, min_w = 3000, 3000  # dummy placeholder
        for img in images:
            h, w = img.size
            min_h, min_w = min(h, min_h), min(w, min_w)

        processed_imgs = []
        for img in images:
            _img = img.resize((min_h, min_w), Image.LANCZOS)
            _img = self.to_tensor(_img)
            processed_imgs.append(_img)

        return torch.stack(processed_imgs)

    def _process_near_far(self, extrinsics):
        if self.cfg.planes is None:
            # https://github.com/facebookresearch/co3d/issues/18
            cam_loc = extrinsics[:, :3, 3]
            near = (cam_loc.norm(dim=-1) - 8).clamp_(0.5)  # to avoid -ve near
            far = cam_loc.norm(dim=-1) + 8
        else:
            near, far = self.cfg.planes
            num_views = extrinsics.shape[0]
            near = repeat(torch.tensor(near, dtype=torch.float32), "-> v", v=num_views)
            far = repeat(torch.tensor(far, dtype=torch.float32), "-> v", v=num_views)
        return near, far
    
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        if (self.stage == "train" and not self.cfg.overfit_to_scene) \
            or self.force_shuffle:
            self.sequence_names = self.shuffle(self.sequence_names)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.sequence_names = [
                seq_name 
                for i, seq_name in enumerate(self.sequence_names) 
                if i % worker_info.num_workers == worker_info.id
            ]

        for seq_name in self.sequence_names:
            seq_len = len(self.dataset[seq_name])
            try:
                view_indices = self.view_sampler.sample(seq_name, seq_len)
            except ValueError:
                # Skip because the example doesn't have enough frames.
                continue

            for view_index in view_indices:
                context_indices, target_indices = view_index.context, view_index.target

                context_examples = [self.dataset[seq_name][j.item()] for j in context_indices]
                target_examples = [self.dataset[seq_name][j.item()] for j in target_indices]

                # Skip the example if the images that smaller than cfg.image_shape.
                csize = np.stack([x.image.size for x in context_examples])
                tsize = np.stack([x.image.size for x in target_examples])
                context_image_invalid = csize <= self.cfg.image_shape
                target_image_invalid = tsize <= self.cfg.image_shape
                if context_image_invalid.any() or target_image_invalid.any():
                    print(
                        f"Skipped bad example {seq_name}. Context shape was "
                        f"{csize} and target shape was "
                        f"{tsize}."
                    )
                    continue

                # Load the images.
                context_images = [self._load_image(x.image.path) for x in context_examples]
                target_images = [self._load_image(x.image.path) for x in target_examples]
                # Skip the example if the images that smaller than cfg.image_shape.

                # Load the extrinsic.
                context_extrinsic = torch.stack(
                    [self._process_extrinsic(x) for x in context_examples], dim=0
                )
                target_extrinsic = torch.stack(
                    [self._process_extrinsic(x) for x in target_examples], dim=0
                )
                # check if rotation is not == 1; co3d teddybear
                check_cR = self._check_rotation(context_extrinsic)
                check_tR = self._check_rotation(target_extrinsic)
                if not check_cR or not check_tR:
                    print("Found rotation matrix det != 1, skipping!!")
                    continue

                # Load Near and Far
                context_near, context_far = self._process_near_far(context_extrinsic)
                target_near, target_far = self._process_near_far(target_extrinsic)

                # Load the intrinsic.
                context_intrinsic = torch.stack(
                    [self._process_intrinsic(x) for x in context_examples], dim=0
                )
                target_intrinsic = torch.stack(
                    [self._process_intrinsic(x) for x in target_examples], dim=0
                )

                # print(context_intrinsic.shape, target_intrinsic.shape)
                # Skip the example if the field of view is too wide.
                # if (get_fov(context_intrinsic).rad2deg() > self.cfg.max_fov).any():
                #     continue

                # Process images.
                context_images = self._process_images(context_images)
                target_images = self._process_images(target_images)

                example = {
                    "context": {
                        "extrinsics": context_extrinsic,
                        "intrinsics": context_intrinsic,
                        "image": context_images,
                        "near": context_near,
                        "far": context_far,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": target_extrinsic,
                        "intrinsics": target_intrinsic,
                        "image": target_images,
                        "near": target_near,
                        "far": target_far,
                        "index": target_indices,
                    },
                    "scene": seq_name,
                }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def __len__(self) -> int:
        if isinstance(self.view_sampler, ViewSamplerEvaluation):
            return self.view_sampler.total_samples
        return len(self.dataset)
