# latentSplat

[![arXiv](https://img.shields.io/badge/arXiv-2403.16292-b31b1b.svg)](https://arxiv.org/abs/2403.16292)

This is the code for **latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction** by Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen.

Check out the [project website here](https://geometric-rl.mpi-inf.mpg.de/latentsplat/).

https://github.com/Chrixtar/latentsplat/assets/38473808/c7976773-16ad-4ef1-9526-29ae46221549

## Installation

To get started, create a conda environment:

```bash
conda env create -f environment.yml
conda activate latentsplat
```

or a virtual environment using Python 3.10+:

```bash
python3.10 -m venv latentsplat
source latentsplat/bin/activate
pip install -r requirements.txt
```

Please not that for training you need to download the pre-trained checkpoints of the VAE-GAN from LDM as explained in section [Acquiring Pre-trained Checkpoints](#acquiring-pre-trained-checkpoints).
<details>
<summary>Troubleshooting</summary>
<br>

If you face unrealistic CUDA out of memory issues (probably because of different GPU architectures during kernel compilation and training), try deinstalling the rasterizer and installing it with specified architectures:
```bash
pip uninstall diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" pip install git+https://github.com/Chrixtar/latent-gaussian-rasterization
```
</details>

## Acquiring Datasets
Please move all dataset directories into a newly created `datasets` folder in the project root directory or modify the root path as part of the dataset config files in `config/dataset`.

### RealEstate10k
For experiments on RealEstate10k, we use the same dataset version and preprocessing into chunks as pixelSplat. Please refer to their codebase [here](https://github.com/dcharatan/pixelsplat#acquiring-datasets) for information about how to obtain the data.

### CO3D
Simply download the hydrant and teddybear categories of CO3D from [here](https://ai.meta.com/datasets/co3d-downloads/), extract them into the created `datasets` folder (see above), and rename them to `hydrant` or `teddybear`, respectively.

## Acquiring Pre-trained Checkpoints
We provide two sets of checkpoints as part of our releases [here](https://github.com/Chrixtar/latentsplat/releases):
1. Pre-trained autoencoders and discriminators from [LDM](https://github.com/CompVis/latent-diffusion) adapted for finetuning within latentSplat. They serve as a starting point for latentSplat training. Please download the [pretrained.zip] and extract it in the project root directory for training from scratch.

2. Trained versions of latentSplat for RealEstate10k and CO3D hydrants and teddybears.

## Running the Code

### Training

The main entry point is `src/main.py`. Call it via:

```bash
python3 -m src.main +experiment=co3d_hydrant
```

This configuration requires a GPU with at least 40 GB of VRAM. To reduce memory usage, you can change the batch size as follows:

```bash
python3 -m src.main +experiment=co3d_hydrant data_loader.train.batch_size=1
```

Our code supports multi-GPU training. The above batch size is the per-GPU batch size.

### Evaluation

To render frames from an existing checkpoint, run the following:

```bash
python3 -m src.main +experiment=co3d_hydrant mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index/co3d_hydrant_extra.json checkpointing.load=checkpoints/co3d_hydrant.ckpt
```

### Ablations

You can run the ablations from the paper by using the corresponding experiment configurations. For example, to ablate the deterministic version:

```bash
python3 -m src.main +experiment=co3d_hydrant_det
```

## Camera Conventions

Our extrinsics are OpenCV-style camera-to-world matrices. This means that +Z is the camera look vector, +X is the camera right vector, and -Y is the camera up vector. Our intrinsics are normalized, meaning that the first row is divided by image width, and the second row is divided by image height.

## Figure Generation Code

We've included the scripts that generate tables and figures in the paper. Note that since these are one-offs, they might have to be modified to be run.

## BibTeX

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{wewer24latentsplat,
    title     = {latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction},
    author    = {Wewer, Christopher and Raj, Kevin and Ilg, Eddy and Schiele, Bernt and Lenssen, Jan Eric},
    booktitle = {arXiv},
    year      = {2024},
}</code></pre>
  </div>
</section>

## Acknowledgements

This project was partially funded by the Saarland/Intel Joint Program on the Future of Graphics and Media. We also thank David Charatan and co-authors for the great pixelSplat codebase, on which our implementation is based on.
