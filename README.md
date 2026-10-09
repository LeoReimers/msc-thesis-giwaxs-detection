# GIWAXS peak detection with DINO-Swin

Code for the Master's thesis *Transformer-based object detection for X-ray scattering data* (Leo Reimers, University of Tübingen, 2026).

This repository is a modified version of [DINO](https://github.com/IDEA-Research/DINO) (DETR with Improved deNoising anchOr boxes) by IDEA Research. It builds on the GIWAXS adaptation of DINO by Völter et al. and detects Bragg peaks in GIWAXS images in polar representation. The model uses a Swin-L backbone. It is trained on synthetic GIWAXS images and evaluated on real, manually annotated images.

For everything that is not specific to this thesis (model architecture, installation, general usage), please refer to the original [DINO repository](https://github.com/IDEA-Research/DINO) and paper.

## What this thesis adds

- Correction of the cyclic shift in Shifted-Window Attention for asymmetric windows (separate `shift_size_h` and `shift_size_w`).
- Systematic comparison of rectangular attention windows in the Swin backbone (`window_size_h`, `window_size_w`).
- Adaptive Patch Merging (APM): optional asymmetric downsampling between Swin stages.
- Training options: learning-rate schedule with warm-up (`lr_drop_list`, `lr_gammas`), EMA of the model weights, frozen backbone stages (`frozen_stages`), and a separate backbone learning rate.

The parameter sets of all runs are listed in Appendix A of the thesis.

## Recommended configuration

| Parameter | Value |
|---|---|
| Window size (height × width) | 32 × 8 |
| Epochs | 350 |
| Batch size | 2 |
| Initial learning rate | 1.5e-5 (also for the backbone) |
| `lr_drop_list` | [5, 10, 15, 300] |
| `lr_gammas` | [1.2, 1.3, 1.4, 0.1] |
| EMA | `ema_decay` = 0.9997 |
| `frozen_stages` | 2 |
| APM | off |

## Installation

Same as DINO (tested with Python 3.7.3, PyTorch 1.9.0, CUDA 11.1):

```bash
git clone https://github.com/LeoReimers/msc-thesis-giwaxs-detection.git
cd msc-thesis-giwaxs-detection
pip install -r requirements.txt

# compile CUDA operators
cd models/dino/ops
python setup.py build install
python test.py
cd ../../..
```

Download the Swin-L backbone checkpoint [swin_large_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth).

## Training

Training images are simulated on the fly by `simulation.py`, so no training dataset is needed. The model is evaluated on the real annotated images after every epoch (metric AP_total, see Section 3.7 of the thesis).

On the Slurm cluster, a run is started with `scripts/submit_dino.sh`. It runs `scripts/start_dino.sh`, which starts `main.py` with one process on one GPU:

```bash
export JOB_NAME=r1                 # job name
export CFG_OPTS='epochs=350'       # config overrides (key=value)
export PY_ARGS=''                  # additional arguments for main.py
$WORK/DINO/scripts/submit_dino.sh $WORK/DINO/output/my_run 32 8
```

Arguments: output folder, window height, window width. Without a window size, the default from the config is used. The scripts assume the code is located in `$WORK/DINO`; adjust the paths for other systems.

Each run saves its full configuration (`config_cfg.py`, `config_args_all.json`) in its output folder. The settings of all runs in the thesis are listed in Appendix A.

## Based on

- DINO: H. Zhang et al., *DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection*, ICLR 2023. https://github.com/IDEA-Research/DINO
- Swin Transformer: Z. Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021.
- GIWAXS adaptation and simulation: C. Völter et al., *Automated Peak Detection in Grazing-Incidence Diffraction Using Transformers*, Mach. Learn.: Sci. Technol. 7, 045021 (2026). https://doi.org/10.1088/2632-2153/ae7ec8

If you use the DINO model, please cite the original paper:

```bibtex
@misc{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
  year={2022},
  eprint={2203.03605},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

Apache 2.0, as for the original DINO code (Copyright (c) IDEA). See [LICENSE](LICENSE).
