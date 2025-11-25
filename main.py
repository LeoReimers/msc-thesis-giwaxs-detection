#!/bin/bash
# File: $WORK/DINO/scripts/start_dino.sh
# Robust start: setzt Env, findet letzten Checkpoint, startet Training.
set -euo pipefail

# --------- PARAMS (kommen von au�en per ENV) ----------
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/DINO/output/run_$(date +%Y%m%d-%H%M%S)}"
CONFIG_FILE="${CONFIG_FILE:-config/DINO/DINO_4scale_swin.py}"
WINDOW_H="${WINDOW_H:-4}"
WINDOW_W="${WINDOW_W:-4}"
BATCH_SIZE_ENV="${BATCH_SIZE_ENV:-}"

echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Logs ohne Buffer (schneller im File)
export PYTHONUNBUFFERED=1

# ---- Umgebung setzen ----
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/miniconda3/envs/dino"

export CUDA_VER="${CUDA_VER:-11.8}"
export CUDA_HOME="/usr/local/cuda-${CUDA_VER}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

# Pythonpfade f�r DINO (ops + repo-root)
export PYTHONPATH="${PYTHONPATH:-}:$WORK/DINO/models/dino/ops"
export PYTHONPATH="$WORK/DINO:${PYTHONPATH}"

cd "$WORK/DINO"

# ---- Resume-Logik ----
RESUME_ARG=()
if [ -f "${OUTPUT_DIR}/checkpoint.pth" ]; then
  echo "[INFO] Found checkpoint: ${OUTPUT_DIR}/checkpoint.pth"
  RESUME_ARG=(--resume "${OUTPUT_DIR}/checkpoint.pth")
elif [ -n "${INITIAL_CHECKPOINT:-}" ] && [ -f "${INITIAL_CHECKPOINT}" ]; then
  echo "[INFO] Using initial checkpoint: ${INITIAL_CHECKPOINT}"
  RESUME_ARG=(--resume "${INITIAL_CHECKPOINT}")
else
  echo "[INFO] No checkpoint found; starting fresh."
fi

if [ -n "${BATCH_SIZE_ENV}" ]; then
  echo "[INFO] Using BATCH_SIZE_ENV=${BATCH_SIZE_ENV}"
fi

# ---- Start: Single-GPU Distributed (standalone) ----
# ---- Start: Single-GPU Distributed (standalone) ----
LR_T0="${LR_T0:-10}"
LR_TMULT="${LR_TMULT:-2}"
LR_MIN="${LR_MIN:-1e-7}"
SEED="${SEED:-42}"
: "${FLATCOS:=1}"
: "${LR_WARMUP_EPOCHS:=3}"
: "${LR_HOLD_EPOCHS:=97}"
: "${LR_COSINE_EPOCHS:=40}"
: "${LR_WARMUP_START_FACTOR:=0.3}"
: "${CFG_OPTS:=}"
: "${PY_ARGS:=}"


# Options, die in die Config gemergt werden (als Array, damit sauber gequotet)
OPTS=()
if [ -n "${EPOCHS:-}" ]; then
  OPTS+=("epochs=${EPOCHS}")
fi
if [ -n "${BATCH_SIZE_ENV:-}" ]; then
  OPTS+=("batch_size=${BATCH_SIZE_ENV}")
fi

echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
echo "[INFO] Using Flat→Hold→Cosine schedule (warmup=${LR_WARMUP_EPOCHS}, hold=${LR_HOLD_EPOCHS}, cosine=${LR_COSINE_EPOCHS}, lr_min=${LR_MIN}, seed=${SEED})"

: "${CFG_OPTS:=}"   # key=val Paare für --options
: "${PY_ARGS:=}"    # zusätzliche Python-Flags (--lr_mode, --flatcos, ...)

SCHED_OPTS=()
if [ "$FLATCOS" = "1" ]; then
  SCHED_OPTS=( --flatcos
               --lr_warmup_epochs "$LR_WARMUP_EPOCHS"
               --lr_hold_epochs   "$LR_HOLD_EPOCHS"
               --lr_cosine_epochs "$LR_COSINE_EPOCHS"
               --lr_warmup_start_factor "$LR_WARMUP_START_FACTOR" )
fi

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --standalone \
  main.py \
    --options \
      "window_size_h=${WINDOW_H}" \
      "window_size_w=${WINDOW_W}" \
      ${CFG_OPTS} \
      "${OPTS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda \
    --seed "${SEED}" \
    "${RESUME_ARG[@]}" \
    ${PY_ARGS}

und das die lr drop logik vereinfacht wurde?:

# -*- coding: utf-8 -*-
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from util.evaluation import Evaluator, get_full_conf_results, recall_precision_curve_with_intensities
from util.exp_preprocess import standard_preprocessing
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test
from simulation import FastSimulation
import pickle
from torch import Tensor
from torchvision.utils import save_image
import torch.multiprocessing as mp
from gixdinference.evaluation import eval_on_dataset
from gixdinference.configuration import Config
from gixdinference.dataloader import H5GIWAXSDataset

import math
import signal
from typing import Optional, Callable, Iterable, Dict, Any

import torchvision
from torchvision.utils import save_image
from torchvision.ops import nms
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    MultiStepLR,
    StepLR,
    CosineAnnealingWarmRestarts,
)

_GOT_SIGUSR1 = False
def _on_sigusr1(signum, frame):
    global _GOT_SIGUSR1
    _GOT_SIGUSR1 = True


def filter_non_elong(pred_boxes):
    y_extent = pred_boxes[:, 3] - pred_boxes[:, 1]
    x_extent = pred_boxes[:, 2] - pred_boxes[:, 0]
    keep = x_extent * 1.15 < y_extent
    return keep


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class SimulationDataset(torch.utils.data.Dataset):

    def __init__(self, transforms = None, device = 'cuda'):

        self.device = 'cuda'
        self.transforms = transforms
        self.simulation = FastSimulation(device=self.device)        

    def __getitem__(self, idx):
        slice_into_four = True
        image = None
        while image is None:
            try:
                image, boxes, mask = self.simulation.simulate_img()
            except Exception:
                # Wenn simulate_img() fehlschlägt, erneut versuchen
                image = None
                continue

        # Falls das Bild nur einen Kanal hat, für 3-Kanal-Modelle aufduplizieren
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        num_objects = len(boxes[0:])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        h, w = image.shape[-2:]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], device=self.device)        
        target = {"boxes": boxes}
        target['area'] = area

        target["labels"] = torch.ones((num_objects,), dtype=torch.int64, device=self.device)
        target["image_id"] = torch.tensor(idx, device=self.device)
        target["iscrowd"] = torch.zeros((num_objects,), dtype=torch.int64, device=self.device)
        target["orig_size"] = torch.tensor(image[0].shape, device=self.device)
        target["size"] = torch.tensor(image[0].shape, device=self.device)

        return image, target

    def __len__(self):
        #number of images in epoch
        return 1000

def collate_fn(batch):
    # Initialize lists to hold the tensors
    samples = []
    targets = []

    # Iterate over the batch
    for item in batch:
        # Unpack the item
        image_tensor, target_dict = item

        # Append the image tensor to the samples list
        samples.append(image_tensor)

        # Append the target dict to the targets list
        targets.append(target_dict)

    # Convert the lists to tensors
    samples = torch.stack(samples)
    #targets = [{'boxes': torch.stack([t['boxes'] for t in targets])}]
    # Return a dictionary
    #return {'samples': samples, 'targets': targets}
    return samples, targets



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', default=os.path.dirname(os.path.realpath(__file__)) + '/config/DINO/DINO_4scale_swin.py', type=str, required=False)

    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    
    parser.add_argument('--save_every_minutes', type=int, default=0,
                    help='Zeitbasierter Mid-Epoch-Checkpoint (0=aus).')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--window_size_h", type=int, default=None)
    parser.add_argument("--window_size_w", type=int, default=None)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    parser.add_argument('--lr_mode', choices=['epoch', 'plateau', 'metric'], default='epoch',
                    help='epoch: Step/MultiStep wie bisher; plateau: ReduceLROnPlateau; metric: fester Schwellenwert')
    parser.add_argument('--lr_plateau_patience', type=int, default=3,
                    help='Wieviele Epochen ohne Verbesserung, bevor LR gedroppt wird')
    parser.add_argument('--lr_plateau_factor', type=float, default=0.1,
                    help='Multiplikator beim Drop (z.B. 0.1 -> LR * 0.1)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                    help='Untergrenze für LR')
    parser.add_argument('--lr_cooldown', type=int, default=0,
                    help='Cooldown-Epochen nach einem Drop')
    parser.add_argument('--eval_every', type=int, default=1,
                    help='Wie oft validieren (Plateau braucht Val-Metrik pro Epoche)')
    parser.add_argument('--lr_T0', type=int, default=50,
                    help='CosineWarmRestarts: Schritte bis zum ersten Restart (Epochen)')
    parser.add_argument('--lr_Tmult', type=int, default=2,
                    help='CosineWarmRestarts: Multiplikator für Folge-Zyklen')
    parser.add_argument('--flatcos', action='store_true',
    help='Warmup -> Hold -> Cosine (ohne Restarts)')
    parser.add_argument('--lr_warmup_epochs', type=int, default=3)
    parser.add_argument('--lr_hold_epochs',   type=int, default=90)
    parser.add_argument('--lr_cosine_epochs', type=int, default=37)  # Summe = 130
    parser.add_argument('--lr_warmup_start_factor', type=float, default=0.3)
 
    return parser


def merge_cfg_into_args(args, cfg):
    """Übernimm Werte aus der Config in argparse-Args,
    ohne bereits gesetzte CLI-Args zu überschreiben."""
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    return args


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def _make_weights(model_without_ddp, optimizer, scheduler_to_save, epoch, args, train_stats=None):
    w = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': (scheduler_to_save.state_dict() if scheduler_to_save is not None else None),
        'epoch': epoch,
        'args': args,
    }
    if train_stats is not None and 'global_step' in train_stats:
        w['global_step'] = int(train_stats['global_step'])
    # --- RNG-States mitsichern (billig, hilft bei Repro) ---
    import random, numpy as np, torch
    w['py_rng_state']    = random.getstate()
    w['np_rng_state']    = np.random.get_state()
    w['torch_rng_state'] = torch.get_rng_state()
    if torch.cuda.is_available():
        w['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return w

def _save_ckpt(output_dir, weights, extra_tag=None):
    # immer 'checkpoint.pth' + optionaler Extra-Tag
    from pathlib import Path
    output_dir = Path(output_dir)
    paths = [output_dir / 'checkpoint.pth']
    if extra_tag is not None:
        paths.append(output_dir / f'checkpoint_{extra_tag}.pth')
    for p in paths:
        utils.save_on_master(weights, p)


def main(args):
    #utils.init_distributed_mode(args)
    dataset = SimulationDataset()
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)

    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.rank == 0:
        cfg.dump(os.path.join(args.output_dir, "config_cfg.py"))
        with open(os.path.join(args.output_dir, "config_args_raw.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # EINMAL zusammenführen (CLI > Config):
    args = merge_cfg_into_args(args, cfg)

    # Defaults nachziehen, falls in Config nicht gesetzt
    if not getattr(args, 'use_ema', False):
        args.use_ema = False
    if not getattr(args, 'debug', False):
        args.debug = False

    # Logger
    logger = setup_logger(
        output=os.path.join(args.output_dir, 'info.txt'),
        distributed_rank=args.rank, color=False, name="detr")
    signal.signal(signal.SIGUSR1, _on_sigusr1)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        with open(os.path.join(args.output_dir, "config_args_all.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(
            os.path.join(args.output_dir, "config_args_all.json")))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training ist nur für Segmentation gedacht"

    print(args)

    global_step = 0

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

        # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # ------------------------------------------------------------
    # Lernraten-Scheduler (vereinfachte Variante)
    # ------------------------------------------------------------
    #
    # Annahmen:
    #  - wir unterstützen hier nur lr_mode == 'epoch'
    #  - Multi-Step-Schedule wird über
    #       multi_step_lr, lr_drop_list, lr_gammas
    #    aus der Config gesteuert
    #  - keine Plateau- oder Cosine-Scheduler mehr
    #
    lr_scheduler = None
    plateau_sch = None  # nur für Kompatibilität mit Resume-Code

    if args.lr_mode != 'epoch':
        raise ValueError(
            f"Dieses vereinfachte Setup unterstützt nur lr_mode='epoch', "
            f"nicht lr_mode={args.lr_mode!r}."
        )

    # Sicherheitscheck: nicht beide Modi gleichzeitig
    if getattr(args, "onecyclelr", False) and getattr(args, "multi_step_lr", False):
        raise ValueError("onecyclelr und multi_step_lr dürfen nicht gleichzeitig True sein.")

    if getattr(args, "multi_step_lr", False):
        # Multi-Step-Schedule über lr_drop_list + (optional) lr_gammas
        drops = list(getattr(args, "lr_drop_list", []))
        gammas = list(getattr(args, "lr_gammas", [])) if hasattr(args, "lr_gammas") else None

        if drops and gammas:
            if len(gammas) != len(drops):
                raise ValueError("lr_gammas muss gleich lang sein wie lr_drop_list.")

            # Wir verwenden MultiplicativeLR und wenden Gamma NUR in der Drop-Epoche an.
            drop_map = {int(t): float(g) for t, g in zip(drops, gammas)}

            def oneoff(epoch: int):
                # epoch entspricht intern last_epoch; für alle anderen Epochen Faktor 1.0
                return drop_map.get(int(epoch), 1.0)

            lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, lr_lambda=oneoff
            )

        elif drops:
            # klassischer MultiStepLR mit einem gemeinsamen Gamma
            gamma = float(getattr(args, "lr_gamma", 0.1))
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=drops, gamma=gamma
            )

        else:
            # Fallback: einfacher StepLR mit lr_drop
            step_size = int(getattr(args, "lr_drop", args.epochs))
            gamma = float(getattr(args, "lr_gamma", 0.1))
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

    else:
        # Kein LR-Scheduling: konstante Lernrate über alle Epochen
        lr_scheduler = None

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # richtigen Scheduler-Handler wählen
            scheduler_to_load = plateau_sch if plateau_sch is not None else lr_scheduler
            if scheduler_to_load is not None and checkpoint.get('lr_scheduler') is not None:
                try:
                    scheduler_to_load.load_state_dict(checkpoint['lr_scheduler'])
                    logger.info("LR scheduler state loaded from checkpoint.")
                except Exception as e:
                    logger.info(f"Could not load LR scheduler state: {e}")
            args.start_epoch = checkpoint['epoch'] + 1
            try:
                expected_last_epoch = args.start_epoch - 1
                sched = plateau_sch if plateau_sch is not None else lr_scheduler
                if hasattr(sched, "last_epoch") and sched.last_epoch != expected_last_epoch:
                    logger.info(f"[LR SYNC] correcting scheduler.last_epoch {sched.last_epoch} -> {expected_last_epoch}")
                    sched.last_epoch = expected_last_epoch
            except Exception as e:
                logger.info(f"[LR SYNC] skipped ({e})")

        # --- RNG wiederherstellen (falls vorhanden) ---
        import random as _rnd, numpy as _np, torch as _th
        if 'py_rng_state' in checkpoint:    _rnd.setstate(checkpoint['py_rng_state'])
        if 'np_rng_state' in checkpoint:    _np.random.set_state(checkpoint['np_rng_state'])
        if 'torch_rng_state' in checkpoint: _th.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state_all' in checkpoint and _th.cuda.is_available():
            _th.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        # --- global_step aus Checkpoint übernehmen, falls vorhanden ---
        try:
            if isinstance(checkpoint, dict):
                global_step = int(checkpoint.get('global_step', 0))
            else:
                global_step = 0
        except Exception:
            global_step = 0
        try:
            logger.info(f"[resume] restored global_step={global_step}")
        except Exception:
            pass

 


    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        

    shutil.copy(os.path.dirname(os.path.realpath(__file__)) + '/simulation.py', output_dir / 'simulation.py')
                
    with open('/mnt/lustre/work/schreiber/szb559/DINO/datasets/40_labeled.pkl', 'rb') as pickle_file:
        eval_dataset_polar = pickle.load(pickle_file)
    with open('/mnt/lustre/work/schreiber/szb559/DINO/datasets/40_labeled_1channel_quazi.pkl', 'rb') as pickle_file:
        eval_dataset_quazi = pickle.load(pickle_file)

    eval_recalls = []

    with open(output_dir / 'settings.txt', 'a+') as f:
            f.write('\n' + str(model))
            f.write('\n' + str(args))

    def _get_lr(opt):
        return opt.param_groups[0]['lr']

    shutil.copy(os.path.dirname(os.path.realpath(__file__)) + '/simulation.py', output_dir / 'simulation.py')

    with open(output_dir / 'settings.txt', 'a+') as f:
            f.write('\n' + str(model))
            f.write('\n' + str(args))

    print("Start training")
    start_time = time.time()

    #best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    if args.evaluate:
        args.start_epoch = 1
    for epoch in range(args.start_epoch, args.epochs):
        dataset = SimulationDataset()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        epoch_start_time = time.time()
        logger.info(f"[epoch {epoch}] start lr={_get_lr(optimizer):.3e}")

        # Zeitbasiertes Fallback-Checkpointing (optional)
        save_every_sec = int(getattr(args, 'save_every_minutes', 0)) * 60 if hasattr(args, 'save_every_minutes') else 0
        _next_time_save = [time.time() + save_every_sec]  # mutable via closure

        def _should_save_now():
            # 1) Preemption-Signal?
            if _GOT_SIGUSR1:
                return True
            # 2) Zeitbasierter Save?
            if save_every_sec > 0 and time.time() >= _next_time_save[0]:
                _next_time_save[0] = time.time() + save_every_sec
                return True
            return False

        def _do_save_mid_epoch(tag):
            scheduler_to_save = plateau_sch if plateau_sch is not None else lr_scheduler
            weights = _make_weights(
                model_without_ddp, optimizer, scheduler_to_save, epoch, args,
                train_stats={'global_step': global_step}
            )
            _save_ckpt(output_dir, weights, extra_tag=tag)

            # Debounce: SIGUSR1-Flag zurücksetzen, damit nicht endlos gespeichert wird
            global _GOT_SIGUSR1
            if _GOT_SIGUSR1:
                _GOT_SIGUSR1 = False



        train_stats = train_one_epoch(
            model, criterion, data_loader, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None),
            ema_m=ema_m,
            should_save_callback=_should_save_now,
            save_callback=_do_save_mid_epoch,
            global_step=global_step,
        )
        global_step = int(train_stats.get('global_step', global_step))
       
        # eval
        with open(output_dir / 'training_stats.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + str(train_stats) + "\n")
        with open(output_dir  / 'bbox_loss.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_bbox: ' + str(train_stats['loss_bbox']) + "\n")

        evaluate(
            model, criterion, postprocessors, data_loader, dataset, device, args.output_dir, epoch,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        
        with open(output_dir  / 'loss_giou.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_giou: ' + str(train_stats['loss_giou']) + "\n")

        class ImageProcessing():

            def __init__(self, model, postprocessors) -> None:
                self.model = model
                self.postprocessors = postprocessors

            def infer(self, img: np.array, k: int = None) -> bool:
                img = Tensor(img).cuda()

                raw_results = self.model(img)
                postprocessed =  self.postprocessors['bbox'](raw_results, torch.Tensor([[512, 1024]]).cuda())
                scores = postprocessed[0]['scores']
                boxes = postprocessed[0]['boxes']
                return boxes.cpu(), scores.cpu()
            
        img_process = ImageProcessing(model, postprocessors)


        config = Config()
        config.EVAL_EPOCH = str(epoch)
        config.EVAL_OUTPUT_FOLDER = str(output_dir)
        if args.evaluate:
            eval_recall = eval_on_dataset(config, None, img_process, eval_dataset_quazi, str(output_dir))
            sys.exit()
        print('quazipolar')
        quazi_recall = eval_on_dataset(config, None, img_process, eval_dataset_quazi)
        print('polar')
        polar_recall = eval_on_dataset(config, None, img_process, eval_dataset_polar)

        # Val-Metrik bestimmen (max oder mean; hier max)
        val_metric = float(max(quazi_recall, polar_recall))
        logger.info(f"[epoch {epoch}] val_quazi={float(quazi_recall):.4f} | val_polar={float(polar_recall):.4f} | chosen_val_metric={val_metric:.4f}")

        if lr_scheduler is not None:
            before = _get_lr(optimizer)
            lr_scheduler.step()
            after = _get_lr(optimizer)
            if after < before - 1e-15:
                logger.info(f"[LR DROP][epoch {epoch}] mode=epoch | lr: {before:.3e} -> {after:.3e}")
            else:
                logger.info(f"[LR STEP][epoch {epoch}] lr: {after:.3e}")
        
        if args.output_dir:
            scheduler_to_save = plateau_sch if plateau_sch is not None else lr_scheduler
            weights = _make_weights(model_without_ddp, optimizer, scheduler_to_save, epoch, args,
                                    train_stats={'global_step': global_step})
            _save_ckpt(output_dir, weights)  # checkpoint.pth
            # optional: zusätzliche Meilensteine
            save_every = getattr(args, 'save_checkpoint_interval', None)
            if ((getattr(args, 'multi_step_lr', False) and ((epoch + 1) in getattr(args, 'lr_drop_list', [])))
                or (save_every and (epoch + 1) % save_every == 0)):
                _save_ckpt(output_dir, weights, extra_tag=f"e{epoch:04}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files. 
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Harmonisierung beider Flags
    args.evaluate = bool(getattr(args, "evaluate", False) or getattr(args, "eval", False))
    args.eval = args.evaluate

    # Wenn im angegebenen output_dir bereits ein checkpoint.pth liegt: daraus resumin
    ckpt_in_output = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.isfile(ckpt_in_output):
        args.resume = ckpt_in_output

    # Wenn resume gesetzt ist, output_dir auf den Checkpoint-Ordner setzen
    if getattr(args, "resume", None):
        resume_dir = os.path.dirname(args.resume)
        if os.path.isdir(resume_dir):
            args.output_dir = resume_dir

    # Falls kein sinnvoller output_dir gesetzt ist: automatisch einen unter deinem Lustre-Root anlegen
    if not args.output_dir:
        root = '/mnt/lustre/work/schreiber/szb559/trainingoutputs'
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(root, f'dinodetr{timestamp}')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.export = False

    print(args.output_dir)
    main(args)

    main(args)
    
    #args.resume = "/home/constantin/git_repos/object_detection/outputs/dinodetr20241025-171240/checkpoint0097.pth"
    #args.resume = '/home/constantin/git_repos/object_detection/outputs/dinodetr20241208-162841/checkpoint0030.pth'

    #args.output_dir = '/home/constantin/git_repos/object_detection/outputs/dinodetr20241025-171240/'
    #args.output_dir = '/home/constantin/git_repos/object_detection/outputs/dinodetr20241208-162841/'
    args.evaluate = False
    args.export = False
    #args.save_checkpoint_interval = 50
