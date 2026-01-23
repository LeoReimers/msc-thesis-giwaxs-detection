# Copyright (c) 2022 IDEA. All Rights Reserved.
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
from util.configuration import Config
from util.evaluation import Evaluator, get_full_conf_results
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
import util.misc as utils
import datasets
from engine import evaluate, train_one_epoch
from simulation import FastSimulation
from torch import Tensor
from torchvision.utils import save_image, draw_bounding_boxes
from torchvision.ops import nms
from util.clahe import clahe_2d_numpy
import signal

_GOT_SIGUSR1 = False
def _on_sigusr1(signum, frame):
    global _GOT_SIGUSR1
    _GOT_SIGUSR1 = True

def filter_non_elong(pred_boxes):
    # DEAKTIVIERT FÜR INITIALES TRAINING
    return torch.ones(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None, device='cuda', clahe_cfg=None):
        self.device = device
        self.transforms = transforms
        self.simulation = FastSimulation(device=self.device)
        self.clahe_cfg = clahe_cfg 

    def __getitem__(self, idx):
        image = None
        while image is None:
            try:
                image, boxes, mask = self.simulation.simulate_img()
            except:
                pass
        
        # --- CLAHE APPLICATION ---
        if self.clahe_cfg is not None and self.clahe_cfg.get("enabled", False):
            p = float(self.clahe_cfg.get("p", 1.0))
            if np.random.rand() < p:
                img_hw = image[0].detach().float().cpu().numpy()
                out = clahe_2d_numpy(
                    img_hw,
                    clip_limit=float(self.clahe_cfg.get("clip_limit", 2.0)),
                    tile_grid=tuple(self.clahe_cfg.get("tile_grid", (8, 8))),
                )
                out_t = torch.from_numpy(out).to(image.device).float()
                image = out_t.unsqueeze(0)

        image = image.repeat(3, 1, 1)
        num_objects = len(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        h, w = image.shape[-2:]
        boxes = boxes.to(image.device).float()
        boxes_c = box_xyxy_to_cxcywh(boxes)
        
        if boxes.max() > 2.0:
            denom = torch.tensor([w, h, w, h], device=image.device, dtype=boxes_c.dtype)
            boxes_c = boxes_c / denom
        boxes_c = boxes_c.clamp(0.0, 1.0)
        
        target = {
            "boxes": boxes_c,
            "area": area,
            "labels": torch.ones((num_objects,), dtype=torch.int64, device=self.device),
            "image_id": torch.tensor(idx, device=self.device),
            "iscrowd": torch.zeros((num_objects,), dtype=torch.int64, device=self.device),
            "orig_size": torch.tensor(image[0].shape, device=self.device),
            "size": torch.tensor(image[0].shape, device=self.device)
        }
        return image, target

    def __len__(self):
        return 1000

def collate_fn(batch):
    samples = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return samples, targets

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', default=os.path.dirname(os.path.realpath(__file__)) + '/config/DINO/DINO_4scale_swin.py', type=str)
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--frozen_weights', default=None, type=str)
    parser.add_argument('--masks', action='store_true')
    
    # Dummy args
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--note', default='')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    return parser

def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def _make_weights(model, optimizer, scheduler, epoch, args, global_step=0):
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'args': args,
        'global_step': global_step
    }

def _save_ckpt(output_dir, weights, extra_tag=None):
    p = Path(output_dir) / ('checkpoint.pth' if extra_tag is None else f'checkpoint_{extra_tag}.pth')
    utils.save_on_master(weights, p)

def main(args):
    utils.init_distributed_mode(args)
    
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # Config in args übernehmen
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    # CLAHE Konfiguration
    clahe_cfg = None
    if bool(getattr(args, "use_clahe", False)) and bool(getattr(args, "clahe_apply_train", True)):
        clahe_cfg = {
            "enabled": True,
            "clip_limit": float(getattr(args, "clahe_clip_limit", 2.0)),
            "tile_grid": tuple(getattr(args, "clahe_tile_grid", [8, 8])),
            "p": float(getattr(args, "clahe_prob", 1.0)), 
        }

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    if getattr(args, 'use_ema', False):
        ema_m = ModelEma(model_without_ddp, args.ema_decay)
    else:
        ema_m = None

    param_dicts = get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Resume Logic
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint.get('global_step', 0)
        if ema_m and 'ema_model' in checkpoint:
             ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))

    elif args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        _ignore = args.finetune_ignore if args.finetune_ignore else []
        _tmp = {k:v for k,v in checkpoint.items() if not any(x in k for x in _ignore)}
        model_without_ddp.load_state_dict(_tmp, strict=False)

    shutil.copy(os.path.dirname(os.path.realpath(__file__)) + '/simulation.py', Path(args.output_dir) / 'simulation.py')
    
    # Save settings.txt (wiederhergestellt)
    with open(Path(args.output_dir) / 'settings.txt', 'a+') as f:
        f.write('\n' + str(model))
        f.write('\n' + str(args))

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        dataset = SimulationDataset(device=args.device, clahe_cfg=clahe_cfg)
        
        # Batch Size wieder auf 4 festgesetzt
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        train_stats = train_one_epoch(
            model, criterion, data_loader, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=False, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), ema_m=ema_m, global_step=global_step
        )
        global_step = train_stats.get('global_step', global_step)
        
        # LOG FILES WIEDERHERGESTELLT
        with open(Path(args.output_dir) / 'training_stats.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + str(train_stats) + "\n")
        with open(Path(args.output_dir) / 'bbox_loss.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_bbox: ' + str(train_stats['loss_bbox']) + "\n")
        with open(Path(args.output_dir) / 'loss_giou.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_giou: ' + str(train_stats['loss_giou']) + "\n")

        if lr_scheduler: lr_scheduler.step()

        if args.output_dir:
            weights = _make_weights(model_without_ddp, optimizer, lr_scheduler, epoch, args, global_step)
            if ema_m: weights['ema_model'] = ema_m.module.state_dict()
            _save_ckpt(args.output_dir, weights)
            if hasattr(args, 'save_checkpoint_interval') and args.save_checkpoint_interval and (epoch+1) % args.save_checkpoint_interval == 0:
                _save_ckpt(args.output_dir, weights, extra_tag=f'e{epoch:04}')

        # --- EVALUATION ---
        def eval_ap_func(dset_path, epoch, output_dir):
            config = Config()
            config.EVAL_EPOCH = str(epoch)
            config.EVAL_OUTPUT_FOLDER = str(output_dir)
            config.INPUT_DATASET = dset_path
            config.PREPROCESSING_POLAR_SHAPE = [512,1024]
            config.PREPROCESSING_LINEAR_CONTRAST = True 
            
            # FIX: Variable hinzugefügt
            config.PREPROCESSING_LINEAR_PERC_977 = False 
            
            data = H5GIWAXSDataset(config, path=dset_path, preprocess_func=standard_preprocessing, buffer_size=5)
            evaluator = Evaluator()

            for i, container in enumerate(data.iter_images()):
                img = torch.tensor(container.converted_polar_image[:,0,:,:]).unsqueeze(0).cuda().repeat(1,3,1,1)
                labels = container.polar_labels
                
                outputs = model(img)
                post = postprocessors['bbox'](outputs, torch.Tensor([[512, 1024]]).cuda())[0]
                scores, boxes = post['scores'], post['boxes']
                
                keep = nms(boxes, scores, 0.4)
                boxes = boxes[keep]
                scores = scores[keep]
                
                # Filter (aktuell dummy)
                keep_elong = filter_non_elong(boxes)
                boxes = boxes[keep_elong]
                scores = scores[keep_elong]

                # --- VISUALISIERUNG (Erstes Bild jeder Epoche) ---
                if i == 0:
                    try:
                        # Ground Truth
                        gt_boxes = torch.tensor(labels.boxes)
                        print(f"[EVAL VIS] Epoch {epoch}: GT Boxes: {len(gt_boxes)}, Pred Boxes: {len(boxes)}")
                        
                        # Bild für Vis (uint8 0-255)
                        vis_img = (img[0].detach().cpu() * 255).clamp(0,255).byte()
                        
                        # GT in Grün
                        if len(gt_boxes) > 0:
                            vis_img = draw_bounding_boxes(vis_img, gt_boxes, colors="green", width=3)
                        
                        # Pred in Rot (Top 20)
                        if len(boxes) > 0:
                            vis_img = draw_bounding_boxes(vis_img, boxes[:20].cpu(), colors="red", width=3)
                        
                        save_image(vis_img.float()/255.0, Path(output_dir) / f"debug_eval_viz_epoch_{epoch}.png")
                        print(f"[EVAL VIS] Saved to debug_eval_viz_epoch_{epoch}.png")
                    except Exception as e:
                        print(f"[EVAL VIS ERROR] {e}")

                conf = np.asarray(labels.confidences, dtype=np.float32)
                conf_binned = np.where(conf >= 0.66, 1.0, np.where(conf >= 0.33, 0.5, 0.1)).astype(np.float32)
                
                evaluator.get_exp_metrics(boxes, scores, torch.tensor(labels.boxes, device='cuda'), conf_binned)
            
            ms = np.asarray(evaluator.metrics.matched_scores)
            fs = np.asarray(evaluator.metrics.fp_scores)
            if (ms.size + fs.size) == 0: return 0.0
            _, df_ap = get_full_conf_results(evaluator.metrics)
            return float(df_ap['ap_total'].values[0])

        dset_path = Path("/mnt/lustre/work/schreiber/szb559/DINO/datasets/41.h5")
        if dset_path.is_file():
            model.eval()
            try:
                ap = eval_ap_func(str(dset_path), epoch, args.output_dir)
                with open(Path(args.output_dir)/'exp_ap_40_polar.txt', 'a+') as f:
                    f.write(f"{ap}\n")
            except Exception as e:
                import traceback
                traceback.print_exc()
            model.train()

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir: Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
