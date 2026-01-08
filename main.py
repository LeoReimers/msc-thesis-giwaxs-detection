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
from util.configuration import Config
from util.evaluation import Evaluator, get_full_conf_results, recall_precision_curve_with_intensities
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test
from simulation import FastSimulation
import pickle
from torch import Tensor
from torchvision.utils import save_image
import torch.multiprocessing as mp
import torchvision
from torchvision.utils import save_image
from torchvision.ops import nms

import signal

_GOT_SIGUSR1 = False
def _on_sigusr1(signum, frame):
    global _GOT_SIGUSR1
    _GOT_SIGUSR1 = True


def filter_non_elong(pred_boxes):
    y_extent = pred_boxes[:,3] - pred_boxes[:,1]
    x_extent = pred_boxes[:,2] - pred_boxes[:,0]
    keep = x_extent*1.15 < y_extent
    return keep

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class SimulationDataset(torch.utils.data.Dataset):

    def __init__(self, transforms=None, device='cuda'):
        self.device = device
        self.transforms = transforms
        self.simulation = FastSimulation(device=self.device)
       

    def __getitem__(self, idx):
        image = None
        while image is None:
            try:
                image, boxes, mask = self.simulation.simulate_img()
            except:
                pass 

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
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--frozen_weights', default=None, type=str)
    parser.add_argument('--masks', action='store_true')

    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
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
    # RNG-States mitsichern (wichtig für stabilen Restart)
    import random, numpy as np, torch
    w['py_rng_state']    = random.getstate()
    w['np_rng_state']    = np.random.get_state()
    w['torch_rng_state'] = torch.get_rng_state()
    if torch.cuda.is_available():
        w['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return w

def _save_ckpt(output_dir, weights, extra_tag=None):
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
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')
    signal.signal(signal.SIGUSR1, _on_sigusr1)


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

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
    
    if getattr(args, "multi_step_lr", False):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_drop_list
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # global_step für Logging/Resuming
    global_step = 0

    # Falls im output_dir schon ein Checkpoint liegt: standardmäßig resumin
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
            if lr_scheduler is not None and checkpoint.get('lr_scheduler') is not None:
                try:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    logger.info("LR scheduler state loaded from checkpoint.")
                except Exception as e:
                    logger.info(f"Could not load LR scheduler state: {e}")
            args.start_epoch = checkpoint['epoch'] + 1

            # Scheduler-Epoch synchronisieren
            try:
                expected_last_epoch = args.start_epoch - 1
                if hasattr(lr_scheduler, "last_epoch") and lr_scheduler.last_epoch != expected_last_epoch:
                    logger.info(f"[LR SYNC] correcting scheduler.last_epoch "
                                f"{lr_scheduler.last_epoch} -> {expected_last_epoch}")
                    lr_scheduler.last_epoch = expected_last_epoch
            except Exception as e:
                logger.info(f"[LR SYNC] skipped ({e})")

        # RNG-Zustände wiederherstellen
        import random as _rnd, numpy as _np, torch as _th
        if 'py_rng_state' in checkpoint:    _rnd.setstate(checkpoint['py_rng_state'])
        if 'np_rng_state' in checkpoint:    _np.random.set_state(checkpoint['np_rng_state'])
        if 'torch_rng_state' in checkpoint: _th.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state_all' in checkpoint and _th.cuda.is_available():
            _th.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])

        # global_step aus Checkpoint lesen (falls vorhanden)
        try:
            if isinstance(checkpoint, dict):
                global_step = int(checkpoint.get('global_step', 0))
            else:
                global_step = 0
        except Exception:
            global_step = 0
        logger.info(f"[resume] restored global_step={global_step}")


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

    with open(output_dir / 'settings.txt', 'a+') as f:
            f.write('\n' + str(model))
            f.write('\n' + str(args))

    print("Start training")
    start_time = time.time()

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'

    for epoch in range(args.start_epoch, args.epochs):
        dataset = SimulationDataset()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        epoch_start_time = time.time()
        logger.info(f"[epoch {epoch}] start")

        # Zeitbasierte / Signal-basierte Mid-Epoch-Checkpoints
        save_every_sec = int(getattr(args, 'save_every_minutes', 0)) * 60 if hasattr(args, 'save_every_minutes') else 0
        _next_time_save = [time.time() + save_every_sec]

        def _should_save_now():
            if _GOT_SIGUSR1:
                return True
            if save_every_sec > 0 and time.time() >= _next_time_save[0]:
                _next_time_save[0] = time.time() + save_every_sec
                return True
            return False

        def _do_save_mid_epoch(tag):
            scheduler_to_save = lr_scheduler
            weights = _make_weights(
                model_without_ddp, optimizer, scheduler_to_save, epoch, args,
                train_stats={'global_step': global_step}
            )
            if args.use_ema:
                weights['ema_model'] = ema_m.module.state_dict()
            _save_ckpt(output_dir, weights, extra_tag=tag)
            global _GOT_SIGUSR1
            if _GOT_SIGUSR1:
                _GOT_SIGUSR1 = False

        train_stats = train_one_epoch(
            model, criterion, data_loader, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None),
            ema_m=ema_m,
            should_save_callback=_should_save_now,
            save_callback=_do_save_mid_epoch,
            global_step=global_step,
        )
        # global_step aus train_stats übernehmen
        global_step = int(train_stats.get('global_step', global_step))

        # Logs
        with open(output_dir / 'training_stats.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + str(train_stats) + "\n")
        with open(output_dir / 'bbox_loss.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_bbox: ' + str(train_stats['loss_bbox']) + "\n")
        with open(output_dir / 'loss_giou.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_giou: ' + str(train_stats['loss_giou']) + "\n")

        # LR-Scheduler pro Epoche updaten (wie vorher)
        if lr_scheduler is not None:
            lr_scheduler.step()


        # Am Ende der Epoche Haupt-Checkpoint speichern
        if args.output_dir:
            scheduler_to_save = lr_scheduler
            weights = _make_weights(
                model_without_ddp, optimizer, scheduler_to_save, epoch, args,
                train_stats={'global_step': global_step}
            )
            if args.use_ema:
                weights['ema_model'] = ema_m.module.state_dict()
            _save_ckpt(output_dir, weights)

            # optional: Zusatz-Checkpoints alle N Epochen
            save_every = getattr(args, 'save_checkpoint_interval', None)
            if save_every and (epoch + 1) % save_every == 0:
                _save_ckpt(output_dir, weights, extra_tag=f"e{epoch:04}")

        # --- Dein GIWAXS-Eval-Block bleibt UNVERÄNDERT dahinter ---
        class ImageProcessing():

            def __init__(self, model, postprocessors) -> None:
                self.model = model
                self.postprocessors = postprocessors

            def infer(self, img: np.array, k: int = None) -> bool:
                img = Tensor(img).cuda()
                raw_results = self.model(img)
                postprocessed =  self.postprocessors['bbox'](raw_results, torch.Tensor([[512, 512]]).cuda())
                scores = postprocessed[0]['scores']
                boxes = postprocessed[0]['boxes']
                return boxes.cpu(), scores.cpu()
            
        img_process = ImageProcessing(model, postprocessors)

        def eval_ap_func(dset_path, epoch, output_dir):
            config = Config()
            config.EVAL_EPOCH = str(epoch)
            config.EVAL_OUTPUT_FOLDER = str(output_dir)
            config.INPUT_DATASET = dset_path
            config.PREPROCESSING_POLAR_SHAPE = [512,1024]
            config.PREPROCESSING_LINEAR_CONTRAST = True
            config.PREPROCESSING_LINEAR_PERC_977 = False
            data = H5GIWAXSDataset(config, path = dset_path, preprocess_func=standard_preprocessing , buffer_size=5)   
            evaluator = Evaluator()

            for i, giwaxs_img_container in enumerate(data.iter_images()):

                giwaxs_img = giwaxs_img_container.converted_polar_image
                giwaxs_img = torch.tensor(giwaxs_img[:,0,:,:]).unsqueeze(0).cuda().repeat(1,3,1,1)
                raw_giwaxs_img = giwaxs_img_container.raw_polar_image
                labels = giwaxs_img_container.polar_labels
                outputs = model(giwaxs_img)

                postprocessed =  postprocessors['bbox'](outputs, torch.Tensor([[512, 1024]]).cuda())
                
                scores = postprocessed[0]['scores']
                pred_boxes = postprocessed[0]['boxes']

                idx_keep = nms(pred_boxes, scores, 0.4)
                pred_boxes = pred_boxes[idx_keep]
                scores = scores[idx_keep]

                idx_elong = filter_non_elong(pred_boxes)
                scores = scores[idx_elong]
                pred_boxes = pred_boxes[idx_elong]

                evaluator.get_exp_metrics(pred_boxes, scores, torch.tensor(labels.boxes, device = 'cuda'), labels.confidences)
            
            recalls, precisions, accuracies, scores, av_precision, recalls_levels, fp_nums = recall_precision_curve_with_intensities(evaluator.metrics)
            df1, df2 = get_full_conf_results(evaluator.metrics)
            print(df1)
            print(df2)
            return df2['ap_total'].values[0]

        try:
            dset_path = Path("/mnt/lustre/work/schreiber/szb559/DINO/datasets/41.h5")
            if dset_path.is_file():
                model.eval()
                eval_ap = eval_ap_func(str(dset_path), epoch, output_dir)
                with open(output_dir / 'exp_ap_40_polar.txt', 'a+') as f:
                    f.write(f"{eval_ap}\n")
            else:
                # Falls der Pfad doch mal nicht stimmt, explizite Meldung:
                with open(output_dir / 'eval_error.txt', 'a+') as f:
                    f.write(f"epoch {epoch}: dataset not found at {dset_path}\n")
        except Exception as e:
            import traceback
            with open(output_dir / 'eval_error.txt', 'a+') as f:
                f.write(f"epoch {epoch}: {repr(e)}\n{traceback.format_exc()}\n")



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
