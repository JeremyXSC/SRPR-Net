import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision.ops import nms
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import torch.nn.functional as F

import segment.val as validate
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from models.segment_anything import sam_model_registry
from models.sam_lora_mask_decoder import LoRA_Sam
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.plots import output_to_target_sam
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, one_cycle,
                           print_args, print_mutation, strip_optimizer, yaml_save, non_max_suppression)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader, LoadImagesAndLabelsAndMasks
from utils.segment.loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.segment.general import crop
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
from utils.dataloaders import InfiniteDataLoader, seed_worker

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def loss_cal(model, imgs, targets, masks, nb, ema, single_cls, sam_lora, imgsz, device):
    out, pred = model(imgs)
    label_gt = targets[:, 1].to(torch.uint8)
    lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
    nm = de_parallel(ema.ema).model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
    out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.25, labels=lb, 
                                multi_label=True, agnostic=single_cls, max_det=300, nm=nm)
    
    sam_bboxes, scores, labels = output_to_target_sam(out, max_det=300, thre=0.25)
    keep_bboxes = []
    for i, bbox in enumerate(sam_bboxes):
        if (bbox[2]-bbox[0]) < 1 or (bbox[3]-bbox[1]) < 1:
            pass
        else:
            keep_bboxes.append(i)
    sam_bboxes = sam_bboxes[keep_bboxes] 
    labels = labels[keep_bboxes]
    pred_bboxes = sam_bboxes.clone()
    pred_labels = labels.clone()

    # assert label_gt.shape[0] == (torch.unique(masks).shape[0]-1), "the masks is incompitible with the labels_gt"
    real_masks = masks.clone()
    for i in torch.unique(real_masks):
        if i == 0:
            continue
        real_masks[real_masks==i.item()] = label_gt[i.item()-1] + 1

    if sam_bboxes.shape[0] != 0:
        sam_output = sam_lora(imgs, multimask_output=False, image_size=imgsz, bbox=sam_bboxes)
        # pred_masks_logits, pred_masks = torch.max(sam_output['low_res_logits'],dim=1)
        pred_masks_logits = sam_output['low_res_logits'][:,0,:,:]        
        mask_gt = torch.cat([real_masks.to(device).float()]*sam_bboxes.shape[0], dim=0)
        mask_gt = (mask_gt == (labels[:, None, None]+1)).float()
        loss_sam = F.binary_cross_entropy_with_logits(torch.sigmoid(pred_masks_logits), mask_gt, reduction="none")
        # re-scale the sam_bboxes to fit the shape of masks
        scale_x = pred_masks_logits.shape[-2]/imgs.shape[-2]
        scale_y = pred_masks_logits.shape[-1]/imgs.shape[-1]
        sam_bboxes[:, 0], sam_bboxes[:, 2] = sam_bboxes[:, 0]*scale_x, sam_bboxes[:, 2]*scale_x
        sam_bboxes[:, 1], sam_bboxes[:, 3] = sam_bboxes[:, 1]*scale_x, sam_bboxes[:, 3]*scale_y
        loss_sam = (crop(loss_sam, sam_bboxes).mean(dim=(1, 2))).sum()
        return loss_sam, pred_bboxes, pred_masks_logits, pred
    else:
        return None, None, None, pred
            

def visualize_masks_on_image(pred_masks, image, output_path="output_image_with_masks.png"):
    """
    Visualizes predicted masks overlaid on the image and saves the result.

    Args:
        pred_masks (torch.Tensor): Tensor of shape [n, 64, 64] containing the predicted masks.
        image (torch.Tensor): Tensor of shape [1, 3, 256, 256] containing the original image.
        output_path (str): Path where the output image will be saved.
    """
    # Define colors for each mask (You can change or add more colors)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    # Convert the image tensor to PIL image (shape [1, 3, 256, 256] to [256, 256, 3])
    image_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [3, 256, 256] -> [256, 256, 3]
    image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8 for image display
    image_pil = Image.fromarray(image_np)

    # Resize the masks from [64, 64] to [256, 256]
    resized_masks = torch.nn.functional.interpolate(pred_masks.float().unsqueeze(1), 
                                                    size=(image.shape[-2], image.shape[-2]), mode="nearest").squeeze(1)

    # Convert the original image to RGBA for transparency handling
    image_pil = image_pil.convert("RGBA")
    
    # Create a blank RGBA image to hold the masks
    overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))  # Fully transparent image

    # Overlay each mask on the original image
    for i, mask in enumerate(resized_masks):
        mask_np = mask.detach().cpu().numpy().astype(np.uint8) * 255  # Binarize mask
        mask_pil = Image.fromarray(mask_np).convert("L")  # Convert mask to grayscale image
        
        # Choose a color for the current mask
        color = colors[i % len(colors)]
        colored_mask = ImageOps.colorize(mask_pil, black=(0, 0, 0), white=color)
        colored_mask.putalpha(mask_pil)  # Set the transparency based on the mask
        
        # Paste the colored mask onto the overlay
        overlay = Image.alpha_composite(overlay, colored_mask.convert("RGBA"))

    # Combine the original image with the overlay
    final_image = Image.alpha_composite(image_pil, overlay)

    # Save the final image
    final_image.save(output_path)
    

def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        logger = GenericLogger(opt=opt, console_logger=LOGGER)

    # Config
    plots = not evolve and not opt.noplots  # create plots
    overlap = not opt.no_overlap
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path, test_path = data_dict['train'], data_dict['val'], data_dict['test']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = SegmentationModel(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), opt=opt).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = SegmentationModel(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    # SAM model
    sam, img_embedding_size = sam_model_registry[opt.vit_name](image_size=opt.imgsz,
                                                               checkpoint=opt.sam_ckpt)

    sam_lora = LoRA_Sam(sam, 4).to(device)    
    
    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        logger.update_params({"batch_size": batch_size})
        # loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    optimizer_sam = torch.optim.SGD(sam_lora.parameters(), lr=hyp['lr_sam'], momentum=hyp['momentum'], nesterov=True)
    
    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler_sam = lr_scheduler.LambdaLR(optimizer_sam, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    # ema_sam = ModelEMA(sam_lora) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls, hyp=hyp,
        cache=None if opt.cache == 'val' else opt.cache, rect=opt.rect, rank=LOCAL_RANK,
        workers=workers, image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
        shuffle=True, mask_downsample_ratio=mask_ratio, overlap_mask=overlap, 
    )
    train_size = int(opt.data_rate * len(dataset))
    meta_size = len(dataset) - train_size
    train_dataset, meta_dataset = torch.utils.data.random_split(dataset, [train_size, meta_size])
    train_loader = InfiniteDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, sampler=None,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if opt.quad else LoadImagesAndLabelsAndMasks.collate_fn,
        pin_memory=True, worker_init_fn=seed_worker,
    )
    val_loader = InfiniteDataLoader(
        meta_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, sampler=None,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if opt.quad else LoadImagesAndLabelsAndMasks.collate_fn,
        pin_memory=True, worker_init_fn=seed_worker,
    )
    
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        test_loader = create_dataloader(test_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                        hyp=hyp, cache=None if noval else opt.cache, rect=opt.rect,
                                        rank=-1, workers=workers * 2, pad=0.0, mask_downsample_ratio=mask_ratio,
                                        overlap_mask=overlap, prefix=colorstr('test: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

            if plots:
                plot_labels(labels, names, save_dir)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    val_nb = len(val_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 0)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model, overlap=overlap)  # init loss class
    # callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    old_state = model.state_dict()
    scheduler_sam = lr_scheduler.OneCycleLR(optimizer_sam, max_lr=hyp['lr_sam'], total_steps=epochs*len(train_loader))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        sam_lora.train()
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 8) %
                    ('Epoch', 'GPU_mem', 'box_loss', 'seg_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _, masks) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            
            val_imgs, val_targets, val_paths, _, val_masks = next(iter(val_loader))
            val_imgs = val_imgs.to(device, non_blocking=True).float() / 255

            # Warmup
            if ni < nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                for j, x in enumerate(optimizer_sam.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                loss_sam, pred_bboxes, pred_masks_logits, pred = loss_cal(
                    model, imgs, targets, masks, nb, ema, single_cls, sam_lora, imgsz, device)
                loss, loss_items = compute_loss(
                    pred, targets.to(device), masks=masks.to(device).float(), loss_seg=loss_sam)
                # out, pred = model(imgs)  # forward
                # loss, loss_items = compute_loss(pred, targets.to(device), masks=masks.to(device).float())
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
            # Backward
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
                
            
            # Optimize SAM
            val_loss_sam, val_pred_bboxes, val_pred_masks_logits, val_pred = loss_cal(
                model, val_imgs, val_targets, val_masks, val_nb, ema, single_cls, sam_lora, imgsz, device)
            if val_loss_sam == None:
                continue
            val_loss, loss_items = compute_loss(
                val_pred, val_targets.to(device), masks=val_masks.to(device).float(), loss_seg=val_loss_sam)      
            # val_loss = val_loss_sam      
            optimizer_sam.zero_grad()
            torch.nn.utils.clip_grad_norm_(sam_lora.parameters(), max_norm=10.0)  # clip gradients
            val_loss.backward()
            optimizer_sam.step()

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 6) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                
                # Mosaic plots
                if mask_ratio != 1:
                    masks = F.interpolate(masks[None].float(), (imgsz, imgsz), mode="bilinear", align_corners=False)[0]
                if plots:
                    if ni < 3:
                        plot_images_and_masks(imgs, targets, masks, paths, save_dir / f"train_batch{ni}.jpg")
                    if ni == 10:
                        files = sorted(save_dir.glob('train*.jpg'))
                        logger.log_images(files, "Mosaics", ni+1)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        scheduler_sam.step()

        if RANK in {-1, 0}:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict, batch_size=batch_size // WORLD_SIZE, imgsz=imgsz, half=amp,
                    model=ema.ema, sam=sam_lora, single_cls=single_cls, dataloader=val_loader,
                    save_dir=save_dir, plots=False, callbacks=callbacks, compute_loss=compute_loss,
                    mask_downsample_ratio=mask_ratio, overlap=overlap, iou_thres=0.25,
                )
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            # Log val metrics and media
            metrics_dict = dict(zip(KEYS, log_vals))
            logger.log_metrics(metrics_dict, epoch+1)
            if plots:
                files = sorted(save_dir.glob('val*.jpg'))
                logger.log_images(files, "Validation", ni+1)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch_{epoch}.pt')
                    logger.log_model(w / f'epoch_{epoch}.pt')
                del ckpt

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict, batch_size=batch_size // WORLD_SIZE * 2, imgsz=imgsz,
                        model=attempt_load(f, device).half(), sam=sam_lora, iou_thres=0.25,
                        single_cls=single_cls, dataloader=test_loader, save_dir=save_dir,
                        save_json=is_coco, verbose=True, plots=plots, callbacks=callbacks,
                        compute_loss=compute_loss, mask_downsample_ratio=mask_ratio, overlap=overlap,
                    )  # val best model with plots
                    if is_coco:
                        metrics_dict = dict(zip(KEYS, list(mloss) + list(results) + lr))
                        logger.log_metrics(metrics_dict, epoch+1)

        # on train end callback using genericLogger
        logger.log_metrics(dict(zip(KEYS, (0,0,0,0)+results)), -1)
        if not opt.evolve:
            logger.log_model(best, ni+1)
        if plots:
            plot_results_with_masks(file=save_dir / 'results.csv')  # save results.png
            files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
            files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
            logger.log_images(files, "Results", ni+1)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s-seg.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-seg', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--wandb_mode', type=str, default='online', help='')
    
    # Instance Segmentation Args
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')
    
    # SAM Args
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    parser.add_argument('--sam_ckpt', type=str, default='<your sam checkpoint path>', help='Pretrained checkpoint')
    parser.add_argument('--data_rate', type=float, default=0.5, help='')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # Resume
    if opt.resume and not opt.evolve:  # resume from specified or most recent last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
