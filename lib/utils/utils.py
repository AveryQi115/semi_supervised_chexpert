import logging
import time
import os

import torch
from utils.lr_scheduler import WarmupMultiStepLR
from net import Network
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import apex
import ipdb


def create_logger(cfg, rank=0):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank > 0: 
        return logger, log_file
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=cfg.TRAIN.LR_SCHEDULER.DECAY_ETA_MIN
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=cfg.TRAIN.LR_SCHEDULER.DECAY_ETA_MIN
            )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_model(cfg, num_classes, device, logger):
    model = Network(cfg, mode="train", num_classes=num_classes)

    if cfg.BACKBONE.FREEZE == True:
        model.freeze_backbone()
        logger.info("Backbone has been freezed")

    if cfg.CPU_MODE:
        model = model.to(device)
    elif cfg.TRAIN.DISTRIBUTED:
        if cfg.TRAIN.SYNCBN:
            model = apex.parallel.convert_syncbn_model(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model

def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    if cfg.DATASET.DATASET == 'CheXpert':
        negative_list = dict()
        positive_list = dict()
        cat_list = []

        # initialize dict
        for key in annotations[0].keys():
            positive_list[key] = 0
            negative_list[key] = 0

        for anno in annotations:
            cat = []
            for key in anno.keys():
                # ipdb.set_trace()
                if key not in ['path','Sex','Age','Frontal/Lateral','AP/PA'] and type(anno[key])!=str and anno[key] >= 1:
                    positive_list[key] += 1
                    cat.append(key)
                
                if key not in ['path','Sex','Age','Frontal/Lateral','AP/PA'] and type(anno[key])!=str and anno[key] == 0:
                    negative_list[key] += 1
            cat_list.append(cat)
        
        for i, key in enumerate(negative_list.keys()):
            assert negative_list[key] != 0,f"negative_list[{key}]=0 error"
            num_list[i] = positive_list[key]/negative_list[key]
        return num_list,cat_list
        
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list