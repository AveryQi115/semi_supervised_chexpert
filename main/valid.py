import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import accuracy, AverageMeter, FusionMatrix
from loss import *
import json


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, device, num_classes, criterion):
    model.eval()
    # fusion_matrix = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)
            # feature = model(image, feature_flag=True)
            output = model(image)
            loss = criterion(output, label)
            if cfg.TRAIN.COMBINER.TYPE == 'multi_label':
                now_result = torch.sigmoid(output).ge(0.5).float()
            else:
                score_result = func(output)
                now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), label.shape[0])
            # fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            if cfg.TRAIN.COMBINER.TYPE == 'multi_label':
                now_acc = (now_result == label).sum()/label.shape[0]/label.shape[1]
                cnt = label.shape[0]
            else:
                now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            all_loss.avg, acc.avg * 100
        )
        
    return acc.avg, all_loss.avg


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = Network(cfg, mode="test", num_classes=num_classes)

    with open('train_data_statistic.json') as json_file:
        num_class_list = json.load(json_file)

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }

    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    '''
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)
    '''

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_model(testLoader, model, cfg, device, num_classes,criterion)
