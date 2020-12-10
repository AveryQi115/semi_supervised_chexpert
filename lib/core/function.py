import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
import torch.nn.functional as F
import numpy as np
import torch
import time
from tqdm import tqdm
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import ipdb


def train_model_for_coteaching(trainLoader,
    model1,
    model2,
    epoch,
    epoch_number,
    optimizer1,
    optimizer2,
    combiner,
    criterion,
    cfg,
    logger,
    rank=0,
    **kwargs):

    if cfg.EVAL_MODE:
        model1.eval()
        model2.eval()
    else:
        model1.train()
        model2.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss1 = AverageMeter()
    acc1 = AverageMeter()
    all_loss2 = AverageMeter()
    acc2 = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        
        # ipdb.set_trace()

        cnt = label.shape[0]
        loss1, now_acc1, loss2, now_acc2 = combiner.forward((model1, model2), criterion, image, label, meta)

        #------use optimizer1 to update model1 params----------------------------#
        optimizer1.zero_grad()
        if cfg.TRAIN.DISTRIBUTED:
            if 'res50_sw' not in cfg.BACKBONE.TYPE:
                with amp.scale_loss(loss1, optimizer1) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss1.backward()
        else:
            loss1.backward()
        optimizer1.step()
        #------model1 params updated---------------------------------------------#

        #------use optimizer2 to update model2 params----------------------------#
        optimizer2.zero_grad()
        if cfg.TRAIN.DISTRIBUTED:
            if 'res50_sw' not in cfg.BACKBONE.TYPE:
                with amp.scale_loss(loss2, optimizer2) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss2.backward()
        else:
            loss2.backward()
        optimizer2.step()
        #------model2 params updated---------------------------------------------#

        all_loss1.update(loss1.data.item(), cnt)
        acc1.update(now_acc1.data.item(), cnt)

        all_loss2.update(loss2.data.item(), cnt)
        acc2.update(now_acc2.data.item(), cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            # ipdb.set_trace()
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Model1_Batch_Loss:{:>5.3f}  Model1_Batch_Accuracy:{:>5.2f}%    Model2_Batch_Loss:{:>5.3f}  Model2_Batch_Accuracy:{:>5.2f}%  ".format(
                epoch, i, number_batch, all_loss1.val, acc1.val, all_loss2.val, acc2.val
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Model1_Avg_Loss:{:>5.3f}   Model1_Epoch_Accuracy:{:>5.2f}    Model2_Avg_Loss:{:>5.3f}   Model2_Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss1.avg, acc1.avg, all_loss2.avg, acc2.avg, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc1.avg, all_loss1.avg, acc2.avg, all_loss2.avg

def train_model(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    combiner,
    criterion,
    cfg,
    logger,
    rank=0,
    **kwargs
):
    if cfg.TRAIN.COMBINER.TYPE=='coteaching':
        model1,model2 = model
        optimizer1,optimizer2 = optimizer
        return train_model_for_coteaching(trainLoader,
                                            model1,
                                            model2,
                                            epoch,
                                            epoch_number,
                                            optimizer1,
                                            optimizer2,
                                            combiner,
                                            criterion,
                                            cfg,
                                            logger,
                                            rank=0,
                                            **kwargs)

    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        # ipdb.set_trace()
        loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()
        if cfg.TRAIN.DISTRIBUTED:
            if 'res50_sw' not in cfg.BACKBONE.TYPE:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        else:
            loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg

def valid_model_for_coteaching(
    dataLoader, epoch_number, model1, model2, cfg, criterion, logger, device, rank, **kwargs
):
    model1.eval()
    model2.eval()

    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)
    fusion_matrix2 = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss1 = AverageMeter()
        acc1 = AverageMeter()
        all_loss2 = AverageMeter()
        acc2 = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature1 = model1(image, feature_flag=True)
            feature2 = model2(image, feature_flag=True)

            output1 = model1(feature1, classifier_flag=True)
            output2 = model2(feature2, classifier_flag=True)

            loss1 = criterion(output1, label)
            score_result1 = func(output1)

            loss2 = criterion(output2, label)
            score_result2 = func(output2)

            now_result1 = torch.argmax(score_result1, 1)
            now_result2 = torch.argmax(score_result2, 1)

            all_loss1.update(loss1.data.item(), label.shape[0])
            all_loss2.update(loss2.data.item(), label.shape[0])

            fusion_matrix.update(now_result1.cpu().numpy(), label.cpu().numpy())
            fusion_matrix2.update(now_result2.cpu().numpy(), label.cpu().numpy())

            now_acc1, cnt1 = accuracy(now_result1.cpu().numpy(), label.cpu().numpy())
            now_acc2, cnt2 = accuracy(now_result2.cpu().numpy(), label.cpu().numpy())

            acc1.update(now_acc1, cnt1)
            acc2.update(now_acc2, cnt2)

        pbar_str = "------- Valid: Epoch:{:>3d}  Model1_Valid_Loss:{:>5.3f}   Model1_Valid_Acc:{:>5.2f}%   Model2_Valid_Loss:{:>5.3f}   Model2_Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss1.avg, acc1.avg * 100, all_loss2, acc2.avg * 100
        )
        if rank == 0:
            logger.info(pbar_str)
    return acc1.avg, all_loss1.avg, acc2.avg, all_loss2.avg



def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, **kwargs
):
    if cfg.TRAIN.COMBINER.TYPE == 'coteaching':
        model1, model2 = model
        return valid_model_for_coteaching(dataLoader, epoch_number, model1, model2, cfg, criterion, logger, device, rank, **kwargs)
    
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    if cfg.TRAIN.COMBINER.TYPE != 'multi_label':
        fusion_matrix = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            if cfg.TRAIN.COMBINER.TYPE == 'multi_label':
                output = model(image)
            else:
                feature = model(image, feature_flag=True)
                output = model(feature, classifier_flag=True)

            loss = criterion(output, label)
            if cfg.TRAIN.COMBINER.TYPE == 'multi_label':
                now_result = torch.sigmoid(output).ge(0.5).float()
            else:
                score_result = func(output)
                now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), label.shape[0])
            if cfg.TRAIN.COMBINER.TYPE != 'multi_label':
                fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            if cfg.TRAIN.COMBINER.TYPE == 'multi_label':
                now_acc = (now_result == label).sum()/label.shape[0]/label.shape[1]
                cnt = label.shape[0]
            else:
                now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc.avg * 100
        )
        if rank == 0:
            logger.info(pbar_str)
    return acc.avg, all_loss.avg
