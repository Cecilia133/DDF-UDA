import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from numpy import sqrt
from util.metrics import *
from util.metrics import DiceLoss
import tqdm

from torch.autograd import Variable

import pdb

import datetime
import pickle
from model.targets import gen_targets
from util import config
import util.fundus_dataloader as DL
import util.custom_transforms as tr
from util.util import AverageMeter, poly_learning_rate, calc_mae, check_makedirs, clip_gradient, adjust_lr

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
from torch.utils.data import DataLoader
import math
import h5py
import os

global logger
global writer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/diff_UDA.yaml', help='config file')
    parser.add_argument('opts', help='see config/diff_UDA.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                    "%s: Detect checkpoint saved in data-parallel mode."
                    " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    # check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        torch.cuda.manual_seed(args.manual_seed)

    main_worker(args.train_gpu, args)


def main_worker(gpu, argss):
    global args
    global logger, writer
    args = argss
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    diff_steps = args.Diffusion_steps
    from model.deeplab import DeepLab
    from model.deeplab_EMA import DeepLab_EMA
    model = DeepLab(backbone=args.backbone, num_classes=args.num_classes, output_stride=args.out_stride)
    ema_model = DeepLab_EMA(backbone=args.backbone, num_classes=args.num_classes, output_stride=args.out_stride)
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr, betas=(0.9, 0.99))

    if main_process():
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.num_classes))
        logger.info(model)
        model = torch.nn.DataParallel(model.cuda())
        ema_model = torch.nn.DataParallel(ema_model.cuda())
        for param in ema_model.parameters():
            param.detach_()

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                logger.info("=> loaded weight '{}', epoch {}".format(args.weight, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
    

    composed_transforms_tr = transforms.Compose([
        #tr.RandomScaleCrop(512),
        tr.Resize(256),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_tr1 = transforms.Compose([
        #tr.Resize(256),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.Resize(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_S = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train/ROIs', transform=composed_transforms_tr)
    domain_loaderS = DataLoader(domain_S, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train/ROIs', transform=composed_transforms_tr1)
    domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.evaluate:
        domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='test/ROIs', transform=composed_transforms_ts)
        domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size_val, shuffle=False, num_workers=1, pin_memory=True)

    date_str = str(datetime.datetime.now().date())
    check_makedirs(args.save_path + '/' + date_str)

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if (epoch+1) % args.decay_epoch == 0:
            cur_lr = args.base_lr * args.decay_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
        else:
            cur_lr = args.base_lr

        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        loss_train, diff_steps_updated = train(domain_loaderS, domain_loaderT, model, ema_model, optimizer, epoch, diff_steps)  # 训练一次得到的loss
        diff_steps = diff_steps_updated

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        # pdb.set_trace()
        if args.evaluate:
            logger.info('>>>>>>>>>>>>>>>>Evaluating Epoch ' + str(epoch_log) + ' >>>>>>>>>>>>>>>>')
            disc_dice, cup_dice = validate(domain_loader_val, model)
            mean_dice = disc_dice + cup_dice
            is_best = mean_dice > best_dice
            if is_best:
                best_epoch = epoch_log
                best_dice = mean_dice
                filename = args.save_path + '/' + date_str + '/train_epoch_best' + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if main_process():
                writer.add_scalar('disc_dice', disc_dice)
                writer.add_scalar('cup_dice', cup_dice)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/' + date_str + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        logger.info('Current best epoch is Epoch {}'.format(best_epoch))


def train(domain_loaderS, domain_loaderT, model, ema_model, optimizer, epoch, diffusion_steps):

    diff_steps = diffusion_steps
    alpha = args.alpha

    batch_time = AverageMeter()
    data_time = AverageMeter()
    source_main_loss_meter = AverageMeter()
    Sloss_meter = AverageMeter()
    diff_main_loss_meter = AverageMeter()
    BN_loss_meter = AverageMeter()
    consistency_loss_meter = AverageMeter()

    BCE = torch.nn.BCELoss()
    MSE = torch.nn.MSELoss()
    L2 = torch.nn.MSELoss(reduction='mean')
    L1_loss = torch.nn.L1Loss()
    
    if args.train_type == 'UDA':
        target_main_loss_meter = AverageMeter()

    tensor = torch.FloatTensor

    model.train()
    ema_model.train()
    end = time.time()
    max_iter = args.epochs * len(domain_loaderS)

    ##### Source training
    for i, (sampleS) in tqdm.tqdm(enumerate(domain_loaderS), total=len(domain_loaderS), ncols=80, leave=False):
        data_time.update(time.time() - end)
        diff_alpha = args.alpha
        max_step = args.Diffusion_steps//args.batch_size
        betas = np.linspace(args.beta_start, args.beta_end, max_step, dtype=np.float64)
        ##### regular training
        imageS = sampleS['image'].to('cuda')
        target_map = sampleS['map'].to('cuda')
        target_boundary = sampleS['boundary'].to('cuda')
        bs = imageS.size(0)

        optimizer.zero_grad()
        for param in model.parameters():
            param.requires_grad = True
        ################################## UDA step #######################################
        prediction, pre_boundary, source_domain_info, source_feature_map = model(imageS)

        prediction = F.interpolate(prediction, size=(target_map.shape[2], target_map.shape[3]), mode='bilinear')

        # new added
        #loss_seg = BCE(torch.sigmoid(prediction), target_map)
        loss_seg = DiceLoss(torch.sigmoid(prediction), target_map)
        loss_edge = MSE(torch.sigmoid(pre_boundary), target_boundary)

        main_loss = loss_seg + loss_edge
        source_main_loss = main_loss
        
        if args.train_type == 'UDA':
            loss = source_main_loss # Use this to train UDA Deeplab
        else:
            loss = main_loss # Use this to train source-only Deeplab

        loss.backward()
        optimizer.step()

        ################################## Update EMA model #######################################
        if i == 0:
            alpha = min(1 - 1 / (i + 1), alpha)
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        ################################## Dissfusion step #######################################
        
        Max_diff_step = diff_steps//args.batch_size
        for T, (sampleT) in enumerate(domain_loaderT):
            if T < Max_diff_step:
                imageT = sampleT['image'].to("cuda")
                _, domain_info_pixel, domain_info_ema = ema_model(imageT)
                beta = betas[T]
                if imageS.size()[0] == domain_info_ema.size()[0]:
                    update_imgs = sqrt((1-beta))*imageS + sqrt(beta)*(domain_info_ema + domain_info_pixel)
                    bs = imageT.size(0)

                    optimizer.zero_grad()
                    for param in model.parameters():
                        param.requires_grad = True

                    diff_prediction, diff_boundary, diff_domain_info, diff_feature_map = model(update_imgs)
                    
                    consistency_loss = F.kl_div(diff_feature_map.softmax(dim=-1).log(), source_feature_map.detach().softmax(dim=-1), reduction='mean')
                    #consistency_loss = L2(diff_feature_map, source_feature_map.detach())
                    #consistency_loss = L1_loss(diff_feature_map, source_feature_map.detach())


                    #new added
                    diff_main_loss = BCE(torch.sigmoid(diff_prediction), target_map)
                    
                    BN_loss = 1 - F.cosine_similarity(diff_domain_info, domain_info_ema.detach())
                    BN_loss = torch.mean(BN_loss)


                    loss = diff_main_loss + BN_loss + consistency_loss
                    
                    loss.backward()
                    optimizer.step()

                    ################################## Update EMA model #######################################
                    diff_alpha = min(1 - 1 / (T + 1), diff_alpha)
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

                    ################################## Visualization Diffusion Training process #######################################
                    n = imageS.size(0)
                    if diff_feature_map.size()[0] != source_feature_map.size()[0]:
                        diff_main_loss_meter.update(diff_main_loss, n)
                        consistency_loss_meter.update(consistency_loss, n)
                    else:
                        diff_main_loss_meter.update(diff_main_loss.item(), n)
                        consistency_loss_meter.update(consistency_loss.item(), n)
                    BN_loss_meter.update(BN_loss.item(), n)

                    if (T+1) % args.diff_print_freq == 0 and main_process():
                        logger.info('Epoch: Diffusion_Steps: {}[{}/{}]/[{}/{}] '
                                    'diff_main_Loss {diff_main_loss_meter.val:.3f} '
                                    'domain_loss {BN_loss_meter.val:.3f} '
                                    'consistency_loss {consistency_loss_meter.val:.3f} '
                                    .format(epoch + 1, i + 1, len(domain_loaderS), T + 1, Max_diff_step,
                                                                                diff_main_loss_meter=diff_main_loss_meter,
                                                                                BN_loss_meter=BN_loss_meter,
                                                                                consistency_loss_meter=consistency_loss_meter,))
                    if main_process():
                        writer.add_scalar('diff_loss_train_step', diff_main_loss_meter.val, T + 1)
                    torch.cuda.empty_cache()

        ################################## Nash Balance #######################################
        
        N1 = BN_loss/consistency_loss
        N = torch.tanh(N1, out=None).cpu().detach().numpy()
        if N1 >= 1:
            N = N + 1
        else:
            N = 1.31 * N
        diff_steps_updated = np.round(diff_steps * N)

        ################################## Visualization Epoch Training process #######################################
        n = imageS.size(0)
        Sloss_meter.update(main_loss.item(), n)
        if args.train_type =='UDA':
            source_main_loss_meter.update(source_main_loss.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(domain_loaderS) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if args.train_type == 'UDA':
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'MainLoss {main_loss_meter.val:.3f} '
                            'Sloss {Sloss_meter.val:.3f} '
                            .format(epoch + 1, args.epochs, i + 1, len(domain_loaderS),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    remain_time=remain_time,
                                    main_loss_meter=source_main_loss_meter,
                                    Sloss_meter=Sloss_meter,
                                    ))
        else:
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'MainLoss {main_loss_meter.val:.3f} '
                            .format(epoch + 1, args.epochs, i + 1, len(domain_loaderS),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    remain_time=remain_time,
                                    main_loss_meter=Sloss_meter,
                                    ))

        if main_process():
            if args.train_type == 'UDA':
                writer.add_scalar('loss_source_train_batch', source_main_loss_meter.val, current_iter)
            else:
                writer.add_scalar('loss_source_train_batch', Sloss_meter.val, current_iter)
    if args.train_type == 'UDA' and args.dis_loss == True:
        # Target training
        for i, (sampleT) in tqdm.tqdm(enumerate(domain_loaderT), total=len(domain_loaderT), ncols=80, leave=False):
            data_time.update(time.time() - end)

            imageT = sampleT['image'].to('cuda')
            target_t = None
            tensor = torch.FloatTensor
            bs = imageT.size(0)

            optimizer.zero_grad()
            for param in model.parameters():
                param.requires_grad = True

            ################################## UDA step #######################################
            prediction, pre_boundary, _, _ = model(imageT)

            loss = target_dis_loss
            loss.backward()
            optimizer.step()

            ################################## Visualization Epoch Training process #######################################
            n = imageT.size(0)
            target_main_loss_meter.update(target_dis_loss.item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

            current_iter = epoch * len(domain_loaderT) + i + 1
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'MainLoss {main_loss_meter.val:.4f} '
                            .format(epoch + 1, args.epochs, i + 1, len(domain_loaderT),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    remain_time=remain_time,
                                    main_loss_meter=target_main_loss_meter,
                                    ))

        if main_process():
            logger.info('Train result at epoch [{}/{}]'.format(epoch + 1, args.epochs))

    torch.cuda.empty_cache()
    if args.train_type == 'UDA':
        return source_main_loss_meter.avg + target_main_loss_meter.avg, diff_steps
    else:
        return Sloss_meter.avg


def validate(domain_loader_val, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Target Evaluation >>>>>>>>>>>>>>>>')

    model.eval()
    val_loss = 0.0
    val_cup_dice = 0.0
    val_disc_dice = 0.0
    datanum_cnt = 0.0
    with torch.no_grad():
        for i, (sample) in enumerate(domain_loader_val):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            prediction = model(data)
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            loss_data = loss.data.item()

            val_loss += loss_data
            dice_cup, dice_disc = dice_coeff_2label(prediction, target)

            val_cup_dice += np.sum(dice_cup)
            val_disc_dice += np.sum(dice_disc)
            datanum_cnt += float(prediction.shape[0])

        val_loss /= datanum_cnt
        val_cup_dice /= datanum_cnt
        val_disc_dice /= datanum_cnt

    if main_process():
        logger.info(
            'Dice val result: Val_Loss / Disc_dice / Cup_dice {:.4f} / {:.4f} / {:.4f} '.format(val_loss, val_disc_dice, val_cup_dice))
        logger.info('<<<<<<<<<<<<<<<<< End Target Evaluation <<<<<<<<<<<<<<<<<')

    return val_disc_dice, val_cup_dice


if __name__ == '__main__':
    main()


