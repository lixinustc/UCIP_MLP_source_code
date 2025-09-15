import argparse
import os

import numpy as np
import cv2
import utils
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch import distributed as dist
import torch.optim as optim
import torch.nn.functional as F

import srdata
from model_ucip import UCIP_Middle
import utils_logger
import logging
import util_calculate_psnr_ssim as util

from my_loss import CharbonnierLoss


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description='Train HST')

    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to the checkpoints for pretrained model",
    )
    parser.add_argument(
        '--distributed',
        action='store_true'
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument('--save_every', type=int, default=5000, help='save weights')
    parser.add_argument('--eval_every', type=int, default=5000, help='test network')
    parser.add_argument('--ranker', type=int)
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint', help='path to save checkpoints')

    args = parser.parse_args()

    return args

def data_sampler(dataset, shuffle=True, distributed=True):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def get_bare_model(network):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(network, DistributedDataParallel):
        network = network.module
    return network

def update_E(model, model_E, decay=0.999):
    netG = get_bare_model(model)
    netG_params = dict(netG.named_parameters())
    netE_params = dict(model_E.named_parameters())
    for k in netG_params.keys():
        netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)



def test(model, img):
    _, _, h_old, w_old = img.size()
    padding = 4
    h_pad = (h_old // padding + 1) * padding - h_old
    w_pad = (w_old // padding + 1) * padding - w_old
    img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
    
    img = model(img)
    img = img[..., :h_old * 4, :w_old * 4]
    return img

def main():
    args = parse_args()

    checkpoint_save_path = os.path.join(args.ckpt_path, f'M_p64_b{args.batch_size}_g{args.gpus}_64_maximlocal')
    if not os.path.exists(checkpoint_save_path) and torch.cuda.current_device() == 0:
        os.makedirs(checkpoint_save_path, exist_ok=True)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(checkpoint_save_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        # synchronize()
    
    # set random seeds and log
    if args.seed is not None:
        if torch.cuda.current_device() == 0:
            logger.info('Set random seed to {}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model = UCIP_Middle(p_dim=64, img_size=64) 
    if args.use_ema:
        model_E = UCIP_Middle(p_dim=64, img_size=64).to('cuda').eval()

    model = model.to('cuda')
   
    if args.resume is not None:
        if torch.cuda.current_device() == 0:
            logger.info("load model: ", args.resume)
        if not os.path.isfile(args.resume):
            raise ValueError

        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt)
        if args.use_ema:
            model_E.load_state_dict(ckpt)
        if torch.cuda.current_device() == 0:
            logger.info("model checkpoint load!")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100000, 200000], 0.5)

    loss_fn = torch.nn.L1Loss()
    loss_fn = loss_fn.to('cuda')

    trainset = srdata.Data_Train(patch_size=64, data_root=args.data_root)
    testset = srdata.Data_Test()

    data_loader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        sampler=data_sampler(trainset, shuffle=True, distributed=args.distributed),
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    data_loader_test = data.DataLoader(
        testset, 
        batch_size=1,
        sampler=data_sampler(testset, shuffle=False, distributed=False),
        num_workers=1,
        pin_memory=True
    )


    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

    total_epochs = 1000000

    lr_decay = [600000, 1200000, 1800000]

    current_step = 0

    for epoch in range(0, total_epochs+1):

        m = model.module if args.distributed else model

        model.train()
        if args.use_ema:
            model_E.eval()

        for batch, pik in enumerate(data_loader):
            current_step += 1
            learning_rate = optimizer.param_groups[0]['lr']

            lr = pik['L']
            hr = pik['H']
            filename = pik['N']
            task_id = pik['TASK_ID']
            task_id = list(task_id.numpy())

            filename = filename[0].split('/')[-1]

            optimizer.zero_grad()
            lr = lr.to('cuda')
            hr = hr.to('cuda')

            sr = model(lr, task_id)
            loss = loss_fn(sr, hr)
       
            loss_print = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            if args.use_ema:
                update_E(model, model_E)

            if current_step % 200 == 0 and torch.cuda.current_device() == 0:
                logger.info('Epoch: {}\tStep: {}\t[{}/{}]\t{}LR: {}'.format(
                    epoch,
                    current_step,
                    (batch + 1) * args.batch_size,
                    int(len(trainset)/args.gpus),
                    loss_print,
                    learning_rate))
            
            if not current_step % args.save_every and torch.cuda.current_device() == 0:
                model_dict = m.state_dict()
                torch.save(
                    model_dict,
                    os.path.join(checkpoint_save_path, 'model_{}.pt'.format(current_step))
                )
                if args.use_ema:
                    torch.save(
                        model_E.state_dict(),
                        os.path.join(checkpoint_save_path, 'model_E_{}.pt'.format(current_step))
                    )
            # valid
            if args.eval_every and not current_step % args.eval_every and torch.cuda.current_device() == 0:

                model_E.eval()
                # model.eval()

                p = 0
                s = 0
                p_y = 0
                s_y = 0
                count = 0
                for batch in data_loader_test:
                    count += 1
                    lr = batch['L']
                    hr = batch['H']
                    filename = batch['N']
                    lr = lr.to('cuda')
                    b, c, h, w = lr.size()
                    h_pad = 8 - h % 8
                    w_pad = 8 - w % 8
                    filename = filename[0]
                    task_id = [0]
                    lr = F.pad(lr, (0, w_pad, 0, h_pad), mode='reflect')
                    with torch.no_grad():
                        sr = model_E(lr, task_id)
                        sr = sr[:, :, :h*4, :w*4]
                    sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
                    sr = sr * 255.
                    sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
                    hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
                    hr = hr * 255.
                    hr = np.clip(hr.round(), 0, 255).astype(np.uint8)
                    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
                    psnr = util.calculate_psnr(sr, hr, crop_border=4, test_y_channel=False)
                    ssim = util.calculate_ssim(sr, hr, crop_border=4, test_y_channel=False)
                    psnr_y = util.calculate_psnr(sr, hr, crop_border=4)
                    ssim_y = util.calculate_ssim(sr, hr, crop_border=4)
                    p += psnr
                    s += ssim
                    p_y += psnr_y
                    s_y += ssim_y
                    logger.info('{}: {}, {}'.format(filename, psnr, ssim))
                    #sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                p /= count
                s /= count
                p_y /= count
                s_y /= count

                logger.info("Epoch: {}, Step: {}, psnr: {}. ssim: {}. psnr_y: {}. ssim_y: {}.".format(epoch, current_step, p, s, p_y, s_y))
                model.train()

        if current_step > 300000:
           break
        
    logger.info('Done')

if __name__ == '__main__':
    main()
