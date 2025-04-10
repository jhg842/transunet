import argparse
import os
import random
import time
import datetime

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from model import build_model
from dataset import build_dataset
import util.misc as utils
from engine import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('Set transunet', add_help=False)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=100, type=float)
    parser.add_argument('--batch_size', default=24, type=int)    
    parser.add_argument('--epochs', default=100, type=int)
    
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--layers', default=True, type=bool)
    
    # Embedding
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--num_patches', default=16, type=int)
    parser.add_argument('--d_model', default=2048, type=int)
    parser.add_argument('--input_dim', default=784, type=int)
    parser.add_argument('--hidden_dim', default=2048, type=int)
    
    # Transformer
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--num_layers', default=12, type=int)
    
    parser.add_argument('--n_classes', default=8, type=int)
    
    
    parser.add_argument('--dataset_file', default='NG', type=str)
    parser.add_argument('--NG_path', default='', type=str)
    parser.add_argument('--output_dir',default='', type=str)
    parser.add_argument('--seed', default=777, type=int)
    
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help = 'url sued to set up distributed training')    
    
    return parser
    
def main(args):

    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, dice_loss = build_model(args)
    cross_loss = nn.CrossEntropyLoss()
    model.to(device)
    
    model_withiout_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_withiout_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_dicts = [
        {'params': [p for p in model.parameters() if p.requires_grad]},
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr = args.lr, weight_decay = args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    dataset_train = build_dataset(image_set = 'train', args=args)
    dataset_val = build_dataset(image_set = 'val', args=args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle = False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)    

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last = True)
    
    data_loader_train = DataLoader(dataset_train, batch_sampler = batch_sampler_train,
                                  num_workers = args.num_workers)
    data_loader_val = DataLoader(dataset_val, args_batch_size, sampler = sampler_val,
                                  drop_last = False)
    
    output_dir = Path(args.output_dir)
    
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, cross_loss, dice_loss, data_loader_train, optimizer, device)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 50 == 0:
                checkpoint_paths.append(otuput_dir / f'checkpoit{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_withiout_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args':args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds = int(total_time)))
    print('Training time {}'.format(total_time_str))    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransUnet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)

