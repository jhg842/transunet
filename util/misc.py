import torch
import torch.distributed as dist
import os
import subprocess

def is_main_process():
    
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
            
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    
    return dist.get_rank()

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        
    else:
        print("Not Using distributed mode")
        args.distributed = False
        
        return
    
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    
    dist.init_process_group(backend = args.dist_backend, init_method = args.dist_url,
                            world_size = args.world_size, rank = args.rank)
    
    dist.barrier()
    setup_for_distributed(args.rank == 0)
    