import os
import torch
import torch.distributed as dist
from typing import Tuple

def setup_ddp() -> Tuple[int, int, int]:
    """初始化DDP进程组，并返回rank, local_rank, world_size"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"DDP setup complete. World size: {world_size}")
    return rank, local_rank, world_size

def cleanup_ddp() -> None:
    """清理DDP进程组"""
    dist.destroy_process_group()

def reduce_metrics(metrics_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """对来自所有进程的指标进行All-Reduce求和"""
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    return metrics_tensor