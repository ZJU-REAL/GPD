"""
Test NCCL communication between GPU 4 and GPU 5,
each in a separate process with CUDA_VISIBLE_DEVICES=single GPU.
This replicates Ray's per-worker GPU isolation.
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    gpu = [2, 4][rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["RANK"] = str(rank)
    os.environ["NCCL_DEBUG"] = "INFO"

    torch.cuda.set_device(0)  # device 0 = the single visible GPU
    print(f"[rank {rank}] using GPU {gpu} (cuda:{torch.cuda.current_device()})", flush=True)

    dist.init_process_group(
        backend="nccl",
        world_size=2,
        rank=rank,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    print(f"[rank {rank}] init_process_group done", flush=True)

    t = torch.ones(1).cuda(0)
    dist.all_reduce(t)
    print(f"[rank {rank}] ALLREDUCE OK, result={t.item()}", flush=True)

    dist.barrier()
    print(f"[rank {rank}] barrier OK", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("ALL RANKS DONE — NCCL works correctly between GPU 4 and GPU 5")
