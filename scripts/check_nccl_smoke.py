"""Small collective check; wrap torchrun in `timeout 90s` to bound init hangs.

The process-group timeout does not reliably bound communicator initialization.
"""
import datetime
import os

import torch
import torch.distributed as dist


if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=45))
    print(f"rank={rank} group ready", flush=True)
    x = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(x)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    assert x.item() == expected
    print(f"rank={rank} all_reduce={x.item()} PASS", flush=True)
    dist.destroy_process_group()
