import os
import subprocess

import torch


def pick_device():
    """GPU guard from the execution protocol: the first card with under 1 GB in use, else CPU.

    When CUDA_VISIBLE_DEVICES pins one card, only that card is checked, and it is cuda:0 inside the process.
    """
    if os.environ.get("REBUILD_DEVICE"):   # a launcher that already ran this guard pins the device
        return torch.device(os.environ["REBUILD_DEVICE"])
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        used = dict(tuple(int(v) for v in r.split(",")) for r in rows)
    except Exception:
        return torch.device("cpu")
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pinned:
        card = int(pinned.split(",")[0])
        return torch.device("cuda:0") if used.get(card, 1 << 30) < 1024 else torch.device("cpu")
    for card, mib in sorted(used.items()):
        if mib < 1024:
            return torch.device(f"cuda:{card}")
    return torch.device("cpu")


def describe(dev):
    return f"device {dev} | node {os.uname().nodename} | pid {os.getpid()}"
