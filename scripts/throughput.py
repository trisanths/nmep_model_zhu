"""Q2.4: AlexNet training throughput vs batch size (1, 2, 4, ... until CUDA OOM).

Runs the same training step as main.py (AdamW, cross entropy, the CIFAR-10
loader from the repo) and times whole epochs. Small batches are slow, so
batch sizes up to 16 time 1 epoch and larger ones average 3 epochs, each after
a short warmup. Batches larger than CIFAR-10 (50,000 images) can't come from
the loader, so past that point the OOM search uses random tensors.

    python scripts/throughput.py --out runs/throughput.json
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import CIFAR10Dataset  # noqa: E402
from models.alexnet import AlexNet  # noqa: E402


def step_fn(model, opt, crit):
    def step(x, y):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    return step


def time_epochs(bs, root, n_epochs, workers):
    ds = CIFAR10Dataset(img_size=70, train=True, root=root)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True,
                                     persistent_workers=True, drop_last=False)
    model = AlexNet(num_classes=10).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = step_fn(model, opt, torch.nn.CrossEntropyLoss())
    it = iter(dl)
    for _ in range(min(20, len(dl))):  # warmup: cudnn autotune, worker startup
        x, y = next(it)
        step(x.cuda(non_blocking=True), y.cuda(non_blocking=True))
    torch.cuda.synchronize()
    times = []
    for _ in range(n_epochs):
        t = time.time()
        for x, y in dl:
            step(x.cuda(non_blocking=True), y.cuda(non_blocking=True))
        torch.cuda.synchronize()
        times.append(time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20
    del model, opt, dl
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return times, mem, len(ds)


def fits(bs):
    """One synthetic training step at batch size bs. Returns (fits, peak MB, error)."""
    def run():
        model = AlexNet(num_classes=10).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randn(bs, 3, 70, 70, device="cuda")
        y = torch.randint(0, 10, (bs,), device="cuda")
        step_fn(model, opt, torch.nn.CrossEntropyLoss())(x, y)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20

    try:
        out = (True, run(), None)
    except torch.OutOfMemoryError:
        out = (False, None, "CUDA out of memory")
    except RuntimeError as e:  # e.g. cuDNN refusing a tensor this large
        out = (False, None, str(e).splitlines()[0][:200])
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/cifar10")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    torch.backends.cudnn.benchmark = True
    results = []
    bs = 1
    while True:
        if bs > 50000:
            ok, mem, err = fits(bs)
            row = {"batch_size": bs, "synthetic": True, "fits": ok, "max_mem_mb": mem, "error": err}
            print(json.dumps(row), flush=True)
            results.append(row)
            if not ok:
                break
        else:
            try:
                times, mem, n = time_epochs(bs, args.root, 1 if bs <= 16 else 3, args.workers)
            except torch.OutOfMemoryError:
                row = {"batch_size": bs, "fits": False}
                print(json.dumps(row), flush=True)
                results.append(row)
                break
            ep = sum(times) / len(times)
            row = {"batch_size": bs, "fits": True, "epoch_times": times, "epoch_time": ep,
                   "img_per_s": n / ep, "max_mem_mb": mem}
            print(json.dumps(row), flush=True)
            results.append(row)
        with open(args.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(), "results": results}, f, indent=1)
        bs *= 2
    with open(args.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
