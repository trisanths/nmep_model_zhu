"""Fast MediumImageNet trainer for the Kaggle part of HW3.

The whole split sits in RAM as uint8 and every augmentation runs on the GPU,
so busy CPUs never starve the GPU. Recipe: random resized crop + flip +
light color jitter, mixup/cutmix with label smoothing, SGD (nesterov) with
warmup and cosine decay, bf16 autocast, channels_last, weight EMA, flip TTA.

    python fast_train.py --arch timm:resnet50d --epochs 16 --out runs/r50d
"""
import argparse
import json
import math
import os
import queue
import threading
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/data/nmep/cv/medium-imagenet-96.hdf5")
    p.add_argument("--arch", default="timm:resnet50d")
    p.add_argument("--no-maxpool", type=int, default=1, help="drop the stem maxpool of timm resnets (96px input)")
    p.add_argument("--drop-path", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--warmup", type=float, default=1, help="warmup epochs")
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--mix-off-epochs", type=int, default=2, help="turn mixup/cutmix off for the last N epochs")
    p.add_argument("--scale-min", type=float, default=0.30, help="random resized crop min area")
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--res", type=int, default=96, help="final train resolution")
    p.add_argument("--res-start", type=int, default=0, help="progressive resizing start res (0 = off)")
    p.add_argument("--res-ramp", type=float, default=0.6, help="fraction of epochs to reach --res")
    p.add_argument("--eval-res", default="96,112,128")
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_split(path, split):
    with h5py.File(path, "r") as f:
        x = f[f"images-{split}"][:]
        y = f[f"labels-{split}"][:].astype(np.int64) if f"labels-{split}" in f else None
    return x, y


class Loader:
    """Background thread that gathers uint8 batches into pinned memory."""

    def __init__(self, x, y, bs, seed=0):
        self.x, self.y, self.bs = x, y, bs
        self.rng = np.random.default_rng(seed)
        self.n_batches = len(x) // bs

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        order = self.rng.permutation(len(self.x))
        q = queue.Queue(maxsize=8)

        def work():
            for b in range(self.n_batches):
                idx = np.sort(order[b * self.bs:(b + 1) * self.bs])
                q.put((torch.from_numpy(self.x[idx]).pin_memory(), torch.from_numpy(self.y[idx]).pin_memory()))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            yield item[0].cuda(non_blocking=True), item[1].cuda(non_blocking=True)


def normalize(x):
    return (x.float() - MEAN.to(x.device)) / STD.to(x.device)


def gpu_augment(x, out_res, scale_min, jitter):
    """Per-sample random resized crop + horizontal flip in one grid_sample, then color jitter."""
    b, dev = x.shape[0], x.device
    x = normalize(x)
    area = torch.empty(b, device=dev).uniform_(scale_min, 1.0)
    ratio = torch.exp(torch.empty(b, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)))
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (torch.rand(b, device=dev) * 2 - 1) * (1 - w)
    cy = (torch.rand(b, device=dev) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = w * flip, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, (b, 3, out_res, out_res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if jitter > 0:
        def fac():
            return torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - jitter, 1 + jitter)
        x = x * fac()
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * fac() + m
        g = x.mean(dim=1, keepdim=True)
        x = (x - g) * fac() + g
    return x.contiguous(memory_format=torch.channels_last)


def mix_batch(x, y, num_classes, mixup, cutmix, prob, smoothing):
    """Mixup or CutMix (chosen per batch) on label-smoothed one-hot targets."""
    off = smoothing / num_classes
    t = torch.full((len(y), num_classes), off, device=y.device)
    t.scatter_(1, y[:, None], 1 - smoothing + off)
    if (mixup <= 0 and cutmix <= 0) or np.random.rand() > prob:
        return x, t
    perm = torch.randperm(len(y), device=y.device)
    if cutmix > 0 and (mixup <= 0 or np.random.rand() < 0.5):
        lam = np.random.beta(cutmix, cutmix)
        H, W = x.shape[-2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    else:
        lam = np.random.beta(mixup, mixup)
        x = x * lam + x[perm] * (1 - lam)
    return x, t * lam + t[perm] * (1 - lam)


def build_model(args, num_classes):
    if args.arch.startswith("timm:"):
        import timm
        m = timm.create_model(args.arch[5:], num_classes=num_classes, drop_path_rate=args.drop_path)
        if args.no_maxpool and hasattr(m, "maxpool"):
            m.maxpool = nn.Identity()
        return m
    if args.arch.startswith("hw_resnet18"):
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes, stem_stride=1 if args.arch.endswith("_s1") else 2)
    raise ValueError(args.arch)


class EMA:
    """Exponential moving average of the weights (and a copy of BN buffers)."""

    def __init__(self, model, decay):
        import copy
        self.decay, self.steps = decay, 0
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.src_p = [p for p in model.parameters()]
        self.dst_p = [p for p in self.module.parameters()]
        self.src_b = [b for b in model.buffers()]
        self.dst_b = [b for b in self.module.buffers()]

    @torch.no_grad()
    def update(self):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        torch._foreach_lerp_(self.dst_p, self.src_p, 1 - d)
        for s, t in zip(self.src_b, self.dst_b):
            t.copy_(s)


def param_groups(model, wd):
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict(model, x_np, res, bs=1000, tta=True):
    """Log-probabilities; with tta, the average of the image and its mirror."""
    model.eval()
    out = []
    for i in range(0, len(x_np), bs):
        xb = normalize(torch.from_numpy(x_np[i:i + bs]).cuda(non_blocking=True))
        if res != xb.shape[-1]:
            xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
        xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(xb).float().softmax(-1)
            if tta:
                p = (p + model(torch.flip(xb, dims=[3])).float().softmax(-1)) / 2
        out.append(p.clamp_min(1e-8).log().cpu())
    return torch.cat(out).numpy()


def acc(logp, y):
    return float((logp.argmax(1) == y).mean() * 100)


def res_at(args, epoch):
    if not args.res_start:
        return args.res
    end = max(1, int(args.epochs * args.res_ramp))
    if epoch >= end:
        return args.res
    return int(round((args.res_start + (args.res - args.res_start) * epoch / end) / 16) * 16)


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    xtr, ytr = load_split(args.data, "train")
    xva, yva = load_split(args.data, "val")
    num_classes = int(ytr.max()) + 1
    print(f"data: train {xtr.shape} val {xva.shape} classes {num_classes} load {time.time()-t0:.0f}s", flush=True)

    model = build_model(args, num_classes).cuda().to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {args.arch}: {n_params/1e6:.2f}M params", flush=True)

    opt = torch.optim.SGD(param_groups(model, args.wd), lr=args.lr, momentum=0.9, nesterov=True)
    loader = Loader(xtr, ytr, args.bs, seed=args.seed)
    total = args.epochs * len(loader)
    warm = int(args.warmup * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema = EMA(model, args.ema) if args.ema > 0 else None
    fwd = torch.compile(model) if args.compile else model
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        res = res_at(args, epoch)
        mixing = epoch < args.epochs - args.mix_off_epochs
        ep_t0 = time.time()
        loss_sum = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        seen = 0
        for xb, yb in loader:
            x = gpu_augment(xb, res, args.scale_min, args.jitter)
            if mixing:
                x, t = mix_batch(x, yb, num_classes, args.mixup, args.cutmix, args.mix_prob, args.smoothing)
            else:
                x, t = mix_batch(x, yb, num_classes, 0, 0, 0, args.smoothing)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd(x)
            loss = torch.sum(-t * F.log_softmax(out.float(), dim=-1), dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema:
                ema.update()
            loss_sum += loss.detach() * len(yb)
            correct += (out.argmax(1) == yb).sum()
            seen += len(yb)
        ep_time = time.time() - ep_t0
        log = {"epoch": epoch, "res": res, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum.item() / seen,
               "train_acc": 100 * correct.item() / seen, "epoch_time": round(ep_time, 1),
               "img_per_s": round(seen / ep_time), "elapsed_min": round((time.time() - train_t0) / 60, 1)}
        if epoch == args.epochs - 1 or (epoch + 1) % args.val_every == 0:
            lv = predict(model, xva, args.res, tta=False)
            log["val_acc"] = acc(lv, yva)
            log["val_loss"] = float(-lv[np.arange(len(yva)), yva].mean())
            if ema:
                log["val_acc_ema"] = acc(predict(ema.module, xva, args.res, tta=False), yva)
        print(json.dumps(log), flush=True)
        with open(os.path.join(args.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict() if ema else None,
                    "epoch": epoch, "args": vars(args)}, os.path.join(args.out, "last.pth"))
    train_time = time.time() - train_t0

    # Pick the best of {raw, ema} x eval resolution on val (with flip TTA), then predict test with it.
    results, best = {}, None
    for name, m in [("raw", model)] + ([("ema", ema.module)] if ema else []):
        for r in [int(v) for v in args.eval_res.split(",")]:
            lv = predict(m, xva, r)
            results[f"{name}@{r}"] = a = acc(lv, yva)
            if best is None or a > best[0]:
                best = (a, name, r, m, lv)
    a, name, r, m, lv = best
    print(f"eval grid {json.dumps(results)} best {name}@{r} = {a:.2f}", flush=True)
    del xtr
    xte, _ = load_split(args.data, "test")
    lt = predict(m, xte, r)
    np.save(os.path.join(args.out, "val_logp.npy"), lv.astype(np.float16))
    np.save(os.path.join(args.out, "test_logp.npy"), lt.astype(np.float16))
    np.save(os.path.join(args.out, "preds.npy"), lt)
    preds = lt.argmax(1)
    np.savetxt(os.path.join(args.out, "predictions.csv"), np.column_stack((np.arange(len(preds)), preds)),
               fmt="%d", delimiter=",", header="Id,Category", comments="")
    torch.save(m.state_dict(), os.path.join(args.out, "best_weights.pth"))
    summary = {"best_val_acc": a, "pick": f"{name}@{r}", "grid": results, "train_time_s": round(train_time),
               "n_params": n_params, "test_n": len(lt)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
