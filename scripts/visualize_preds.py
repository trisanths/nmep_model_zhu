"""Q3.2: show 20 CIFAR-10 validation images with true and predicted labels.

    python scripts/visualize_preds.py --cfg configs/resnet18_base.yaml \
        --ckpt runs/resnet18/resnet18/ckpt_best.pth --out figs/resnet18_preds.png
"""
import argparse
import os
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import CIFAR10Dataset  # noqa: E402
from models.resnet import ResNet18  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", default="/root/cifar10")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = ResNet18(num_classes=10)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval().cuda()
    ds = CIFAR10Dataset(img_size=32, train=False, root=args.root)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[: args.n]
    x = torch.stack([ds[i][0] for i in idx]).cuda()
    with torch.no_grad():
        pred = model(x).argmax(1).cpu()
    cols = 5
    fig, axes = plt.subplots(args.n // cols, cols, figsize=(cols * 2.2, args.n // cols * 2.5))
    for ax, i, pr in zip(axes.flat, idx.tolist(), pred.tolist()):
        img, label = ds.dataset[i]
        ax.imshow(img)
        ok = pr == label
        ax.set_title(f"true: {CLASSES[label]}\npred: {CLASSES[pr]}", fontsize=8, color="black" if ok else "red")
        ax.axis("off")
    n_ok = sum(int(pr == ds.dataset[i][1]) for i, pr in zip(idx.tolist(), pred.tolist()))
    fig.suptitle(f"ResNet18 on CIFAR-10 val: {n_ok}/{args.n} correct (wrong ones in red)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}: {n_ok}/{args.n} correct")


if __name__ == "__main__":
    main()
