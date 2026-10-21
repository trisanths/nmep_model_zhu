"""Make the Part 2 figures and a summary table from runs/<tag>/<model>/metrics.json.

    python scripts/plots.py --runs runs --out figs
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(runs):
    out = {}
    for p in glob.glob(os.path.join(runs, "*", "*", "metrics.json")):
        tag = p.split(os.sep)[-3]
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if rows:
            out[tag] = rows
    return out


def series(rows, key):
    return [r["epoch"] + 1 for r in rows], [r[key] for r in rows]


def curves(ax_acc, ax_loss, rows, label, both=True):
    line, = ax_acc.plot(*series(rows, "val_acc"), label=f"{label} val")
    if both:
        ax_acc.plot(*series(rows, "train_acc"), "--", color=line.get_color(), label=f"{label} train")
    ax_loss.plot(*series(rows, "val_loss"), color=line.get_color(), label=f"{label} val")
    if both:
        ax_loss.plot(*series(rows, "train_loss"), "--", color=line.get_color(), label=f"{label} train")


def figure(runs, tags, labels, title, path, both=True):
    tags = [(t, l) for t, l in zip(tags, labels) if t in runs]
    if not tags:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for t, l in tags:
        curves(a, b, runs[t], l, both)
    a.set(xlabel="epoch", ylabel="accuracy (%)", title=f"{title}: accuracy")
    b.set(xlabel="epoch", ylabel="cross-entropy loss", title=f"{title}: loss")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.runs)

    summary = {}
    for t, rows in sorted(runs.items()):
        summary[t] = {"epochs": len(rows), "best_val_acc": max(r["val_acc"] for r in rows),
                      "final_val_acc": rows[-1]["val_acc"], "final_train_acc": rows[-1]["train_acc"],
                      "final_train_loss": rows[-1]["train_loss"], "final_val_loss": rows[-1]["val_loss"],
                      "train_time_s": sum(r["epoch_time"] for r in rows),
                      "mean_epoch_s": sum(r["epoch_time"] for r in rows) / len(rows),
                      "max_mem_mb": max(r["max_mem_mb"] for r in rows), "n_params": rows[-1]["n_params"]}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    figure(runs, ["lenet", "alexnet_base"], ["LeNet", "AlexNet"], "AlexNet vs LeNet (CIFAR-10)",
           os.path.join(args.out, "q2_1_alexnet_vs_lenet.png"))
    lrs = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]
    figure(runs, [f"lr_{x}" for x in lrs], [f"lr {x}" for x in lrs], "AlexNet LR sweep",
           os.path.join(args.out, "q2_2_lr_sweep.png"))
    bss = ["32", "64", "100", "256", "500", "1024"]
    figure(runs, [f"bs_{x}" for x in bss], [f"bs {x}" for x in bss], "AlexNet batch-size sweep (lr 3e-4)",
           os.path.join(args.out, "q2_3_bs_sweep.png"))
    bs_tags = [(x, f"bs_{x}") for x in bss if f"bs_{x}" in runs]
    if bs_tags:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([x for x, _ in bs_tags], [summary[t]["train_time_s"] / 60 for _, t in bs_tags])
        ax.set(xlabel="batch size", ylabel="total training time (min)", title="AlexNet training time, 20 epochs")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "q2_3_bs_time.png"), dpi=130)
        plt.close(fig)
    figure(runs, ["alexnet_base", "aug_none", "aug_crop_flip", "aug_trivial_erase"],
           ["jitter+flip (baseline)", "no aug", "crop+flip", "TrivialAugment+flip+erasing"],
           "AlexNet augmentation schemes", os.path.join(args.out, "q2_5_augmentations.png"))
    figure(runs, ["alexnet_base", "opt_sgd", "opt_wd0.1"], ["AdamW wd 0.01 (baseline)", "SGD momentum", "AdamW wd 0.1"],
           "AlexNet optimizer / weight decay", os.path.join(args.out, "q2_6_optional.png"))
    figure(runs, ["resnet18", "alexnet_base"], ["ResNet18", "AlexNet"], "ResNet18 vs AlexNet (CIFAR-10)",
           os.path.join(args.out, "q3_1_resnet18.png"))

    tp_path = os.path.join(args.runs, "throughput.json")
    if os.path.exists(tp_path):
        tp = json.load(open(tp_path))
        rows = [r for r in tp["results"] if r.get("img_per_s")]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["batch_size"] for r in rows], [r["img_per_s"] for r in rows], "o-")
        ax.set_xscale("log", base=2)
        ax.set(xlabel="batch size", ylabel="training throughput (images/s)",
               title=f"AlexNet training throughput, CIFAR-10 at 70px ({tp['gpu']})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "q2_4_throughput.png"), dpi=130)
        plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
