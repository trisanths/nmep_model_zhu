"""Combine several fast_train.py runs into one Kaggle submission.

Each run dir holds val_logp.npy and test_logp.npy (log-probabilities, flip
TTA already applied). Searches a weight per run on validation accuracy, then
applies the same weights to the test predictions and writes preds.npy and
predictions.csv into --out.

    python ensemble_preds.py runs/r50d runs/r34d --out runs/ensemble
"""
import argparse
import itertools
import json
import os

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", default="/data/nmep/cv/medium-imagenet-96.hdf5")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with h5py.File(args.data, "r") as f:
        yva = f["labels-val"][:].astype(np.int64)
    vals = [np.exp(np.load(os.path.join(r, "val_logp.npy")).astype(np.float32)) for r in args.runs]
    tests = [np.exp(np.load(os.path.join(r, "test_logp.npy")).astype(np.float32)) for r in args.runs]
    for r, v in zip(args.runs, vals):
        print(f"{r}: val {100 * (v.argmax(1) == yva).mean():.2f}")

    grid = np.linspace(0, 1, 11)
    best = None
    for w in itertools.product(grid, repeat=len(vals)):
        if not np.isclose(sum(w), 1):
            continue
        a = (sum(wi * v for wi, v in zip(w, vals)).argmax(1) == yva).mean()
        if best is None or a > best[0]:
            best = (a, w)
    a, w = best
    test = sum(wi * t for wi, t in zip(w, tests))
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "preds.npy"), test)
    preds = test.argmax(1)
    np.savetxt(os.path.join(args.out, "predictions.csv"), np.column_stack((np.arange(len(preds)), preds)),
               fmt="%d", delimiter=",", header="Id,Category", comments="")
    summary = {"runs": args.runs, "weights": [round(float(x), 2) for x in w], "val_acc": round(100 * float(a), 3)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
