#!/usr/bin/env bash
# Pod B: the accuracy sweeps run side by side; the batch-size sweep (whose
# training times get reported) runs afterwards one at a time.
set -uo pipefail
cd /root/nmep_model_zhu
R=/root/runs; mkdir -p $R
C=configs/alexnet_base.yaml
D="DATA.DATA_PATH /root/cifar10 DATA.NUM_WORKERS 6"

go() { local tag=$1; shift; python main.py --cfg $C --output $R/$tag --opts $D TAG $tag "$@" > $R/$tag.log 2>&1; echo "== $tag done"; }

for lr in 3e-5 1e-4 3e-4 1e-3 3e-3; do go lr_$lr TRAIN.LR $lr & done
for a in none crop_flip trivial_erase; do go aug_$a AUG.SCHEME $a & done
go opt_sgd TRAIN.OPTIMIZER.NAME sgd TRAIN.LR 1e-2 TRAIN.WEIGHT_DECAY 5e-4 &
go opt_wd0.1 TRAIN.WEIGHT_DECAY 0.1 &
wait

D="DATA.DATA_PATH /root/cifar10 DATA.NUM_WORKERS 16"
for bs in 32 64 100 256 500 1024; do go bs_$bs DATA.BATCH_SIZE $bs TRAIN.EPOCHS 20; done
echo ALL_DONE
