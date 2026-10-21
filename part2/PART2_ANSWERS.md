# HW 3 Model Zhu, Part 2 answers

All runs: CIFAR-10, one RTX 5090 (32 GB), code in my fork
(github.com/trisanths/nmep_model_zhu). Plots are in `figs/`.

Before training I fixed three bugs in the starter code: `main.py` put the model
on the CPU (`device = "cpu"`) while moving batches to CUDA, the cosine LR
scheduler was created but never stepped, and LeNet's first Linear layer
expected 16*6*6 inputs when a 32x32 image gives 16*8*8. I also set LeNet's
NUM_CLASSES to 10 for CIFAR-10.

## Q1.1

AlexNet (10 classes, 70x70 input) has 57,044,810 parameters. LeNet has
136,886, so AlexNet is about 417x bigger. Most of AlexNet's parameters sit in
the classifier: the first Linear layer alone (9216 -> 4096) has 37.8M. The
flattened vector is 256 * 6 * 6 = 9216 because AdaptiveAvgPool2d((6, 6))
always outputs 6x6.

Memory while training at batch size 256 (peak per process from nvidia-smi,
which is what gpustat shows):

- LeNet: 738 MB (only 60 MB of it is tensors allocated by PyTorch; the rest is the CUDA context and caches)
- AlexNet: 1,938 MB (1,123 MB allocated by PyTorch)

## Q1.2

AlexNet on CIFAR-10: 77.09% best validation accuracy (77.06% at the last
epoch), 99.09% training accuracy.

LeNet on CIFAR-10: 60.16% validation, 61.81% training.

AlexNet config (configs/alexnet_base.yaml): IMG_SIZE 70, BATCH_SIZE 256,
EPOCHS 30, AdamW, LR 3e-4, weight decay 0.01, betas (0.9, 0.999), cosine LR
schedule, augmentation = ColorJitter(0.4, 0.4, 0.4) + RandomHorizontalFlip,
dropout 0.5.

LeNet config (configs/lenet_base.yaml): IMG_SIZE 32, BATCH_SIZE 256, EPOCHS 20,
AdamW, LR 3e-3, weight decay 0.01, cosine schedule, same augmentation.

## Q2.1 (attach figs/q2_1_alexnet_vs_lenet.png)

LeNet underfits. Train and val accuracy stay together around 60% and both
losses are still creeping down at epoch 20. The model (sigmoid activations,
137K parameters) is too small for CIFAR-10.

AlexNet overfits. Training accuracy climbs to 99% and training loss goes to
0.03, but validation accuracy flattens around 75 to 77% after epoch 10.
Validation loss bottoms out at 0.75 at epoch 11 and then rises to 1.31 by
epoch 30, even though validation accuracy keeps inching up. The model gets
more confident on the examples it gets wrong. Early stopping around epoch 11
to 15, or more augmentation, would help.

## Q2.2 (attach figs/q2_2_lr_sweep.png)

AlexNet, 30 epochs each, everything else fixed (AdamW, batch 256):

| LR | best val acc | final train acc |
|---|---|---|
| 3e-5 | 69.79% | 72.72% |
| 1e-4 | 76.01% | 94.90% |
| 3e-4 | 76.62% | 99.16% |
| 1e-3 | 69.91% | 78.11% |
| 3e-3 | 35.68% | 34.38% |

The best learning rate is 3e-4, with 1e-4 close behind. 3e-5 is too slow: the
loss is still dropping at epoch 30 and the model never fits the training set.
1e-3 trains noisily and ends 7 points lower. At 3e-3 training barely works
(35%), because the updates are too large for Adam on this network. The higher
the LR (up to 3e-4), the faster the training loss falls and the earlier the
validation loss starts rising again, so the best LR also overfits the most.

## Q2.3 (attach figs/q2_3_bs_sweep.png and figs/q2_3_bs_time.png)

AlexNet, LR fixed at 3e-4, 20 epochs, each run alone on the GPU:

| batch size | best val acc | final train acc | total train time | time per epoch |
|---|---|---|---|---|
| 32 | 75.15% | 89.02% | 189.0 s | 9.45 s |
| 64 | 75.96% | 93.64% | 100.3 s | 5.01 s |
| 100 | 76.42% | 94.27% | 67.6 s | 3.38 s |
| 256 | 76.29% | 94.54% | 51.0 s | 2.55 s |
| 500 | 76.24% | 90.15% | 52.5 s | 2.63 s |
| 1024 | 73.50% | 84.42% | 55.4 s | 2.77 s |

Small batches are slow: batch 32 takes 3.7x as long as batch 256 for the same
20 epochs, because each step has a fixed overhead and there are 8x more
steps. Past 256 the time stops improving (the data loader becomes the limit).
With the LR held fixed, large batches mean fewer updates per epoch, so batch
1024 underfits (84% train, 73.5% val). Batch sizes 100 to 500 give about the
same accuracy, and the odd sizes (100, 500) behave just like the nearby powers
of two; nothing about them is special beyond slightly less even GPU work.

## Q2.4 (attach figs/q2_4_throughput.png)

Throughput = images per epoch (50,000) / time per epoch. Timed over 1 epoch
for batch sizes up to 16 and averaged over 3 epochs above that, after a short
warmup:

| batch | img/s | | batch | img/s |
|---|---|---|---|---|
| 1 | 264 | | 256 | 19,719 |
| 2 | 523 | | 512 | 17,921 |
| 4 | 1,028 | | 1024 | 16,354 |
| 8 | 2,052 | | 2048 | 16,864 |
| 16 | 4,018 | | 4096 | 13,155 |
| 32 | 7,817 | | 8192 | 8,840 |
| 64 | 14,664 | | 16384 | 5,449 |
| 128 | 19,196 | | 32768 | 3,023 |

The largest batch I could train is 65,536 (a single training step on random
data used 24.7 GB). 131,072 hit CUDA OOM on the 32 GB card. Through the real
data loader the largest batch is 32,768, since CIFAR-10 only has 50,000
training images.

Trends: from 1 to 64, throughput doubles every time the batch doubles. At
small batches the GPU is mostly idle, and the time per step is a fixed cost
(Python, kernel launches, and the AdamW update over 57M parameters) that does
not depend on batch size. Throughput peaks at about 19,700 img/s at batch
128 to 256. There the bottleneck moves to the CPU data loader, which has to
decode, jitter and resize every image. Past 512 it drops: each worker has to
build one huge batch before the GPU can start, fewer batches per epoch means
less overlap between loading and compute, and memory grows linearly with the
batch.

## Q2.5 (attach figs/q2_5_augmentations.png)

AlexNet, 30 epochs, LR 3e-4, batch 256:

| augmentation | best val acc | final train acc |
|---|---|---|
| none | 74.59% | 100.00% |
| ColorJitter + flip (baseline) | 77.09% | 99.09% |
| RandomCrop(32, pad 4) + flip | 80.83% | 87.15% |
| TrivialAugmentWide + flip + RandomErasing(0.25) | 77.57% | 67.99% |

With no augmentation the model memorizes the training set (100% train,
validation loss up to 2.30). Random crop + flip was the best scheme: +3.7
points over the baseline and a much smaller train/val gap. TrivialAugment +
erasing regularizes so hard that the model is still underfitting after 30
epochs (68% train, below its 77.6% val), and validation accuracy was still
rising at the end, so it would likely win with a longer schedule.

## Q2.6 (optional) (figs/q2_6_optional.png)

- SGD (momentum 0.9, Nesterov, LR 1e-2, weight decay 5e-4) instead of AdamW:
  79.67% val vs 77.09%, with less overfitting (96% train).
- AdamW with weight decay 0.1 instead of 0.01: 77.26%, about the same as the
  baseline.

## Q3.1 (attach figs/q3_1_resnet18.png)

ResNet18 (11,173,962 parameters) on CIFAR-10 at 32x32: 89.65% best validation
accuracy (also the final epoch), 99.96% training accuracy.

Hyperparameters (configs/resnet18_base.yaml): BATCH_SIZE 256, EPOCHS 30, AdamW,
LR 1e-3, weight decay 0.05, cosine schedule, ColorJitter + flip.

Training time: 2 min 36 s for 30 epochs (4.4 s per epoch) vs 1 min 50 s for
AlexNet's 30 epochs (2.8 s per epoch) on the same GPU. ResNet18 has 5x fewer
parameters than AlexNet but needs about 6x more compute per image (559M vs
95M multiply-adds), because it keeps a 32x32 feature map through its first
stage, while AlexNet shrinks the image 4x in its first conv. It is 12.6 points
more accurate. Validation loss bottoms out at 0.45 around epoch 14 and stays
flat instead of climbing like AlexNet's.

## Q3.2 (optional) (attach figs/q3_2_resnet18_preds.png)

20 random validation images with true and predicted labels: the model got all
20 right.
