# Kaggle writeup: MediumImageNet

## What I did

I trained convolutional networks from scratch on the 607,425 MediumImageNet
training images (96x96, 200 classes). No pretrained weights, no external data,
and no training on the validation split. Validation was used only to pick the
checkpoint, the test resolution and the ensemble weights.

The final submission averages the predicted probabilities of two models, each
trained on one RTX 5090:

| Model | Params | Epochs | Train time | Val acc (flip TTA, 128px) |
|---|---|---|---|---|
| ResNet-50-D | 23.9M | 14 | 33 min | 93.31% |
| ResNet-34-D | 21.4M | 12 | 15 min | 91.87% |
| Ensemble (weights 0.6 / 0.4) | | | | 93.48% |

Kaggle public score: KAGGLE_SCORE.

## Training recipe (`fast_train.py`)

- Architecture: timm's ResNet-D variants (deep 3-conv stem, avg-pool
  downsampling in the shortcuts). I removed the stem max-pool, because at 96px
  input the standard ImageNet stem would shrink the image to 24x24 before the
  first residual stage. Without it, the stages run at 48, 24, 12 and 6 pixels.
- Data pipeline: the whole split is loaded into RAM as uint8. A background
  thread gathers batches into pinned memory, and all augmentation runs on the
  GPU: a per-image random resized crop (30-100% of the area, aspect 3/4 to
  4/3) with horizontal flip, done in a single `grid_sample`, plus brightness,
  contrast and saturation jitter (+-20%). This kept the GPU at 99% utilization.
- Loss: cross entropy with label smoothing 0.1. I checked for duplicates:
  about 2% of val images are exact pixel copies of a training image, and for
  those the val label matches the train label only 85% of the time, so the
  labels are noisy and smoothing helps.
- Optimizer: SGD with Nesterov momentum 0.9, batch 512, peak LR 0.8, one
  epoch of linear warmup, then cosine decay to zero. Weight decay 5e-5 on
  conv and linear weights only, none on BatchNorm and biases.
- Speed: bf16 autocast, channels_last, `torch.compile`. ResNet-50-D trained at
  3,500 to 6,000 images/s on one RTX 5090 (about 2 minutes per epoch).
- EMA: an exponential moving average of the weights (decay 0.9995). Early in
  training the EMA model was 4 to 8 points better than the raw weights.
- Test time: average the softmax of each image and its mirror image. The
  script evaluates raw and EMA weights at 96, 112 and 128px on val and uses
  the best combination for the test set. Upscaling test images to 128px gave
  +0.8 points over 96px (92.51% to 93.31% for the ResNet-50-D), because
  random resized crops make objects look bigger during training than in
  unaugmented test images. By the end of training the raw weights beat the
  EMA weights (93.31% vs 93.08%), so the raw weights were used.
- Ensemble (`ensemble_preds.py`): grid search over per-model weights on val,
  then the same weights on test.

## What worked

- Short, fast schedules: the ResNet-50-D was at 72% val after 2 epochs (EMA)
  and 85.8% after 6. While the learning rate is high, the EMA weights are
  4 to 8 points better than the raw weights.
- Keeping the data pipeline on the GPU. The provided HDF5 `Dataset` reads one
  image at a time and does its transforms on the CPU, which would have left
  the GPU mostly idle.
- Averaging two models: a small gain (+0.17 points), since both models make
  similar mistakes. A third model was cut short when its cloud machine
  restarted.

## What I skipped and why

- Mixup and CutMix: they slow convergence, and they only pay off on long
  schedules. With about 30 minutes of training per model, I turned them off.
- Pretrained weights and the full-resolution archive: excluded to keep the
  result about training on this dataset.

## What I learned

The 96px images hold a lot more signal than I expected: a ResNet-50 passes
85% validation accuracy within 6 epochs and reaches 93% in 14. The bottleneck on a shared machine is
usually feeding the GPU, not the GPU itself. Label noise in scraped data puts
a ceiling on accuracy that more training cannot remove.
