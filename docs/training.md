# Training the COCO512 visual blueprint generator

This restores the original generator training workflow in the `beyond_freq` package:
COCO preparation → SD1.5 initialization → ControlNet training → checkpoint export.
Use the Python 3.10 / Torch 2.5.1 / Lightning 1.9.5 environment in the main README.

## 1. Prepare COCO first

Arrange COCO2017 as follows; symlinks to your existing image directories also work:

```text
datasets/coco/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  images/train2017/*.jpg
  images/val2017/*.jpg
```

Run the recovered data preparation entry point:

```bash
python tools/prepare_coco.py --coco-root datasets/coco
```

It writes original-resolution, opaque class-color blueprints to
`images/80_colors/train2017/` and `images/80_colors/val2017/`, and creates
`annotations/coco_prompts_train2017.txt` / `coco_prompts_val2017.txt`.
Each line is `000000391895.png;a photograph with ...`.
`--output PATH` instead places each split's PNG directory and prompt manifest under PATH.
`--splits train2017` prepares only training data.

As in the original `prepare_coco.py`, a training image containing **any crowd annotation**
is excluded entirely, and images without annotations are excluded. The recovered training
manifest contains **108,151 images**. Validation retains all images and crowd annotations;
empty validation layouts use the valid prompt `a photograph`. A JSON beside each manifest
records prepared and excluded image counts.

Category order, integer box endpoints, large-to-small painting order, instance brightness,
and object-count prompts are preserved. The dataset resizes both prepared PNG and target JPEG
to 512×512, applies the same random horizontal flip to both, and returns target pixels in
[-1,1] and conditioning pixels in [0,1]. Preparation uses the opaque renderer associated
with the released checkpoint, rather than the separate transparency variant.

## 2. Initialize from SD1.5

Obtain a **CompVis/LDM-format Stable Diffusion v1.5 checkpoint** (for example,
`v1-5-pruned.ckpt` or `v1-5-pruned-emaonly.ckpt`) under its model license, and place it
in `checkpoints/`. This interface expects the UNet, VAE and CLIP weights together;
it does not accept a Diffusers directory or an isolated UNet.

```bash
python tools/initialize_generator.py \
  --sd-weights checkpoints/v1-5-pruned.ckpt \
  --output checkpoints/beyondfreq_sd15_initialized.ckpt --seed 23
```

Initialization strictly checks all backbone parameters and tensor shapes. The ControlNet
time embedding, input blocks and middle block are copied from the SD1.5 UNet; its hint
encoder and zero-convolution layers retain their architecture initialization. Missing
backbone tensors fail immediately. An already initialized generator must instead be loaded
with `--weights`.

## 3. Train the original generator

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_generator.py \
  --weights checkpoints/beyondfreq_sd15_initialized.ckpt \
  --prompts datasets/coco/annotations/coco_prompts_train2017.txt \
  --blueprints datasets/coco/images/80_colors/train2017 \
  --images datasets/coco/images/train2017 \
  --val-prompts datasets/coco/annotations/coco_prompts_val2017.txt \
  --val-blueprints datasets/coco/images/80_colors/val2017 \
  --val-images datasets/coco/images/val2017 \
  --output logs/beyondfreq_coco512 --devices 4 --batch-size 4 --max-steps 100000
```

Alternatively use `--sd-weights checkpoints/v1-5-pruned.ckpt` in the training command
to initialize in memory, omitting the separate initialization step.

The defaults are in [`configs/training/coco512.yaml`](../configs/training/coco512.yaml).
Only the **ControlNet** is optimized; UNet, VAE and CLIP are frozen. The original objective
is diffusion epsilon-prediction MSE, with uniformly sampled timesteps among 1,000 noise
levels. AdamW uses learning rate 5e-6, default betas (0.9,0.999) and weight decay 0.01,
without a learning-rate scheduler. Training control strength is **1.0**; inference's
default 1.5 is separate. The original recipe uses FP32, seed 23, batch 4 per GPU,
4 GPUs, and no accumulation (effective batch 16).

For two GPUs with the same effective batch size, set
`CUDA_VISIBLE_DEVICES=0,1 --devices 2 --batch-size 4 --accumulate 2`.
For one GPU use `--devices 1 --batch-size 4 --accumulate 4`.
These changes preserve effective batch size, but do not guarantee the identical optimization
trajectory or throughput. `--workers` sets workers per process; `--precision bf16` is an
optional memory-saving change to the original FP32 recipe.

Validation noise losses are evaluated at the end of each training epoch when validation
paths are supplied. CSV metrics and each invocation's resolved training configuration are
saved under `OUTPUT/metrics/`. Every 5,000 optimizer updates, `OUTPUT/image_log/train/`
receives reconstruction, conditioning and 50-step DDIM sample previews (guidance 9.0,
up to 4 images). Set `--preview-every 0` to disable previews. These previews are not a
full FID/AP benchmark.

## 4. Save, resume and export

Checkpoints include model weights, optimizer states, loop state, epoch and global step.
The newest periodic checkpoint and `checkpoints/last.ckpt` are kept; the final update is
saved even if it falls between periodic checkpoints. Allow approximately 9 GB per full
FP32 training checkpoint, plus transient space while replacing an older one.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_generator.py \
  --resume logs/beyondfreq_coco512/checkpoints/last.ckpt \
  --prompts datasets/coco/annotations/coco_prompts_train2017.txt \
  --blueprints datasets/coco/images/80_colors/train2017 \
  --images datasets/coco/images/train2017 \
  --val-prompts datasets/coco/annotations/coco_prompts_val2017.txt \
  --val-blueprints datasets/coco/images/80_colors/val2017 \
  --val-images datasets/coco/images/val2017 \
  --output logs/beyondfreq_coco512 --devices 4 --max-steps 100000
```

`--resume` is used on its own, without `--weights` / `--sd-weights`; `--max-steps` is
the total update target, including restored steps. Lightning restores optimizer state
and loop counters, but this CLI does not promise bitwise replay of the original shuffled
data and worker RNG stream after interruption. Conversely, `--weights` loads only model
tensors and starts fresh optimizer/counters, suitable for fine-tuning.

The existing `--annotations ... --images ...` input remains available for on-the-fly
fine-tuning. It filters individual crowd annotations and retains all images, so it is a
different data selection from the prepared original-training manifest.

Export a trained checkpoint with the lossless weight exporter, or pass the checkpoint
directly to `tools/generate.py`:

```bash
python tools/export_weights.py \
  --checkpoint logs/beyondfreq_coco512/checkpoints/last.ckpt \
  --output release_assets/trained_model

CUDA_VISIBLE_DEVICES=0 python tools/generate.py \
  --weights logs/beyondfreq_coco512/checkpoints/last.ckpt \
  --layout-json examples/layout.json --output outputs/trained_example
```

Exported shard names use the saved global step; `--prefix` can override the filename stem.
Model-only checkpoints without a global step use a neutral `trained` suffix. The original
100k naming remains for checkpoints with exactly 100,000 saved updates. When publishing,
record the actual training data, sampling protocol and evaluation results for the new model.

## Scope and provenance

The recovered source training entry uses the same ControlNet-only recipe, but its current
settings load a 30k checkpoint and request 80k updates with fresh optimizer state. Its filename
alone cannot establish the exact history of the released 100k weights. This guide provides
a complete fresh-run pipeline targeting 100k updates; it does not claim that rerunning it
reproduces the published checkpoint or paper metrics exactly. Generator training is separate
from the paper's layout planner and detector alignment components.

Recovered source hashes and validation results are recorded in
[`training_workflow_validation.json`](training_workflow_validation.json). The validation
covers preparation parity, initialization, short GPU training and checkpoint resume;
a new full 100k training run is not part of this code release.
