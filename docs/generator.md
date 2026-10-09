# BeyondFreq-L2I generator

This is the visual blueprint generator associated with §3.2 of the ICLR2026 paper. The public entry points are `beyond_freq.generator.BeyondFreqL2I`, `tools/generate.py`, and `tools/train_generator.py`.

## Checkpoint-compatible blueprints

The COCO512 checkpoint expects its original class-color encoding: preserve the order of the80 COCO categories, use HSV hue `(index+1)/80`, saturation1, and decrement brightness by0.02 for repeated instances. Rectangles are drawn with integer OpenCV endpoints, from large annotation areas to small areas, on a black canvas. Rendering occurs in the original image coordinates and is resized to512×512.

For the supplied checkpoint, rectangle fills remain opaque and the painting order follows COCO annotation `area` when available. These are the existing implementation conventions. The paper discusses additional overlap handling, including transparency; changing that convention would change the pretrained model's input distribution. Renaming the package and checkpoint does not modify the encoding or the learned weights.

Object-count text prompts follow `a photograph with ...`; they complement the RGB blueprint. The generator has no fixed18-box limit. CLIP still has a77-token text limit.

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python tools/generate.py \
  --weights checkpoints/beyondfreq_l2i_coco512_100k \
  --blueprint examples/blueprint.png \
  --prompt "$(cat examples/prompt.txt)" \
  --output outputs/example --seed 451
```

Alternatively, pass `--hf-repo Strike1999/BeyondFreq-L2I-COCO512` and optionally `--revision COMMIT_SHA`. COCO-format layout input is demonstrated in the main README. An image's initial noise uses `seed+image_id*1009`, so unfinished batches can resume without regenerating completed PNGs. The default CUDA denoiser precision is BF16; weights are stored at their original precision. Use `--precision fp32` for FP32 denoising.

## Generator fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_generator.py \
  --weights checkpoints/beyondfreq_l2i_coco512_100k \
  --annotations datasets/coco/annotations/instances_train2017.json \
  --images datasets/coco/train2017 \
  --output logs/beyondfreq_l2i_coco512 \
  --batch-size 8 --learning-rate 5e-6 --max-steps 100000
```

This command fine-tunes the ControlNet using the diffusion noise-prediction loss, with the backbone, VAE and CLIP frozen. It uses paired horizontal flips and filters crowd annotations. It starts from the supplied checkpoint; the requested steps are additional optimization steps. `--resume` restores a Lightning training checkpoint, including optimizer states. `--devices 2` together with `CUDA_VISIBLE_DEVICES=0,1` enables two-GPU DDP.

The model wrapper also exposes `predict_noise`, `set_trainable(control=True, unet=True)` and `sample(..., differentiable=True)` for downstream research. These APIs permit generator gradients; they do not implement a detector loss, representation-score planner or the paper's complete detector training procedure.

## Weight export

```bash
python tools/export_weights.py --checkpoint YOUR_CHECKPOINT.ckpt --output release_assets/model
```

The export preserves tensor names, values and dtypes, verifies every tensor, and writes three shards for the100k checkpoint, an index, `weights_manifest.json` and `SHA256SUMS`. An original `.ckpt` can also be passed directly to the loader.

## Evaluation provenance

The released model is the original100k generator checkpoint, before the ongoing G→D cycle fine-tuning experiments. A separate5000-image evaluation used COCO val2017, seed451, 512×512, DDIM50, guidance4.5, blueprint strength1.5, shared non-crowd layouts capped at30 objects, frozen DEIM and independent YOLOv8x. It obtained DEIM layout AP52.50, YOLO layout AP48.46, Clean-FID5K25.16, KID0.005285 and CLIPScore27.56. The corresponding machine-readable metadata is in `docs/coco512_evaluation.json`.

This evaluation uses a different detector and protocol from the paper's tables. It is recorded as a checkpoint verification benchmark, without claiming reproduction of those tables.
