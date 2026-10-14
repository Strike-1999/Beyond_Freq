# Unbiased Object Detection Beyond Frequency with Visually Prompted Image Synthesis (ICLR2026)

## Introduction 👋

Official repository for **Unbiased Object Detection Beyond Frequency with Visually Prompted Image Synthesis**, ICLR 2026.

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/3edde8c8a05871c0172a6186a7d4850d-Paper-Conference.pdf) · [Generator release guide](docs/generator.md) · [Training guide](docs/training.md)

This release provides **BeyondFreq-L2I**, the visual blueprint generator, with the COCO512 checkpoint trained for 100,000 updates. It includes COCO blueprint rendering, training and inference commands, and losslessly exported weights. The package name is `beyond_freq`. The representation-score layout planner and detector alignment training are separate components of the full paper; this release covers the generator.

<img src="assets/framework.png" alt="Pipeline" width="100%"/>

## Setup

Use Python3.10 and a CUDA GPU. The release is validated with Torch2.5.1 and CUDA12.1.

```bash
conda create -n beyondfreq python=3.10 -y
conda activate beyondfreq
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Pretrained generator

| Model | Data | Resolution | Updates | Format |
|---|---|---|---|---|
| BeyondFreq-L2I | COCO | 512×512 | 100,000 | 3 safetensors shards, original precision |

The pretrained generator is available at [Strike1999/BeyondFreq-L2I-COCO512](https://huggingface.co/Strike1999/BeyondFreq-L2I-COCO512). To download a model snapshot:

```bash
python tools/download_weights.py --repo Strike1999/BeyondFreq-L2I-COCO512
```

The model includes the U-Net, ControlNet, VAE and CLIP text weights. Tokenizer assets and the CLIP configuration are bundled in this code repository. The loader checks shard hashes and loads the model strictly.

## Generate from layouts

Pass a COCO-format JSON containing `images`, `categories` and `annotations`. Boxes use pixel-space `[x,y,width,height]` in the corresponding image coordinates.

Try the bundled example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/generate.py \
  --weights checkpoints/beyondfreq_l2i_coco512_100k \
  --layout-json examples/layout.json --output outputs/example --batch-size 1
```

Generate a full COCO split:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/generate.py \
  --weights checkpoints/beyondfreq_l2i_coco512_100k \
  --layout-json datasets/coco/annotations/instances_val2017.json \
  --output outputs/coco512 --batch-size 4 --seed 451
```

Default sampling uses 50 DDIM steps, guidance4.5 and blueprint strength1.5. All supplied non-crowd boxes are used unless `--max-boxes` is set. See the [generator guide](docs/generator.md) for a prepared-blueprint input and training.

## Train the generator

Start with the original COCO preparation step:

```bash
python tools/prepare_coco.py --coco-root datasets/coco
python tools/initialize_generator.py \
  --sd-weights checkpoints/v1-5-pruned.ckpt \
  --output checkpoints/beyondfreq_sd15_initialized.ckpt
```

Then train using the generated blueprint PNGs and prompt manifests. The [training guide](docs/training.md)
contains the complete training command, four-GPU recipe, one/two-GPU alternatives,
validation, sample previews and optimizer-state resume. The original generator trains
ControlNet while freezing UNet, VAE and CLIP. This also supports fine-tuning the released
checkpoint, using `--weights` instead of SD1.5 initialization.

## Acknowledgements and licenses

The generator uses Stable Diffusion and a ControlNet architecture. Third-party implementation notices and license texts are preserved in [NOTICE](NOTICE) and `licenses/`. Model weights retain the Stable Diffusion OpenRAIL-M license. Existing checkpoint-compatible rendering conventions are documented in the generator guide.

## Citation 📚

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{cai2026unbiased,
  title={Unbiased Object Detection Beyond Frequency with Visually Prompted Image Synthesis},
  author={Cai, Xinhao and Li, Liulei and Pei, Gensheng and Chen, Tao and Pan, Jinshan and Yao, Yazhou and Wang, Wenguan},
  booktitle=ICLR,
  year={2026}
}
```
