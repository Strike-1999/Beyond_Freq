"""Generate images from COCO-format layouts or an existing RGB blueprint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import torch
from beyond_freq.blueprints import BlueprintRenderer
from beyond_freq.generator import BeyondFreqL2I


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--weights', type=Path, help='Checkpoint, safetensors index, or model directory')
    weights.add_argument('--hf-repo', help='Hugging Face model repository')
    parser.add_argument('--revision', help='HF commit or tag')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--layout-json', type=Path, help='COCO-format layout with images/categories/annotations')
    inputs.add_argument('--blueprint', type=Path, help='RGB blueprint rendered with the COCO class palette')
    parser.add_argument('--prompt', default='a photograph')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=451)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg-scale', type=float, default=4.5)
    parser.add_argument('--strength', type=float, default=1.5)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--max-boxes', type=int, default=0, help='0 preserves all supplied boxes')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    generator = BeyondFreqL2I.from_pretrained(args.hf_repo, revision=args.revision, device=args.device) if args.hf_repo else BeyondFreqL2I(args.weights, device=args.device)
    if args.layout_json:
        layout = json.loads(args.layout_json.read_text())
        renderer = BlueprintRenderer(layout['categories'])
        annotations = {i['id']: [] for i in layout['images']}
        for annotation in layout['annotations']:
            if not annotation.get('iscrowd', False):
                annotations[annotation['image_id']].append(annotation)
        rows = []
        for image in layout['images']:
            anns = annotations[image['id']]
            if args.max_boxes:
                anns = sorted(anns, key=lambda a: -a['bbox'][2] * a['bbox'][3])[:args.max_boxes]
            rows.append((image, anns))
    else:
        rows = [({'id': 0}, None)]
    for start in range(0, len(rows), args.batch_size):
        batch = [(im, anns) for im, anns in rows[start:start + args.batch_size]
                 if not (args.output / f"{im['id']:012d}.png").exists()]
        if not batch:
            continue
        ids, blueprints, prompts = [], [], []
        for image, anns in batch:
            if anns is None:
                blueprint = np.asarray(Image.open(args.blueprint).convert('RGB').resize((512, 512)))
                prompt = args.prompt
            else:
                blueprint = renderer.render(anns, image['width'], image['height'])
                prompt = renderer.prompt(anns)
            ids.append(image['id']); blueprints.append(blueprint); prompts.append(prompt)
        tensor = torch.from_numpy(np.stack(blueprints)).permute(0, 3, 1, 2).float() / 255
        images = generator.sample(tensor, prompts, ids, seed=args.seed, steps=args.steps,
                                  cfg_scale=args.cfg_scale, strength=args.strength, precision=args.precision)
        for image_id, image in zip(ids, images):
            path = args.output / f'{image_id:012d}.png'
            temporary = path.with_suffix('.png.tmp')
            array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(array).save(temporary, format='PNG'); temporary.replace(path)
        print(json.dumps({'completed': min(start + args.batch_size, len(rows)), 'total': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
