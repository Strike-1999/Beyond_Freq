"""Prepare COCO blueprint PNGs and prompts using the original generator recipe."""
# Adapted from the original training data preparation; third-party notices: NOTICE.
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tqdm import tqdm
from beyond_freq.blueprints import BlueprintRenderer


def prepare_split(annotation_file, blueprint_dir, prompt_file, training):
    coco = json.loads(Path(annotation_file).read_text(encoding='utf-8'))
    renderer = BlueprintRenderer(coco['categories'])
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)
    blueprint_dir, prompt_file = Path(blueprint_dir), Path(prompt_file)
    blueprint_dir.mkdir(parents=True, exist_ok=True)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = prompt_file.with_suffix(prompt_file.suffix + '.tmp')
    counts = {'images_in_json': len(coco['images']), 'prepared': 0,
              'excluded_crowd_images': 0, 'excluded_empty_images': 0}
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for info in tqdm(coco['images'], desc=Path(annotation_file).stem):
            anns = annotations[info['id']]
            # The original recipe excludes the WHOLE training image, not just its crowd boxes.
            if training and any(a.get('iscrowd', False) for a in anns):
                counts['excluded_crowd_images'] += 1
                continue
            if training and not anns:
                counts['excluded_empty_images'] += 1
                continue
            name = Path(info['file_name']).with_suffix('.png')
            if name.is_absolute() or '..' in name.parts:
                raise ValueError(f'Unsafe COCO filename: {name}')
            canvas = renderer.render_native(anns, info['width'], info['height'])
            destination = blueprint_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
                raise OSError(f'Could not write {destination}')
            # Preserve the original two-pass area ordering, including tied annotation areas.
            order = np.argsort([a.get('area', a['bbox'][2] * a['bbox'][3]) for a in anns])[::-1]
            prompt = renderer.prompt([anns[i] for i in order])
            stream.write(f'{name.as_posix()};{prompt}\n')
            counts['prepared'] += 1
    temporary.replace(prompt_file)
    prompt_file.with_suffix('.json').write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, required=True,
                        help='Contains annotations/instances_{split}.json and images/{split}/')
    parser.add_argument('--splits', nargs='+', choices=['train2017', 'val2017'],
                        default=['train2017', 'val2017'])
    parser.add_argument('--output', type=Path,
                        help='Optional output root: {split}/ PNGs and coco_prompts_{split}.txt')
    args = parser.parse_args()
    for split in args.splits:
        blueprint_dir = (args.output / split if args.output else args.coco_root / 'images/80_colors' / split)
        prompt_file = (args.output if args.output else args.coco_root / 'annotations') / f'coco_prompts_{split}.txt'
        counts = prepare_split(args.coco_root / 'annotations' / f'instances_{split}.json',
                               blueprint_dir, prompt_file, training=split == 'train2017')
        print(json.dumps({'split': split, **counts}), flush=True)


if __name__ == '__main__':
    main()
