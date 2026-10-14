"""Initialize BeyondFreq's blueprint ControlNet from a CompVis-format SD1.5 checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
import torch
from beyond_freq.training import load_training_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sd-weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.output.suffix not in {'.ckpt', '.pt', '.pth'}:
        parser.error('--output must end in .ckpt, .pt or .pth')
    pl.seed_everything(args.seed, workers=True)
    model, report = load_training_model(sd_weights=args.sd_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'initialization': report, 'seed': args.seed}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), **report}), flush=True)


if __name__ == '__main__':
    main()
