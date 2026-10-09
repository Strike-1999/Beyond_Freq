"""Fine-tune the blueprint ControlNet with the original generator noise loss."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from beyond_freq.dataset import CocoBlueprintDataset
from beyond_freq.generator import BeyondFreqL2I


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--resume', type=Path, help='Lightning training checkpoint with optimizer states')
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    generator = BeyondFreqL2I(args.weights, device='cpu')
    generator.set_trainable(control=True, unet=False)
    model = generator.model
    model.sd_locked = True
    model.learning_rate = args.learning_rate
    dataset = CocoBlueprintDataset(args.annotations, args.images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    checkpoint = ModelCheckpoint(dirpath=args.output / 'checkpoints', filename='step-{step}',
                  every_n_train_steps=args.checkpoint_every, save_top_k=-1, save_last=True,
                  save_weights_only=False, auto_insert_metric_name=False)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices,
                        strategy='ddp' if args.devices > 1 else 'auto', precision=32,
                        max_steps=args.max_steps, accumulate_grad_batches=args.accumulate,
                        callbacks=[checkpoint], default_root_dir=args.output,
                        log_every_n_steps=10, num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=loader, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == '__main__':
    main()
