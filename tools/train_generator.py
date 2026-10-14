"""Train the BeyondFreq blueprint generator from SD1.5, or continue a saved generator."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from beyond_freq.dataset import CocoBlueprintDataset, PreparedBlueprintDataset
from beyond_freq.training import configure_control_training, load_training_model
from beyond_freq.training_callbacks import BlueprintImageLogger, TrainingConfigLogger

RECIPE = Path(__file__).resolve().parent.parent / 'configs/training/coco512.yaml'


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--train-config', type=Path, default=RECIPE)
    configuration, _ = pre.parse_known_args()
    defaults = OmegaConf.to_container(OmegaConf.load(configuration.train_config), resolve=True)
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--weights', type=Path, help='Initialized or fine-tuned BeyondFreq model')
    initialization.add_argument('--sd-weights', type=Path, help='Base SD1.5 checkpoint in CompVis/LDM format')
    initialization.add_argument('--resume', type=Path, help='Full Lightning checkpoint: model + optimizer + step')
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--prompts', type=Path, help='coco_prompts_train2017.txt from prepare_coco.py')
    data.add_argument('--annotations', type=Path, help='Alternative: render non-crowd layouts on the fly')
    parser.add_argument('--blueprints', type=Path, help='Prepared training PNG directory (with --prompts)')
    parser.add_argument('--images', type=Path, required=True)
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--val-prompts', type=Path)
    validation.add_argument('--val-annotations', type=Path)
    parser.add_argument('--val-blueprints', type=Path)
    parser.add_argument('--val-images', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--devices', type=int)
    parser.add_argument('--max-steps', type=int, help='Total optimizer updates; includes restored updates with --resume')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--accumulate', type=int)
    parser.add_argument('--precision', choices=['32', '16', 'bf16'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--size', type=int)
    parser.add_argument('--flip-probability', type=float)
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--preview-every', type=int, help='0 disables sample previews')
    parser.add_argument('--preview-images', type=int)
    parser.add_argument('--preview-steps', type=int)
    parser.add_argument('--preview-guidance', type=float)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    for name in ('batch_size', 'devices', 'max_steps', 'accumulate', 'checkpoint_every', 'size',
                 'preview_images', 'preview_steps'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.size % 8 or args.workers < 0 or args.preview_every < 0 or args.learning_rate <= 0:
        parser.error('Size must be divisible by 8; workers/preview frequency nonnegative; learning rate positive')
    if not 0 <= args.flip_probability <= 1:
        parser.error('--flip-probability must be in [0,1]')
    if bool(args.prompts) != bool(args.blueprints):
        parser.error('--prompts and --blueprints must be supplied together')
    has_val = bool(args.val_prompts or args.val_annotations)
    if has_val != bool(args.val_images) or bool(args.val_prompts) != bool(args.val_blueprints):
        parser.error('Validation requires --val-images plus --val-annotations or --val-prompts/--val-blueprints')
    if args.resume and not args.resume.is_file():
        parser.error('--resume checkpoint does not exist')

    pl.seed_everything(args.seed, workers=True)
    model, initialization_report = load_training_model(args.weights, args.sd_weights)
    configure_control_training(model, args.learning_rate)
    if args.prompts:
        dataset = PreparedBlueprintDataset(args.prompts, args.blueprints, args.images,
                                           args.size, args.flip_probability)
    else:
        dataset = CocoBlueprintDataset(args.annotations, args.images, args.size, args.flip_probability)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = None
    if has_val:
        validation_dataset = (PreparedBlueprintDataset(args.val_prompts, args.val_blueprints, args.val_images,
                                                       args.size, 0.) if args.val_prompts else
                              CocoBlueprintDataset(args.val_annotations, args.val_images, args.size, 0.))
        val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True,
                                persistent_workers=args.workers > 0)
    checkpoint = ModelCheckpoint(dirpath=args.output / 'checkpoints', filename='step-{step:06d}',
                                 every_n_train_steps=args.checkpoint_every, save_top_k=1, save_last=True,
                                 save_weights_only=False, auto_insert_metric_name=False,
                                 save_on_train_epoch_end=False)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update({'initialization': initialization_report, 'training_images': len(dataset),
                     'validation_images': len(validation_dataset) if has_val else 0,
                     'effective_batch_size': args.batch_size * args.devices * args.accumulate,
                     'control_strength': 1., 'trainable_components': ['control_model'],
                     'optimizer': 'AdamW', 'weight_decay': .01})
    callbacks = [checkpoint, BlueprintImageLogger(args.preview_every, args.preview_images,
                                                 args.preview_steps, args.preview_guidance),
                 TrainingConfigLogger(metadata)]
    precision = int(args.precision) if str(args.precision).isdigit() else args.precision
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices,
                        strategy='ddp' if args.devices > 1 else 'auto', precision=precision,
                        max_steps=args.max_steps, accumulate_grad_batches=args.accumulate,
                        callbacks=callbacks, default_root_dir=args.output,
                        logger=CSVLogger(str(args.output), name='metrics'),
                        log_every_n_steps=10, num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader,
                ckpt_path=str(args.resume) if args.resume else None)
    # Preserve the final update even when max_steps is not a checkpoint interval multiple.
    trainer.save_checkpoint(str(args.output / 'checkpoints/last.ckpt'))


if __name__ == '__main__':
    main()
