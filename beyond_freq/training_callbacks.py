"""Local sample previews for the recovered generator training workflow."""
from pathlib import Path
import json
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid


class TrainingConfigLogger(Callback):
    def __init__(self, configuration):
        self.configuration = configuration

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # DDP has initialized ranks and synchronized the logger version at this point.
            _ = trainer.logger.experiment
            path = Path(trainer.logger.log_dir) / 'training_config.json'
            path.write_text(json.dumps(self.configuration, indent=2) + '\n', encoding='utf-8')


class BlueprintImageLogger(Callback):
    def __init__(self, every=5000, images=4, steps=50, guidance=9.):
        self.every, self.images, self.steps, self.guidance = every, images, steps, guidance
        self.last_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not self.every or step < 1 or step % self.every or step == self.last_step:
            return
        self.last_step = step
        # All ranks synchronize: preview inference on rank zero may be slower than a batch.
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            was_training = pl_module.training
            pl_module.eval()
            try:
                with torch.no_grad():
                    images = pl_module.log_images(batch, N=self.images, sample=True,
                                                  ddim_steps=self.steps,
                                                  unconditional_guidance_scale=self.guidance)
                destination = Path(trainer.default_root_dir) / 'image_log/train'
                destination.mkdir(parents=True, exist_ok=True)
                for name, pixels in images.items():
                    grid = make_grid(pixels[:self.images].detach().cpu().clamp(-1, 1), nrow=self.images)
                    array = ((grid + 1) * 127.5).permute(1, 2, 0).numpy().astype('uint8')
                    Image.fromarray(array).save(destination / f'{name}_step-{step:06d}.png')
            finally:
                pl_module.train(was_training)
        trainer.strategy.barrier()
