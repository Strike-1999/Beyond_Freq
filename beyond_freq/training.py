"""SD1.5 initialization and the original ControlNet-only training setup."""
# Control branch initialization follows ControlNet's weight-copy convention; see NOTICE.
from pathlib import Path
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.attention import CrossAttention
from beyond_freq.generator import sdpa_attention
from beyond_freq.weights import load_weights

MODEL_CONFIG = Path(__file__).resolve().parent.parent / 'configs/generator/coco512.yaml'
POSITION_IDS = 'cond_stage_model.transformer.text_model.embeddings.position_ids'


def create_model(config=MODEL_CONFIG):
    model = instantiate_from_config(OmegaConf.load(config).model).cpu()
    CrossAttention.forward = sdpa_attention
    return model


def load_generator_state(model, state):
    if POSITION_IDS in state:
        model.cond_stage_model.transformer.text_model.embeddings._non_persistent_buffers_set.discard('position_ids')
    model.load_state_dict(state, strict=True)


def initialize_from_sd15(model, pretrained):
    """Copy SD1.5 backbone and encoder/middle weights; keep new zero-convs/hint layers.

    Input is a CompVis/LDM SD1.5 state dictionary, not a Diffusers UNet directory.
    Every frozen backbone parameter must be present with the expected shape.
    """
    if any(key.startswith('control_model.') for key in pretrained):
        raise ValueError('This is already a ControlNet model; load it with --weights instead')
    backbone_parameters = [key for key, _ in model.named_parameters()
                           if key.startswith(('model.diffusion_model.', 'first_stage_model.', 'cond_stage_model.'))]
    missing = [key for key in backbone_parameters if key not in pretrained]
    if missing:
        raise ValueError(f'Incomplete SD1.5 backbone: {len(missing)} missing parameters, e.g. {missing[:5]}')
    if POSITION_IDS in pretrained:
        model.cond_stage_model.transformer.text_model.embeddings._non_persistent_buffers_set.discard('position_ids')
    target = model.state_dict()
    report = {'backbone_loaded': 0, 'control_copied_from_unet': 0, 'new_control_tensors': 0,
              'default_diffusion_buffers': 0}
    for key in target:
        control = key.startswith('control_model.')
        source_key = 'model.diffusion_model.' + key[len('control_model.'):] if control else key
        if source_key in pretrained:
            tensor = pretrained[source_key]
            if tensor.shape != target[key].shape:
                raise ValueError(f'SD1.5 shape mismatch at {source_key}: {tuple(tensor.shape)} != {tuple(target[key].shape)}')
            target[key] = tensor
            report['control_copied_from_unet' if control else 'backbone_loaded'] += 1
        elif control:
            # New hint encoder / zero-convs remain at the architecture's initialization.
            report['new_control_tensors'] += 1
        else:
            report['default_diffusion_buffers'] += 1
    model.load_state_dict(target, strict=True)
    return report


def configure_control_training(model, learning_rate=5e-6):
    model.requires_grad_(False)
    model.control_model.requires_grad_(True)
    model.model.diffusion_model.backbone_gradients = False
    model.sd_locked = True
    model.learning_rate = learning_rate
    # Inference uses 1.5; the original training model uses its default strength 1.0.
    model.control_scales = [1.] * 13


def load_training_model(weights=None, sd_weights=None, config=MODEL_CONFIG):
    model = create_model(config)
    if sd_weights is not None:
        report = initialize_from_sd15(model, load_weights(sd_weights))
    elif weights is not None:
        load_generator_state(model, load_weights(weights))
        report = {'generator_state_loaded_strictly': True}
    else:
        # Lightning will restore the model and optimizer from --resume before fitting.
        report = {'restore_from_lightning_checkpoint': True}
    return model, report
