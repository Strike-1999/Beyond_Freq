"""BeyondFreq-L2I: visual blueprint conditioning with the released SD1.5 model."""
import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from ldm.util import instantiate_from_config
from ldm.modules.attention import CrossAttention
from beyond_freq.sampling.ddim import DDIMSampler
from beyond_freq.weights import load_weights


def sdpa_attention(self, x, context=None, mask=None):
    context = x if context is None else context
    def heads(tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, -1).transpose(1, 2)
    q, k, v = heads(self.to_q(x)), heads(self.to_k(context)), heads(self.to_v(context))
    mask = None if mask is None else mask.reshape(x.shape[0], 1, 1, -1)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
    output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    return self.to_out(output)


class BeyondFreqL2I:
    def __init__(self, weights, device='cuda', config=None, verify_weights=True):
        config = Path(config) if config else Path(__file__).resolve().parent.parent / 'configs/generator/coco512.yaml'
        configuration = OmegaConf.load(config)
        self.model = instantiate_from_config(configuration.model)
        state = load_weights(weights, verify=verify_weights)
        # Older CLIP versions serialized this deterministic buffer; newer ones do not.
        if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in state:
            self.model.cond_stage_model.transformer.text_model.embeddings._non_persistent_buffers_set.discard('position_ids')
        self.model.load_state_dict(state, strict=True)
        del state
        # This changes the attention kernel, preserving parameter names and weights.
        CrossAttention.forward = sdpa_attention
        self.model.control_scales = [1.5] * 13
        self.model.to(device).eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, repo_id, revision=None, **kwargs):
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id, revision=revision,
                    allow_patterns=['*.safetensors', '*.json', 'model_config.yaml']))
        return cls(path, config=path / 'model_config.yaml', **kwargs)

    def set_trainable(self, control=True, unet=False):
        self.model.requires_grad_(False)
        self.model.control_model.requires_grad_(control)
        self.model.model.diffusion_model.requires_grad_(unet)
        self.model.model.diffusion_model.backbone_gradients = unet

    def text(self, prompts):
        with torch.no_grad():
            return self.model.get_learned_conditioning(prompts)

    def predict_noise(self, latent, timesteps, conditioning, precision='bf16'):
        device = latent.device
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and precision == 'bf16' else contextlib.nullcontext()
        with context:
            return self.model.apply_model(latent.float(), timesteps, conditioning).float()

    def sample(self, blueprints, prompts, image_ids=None, seed=451, steps=50, cfg_scale=4.5,
               strength=1.5, differentiable=False, precision='bf16'):
        """Return BCHW images in [0,1]; optionally retain the full DDIM gradient graph."""
        model = self.model
        device = next(model.parameters()).device
        blueprints = torch.as_tensor(blueprints, dtype=torch.float32, device=device)
        if blueprints.ndim != 4 or blueprints.shape[1] != 3:
            raise ValueError('Expected BCHW RGB blueprints')
        if not torch.isfinite(blueprints).all() or blueprints.min() < 0 or blueprints.max() > 1:
            raise ValueError('Blueprint pixels must be finite and in [0,1]')
        batch, _, height, width = blueprints.shape
        if height % 8 or width % 8 or len(prompts) != batch:
            raise ValueError('Image dimensions must be multiples of8 and prompt count must match')
        ids = list(range(batch)) if image_ids is None else list(image_ids)
        if len(ids) != batch:
            raise ValueError('image_ids must match batch size')
        model.control_scales = [float(strength)] * 13
        sampler = DDIMSampler(model)
        sampler.make_schedule(steps, ddim_eta=0., verbose=False)
        context = contextlib.nullcontext() if differentiable else torch.no_grad()
        with context:
            conditioning = {'c_concat': [torch.cat([torch.zeros_like(blueprints), blueprints])],
                            'c_crossattn': [torch.cat([self.text([''] * batch), self.text(prompts)])]}
            latent = torch.stack([torch.randn(4, height // 8, width // 8, device=device,
                     generator=torch.Generator(device=device).manual_seed(seed + int(i) * 1009)) for i in ids])
            for index in range(len(sampler.ddim_timesteps) - 1, -1, -1):
                timesteps = torch.full((2 * batch,), int(sampler.ddim_timesteps[index]), device=device, dtype=torch.long)
                inputs = torch.cat([latent, latent])
                def denoise(z, t):
                    return self.predict_noise(z, t, conditioning, precision)
                noise = checkpoint(denoise, inputs, timesteps, use_reentrant=False) if differentiable else denoise(inputs, timesteps)
                unconditional, conditional = noise.chunk(2)
                noise = unconditional + cfg_scale * (conditional - unconditional)
                alpha = torch.as_tensor(sampler.ddim_alphas[index], device=device, dtype=torch.float32)
                previous = torch.as_tensor(sampler.ddim_alphas_prev[index], device=device, dtype=torch.float32)
                clean = (latent - (1 - alpha).sqrt() * noise) / alpha.sqrt()
                latent = previous.sqrt() * clean + (1 - previous).sqrt() * noise
            decoded = [model.first_stage_model.decode(z[None].float() / model.scale_factor) for z in latent]
            return ((torch.cat(decoded) + 1) * .5).clamp(0, 1)
