"""CLIP architecture and tokenizer; trained text weights come from our checkpoint."""
from pathlib import Path
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


class CheckpointCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        assets = Path(__file__).resolve().parent / 'assets'
        self.tokenizer = CLIPTokenizer.from_pretrained(assets / 'tokenizer', local_files_only=True)
        config = CLIPTextConfig.from_pretrained(assets / 'clip_text_config.json', local_files_only=True)
        self.transformer = CLIPTextModel(config)
        self.transformer.eval().requires_grad_(False)

    def forward(self, text):
        ids = self.tokenizer(text, truncation=True, max_length=77, padding='max_length',
                             return_tensors='pt')['input_ids']
        return self.transformer(input_ids=ids.to(next(self.transformer.parameters()).device)).last_hidden_state

    def encode(self, text):
        return self(text)
