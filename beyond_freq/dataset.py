"""Paired real images and visual blueprints for generator noise-prediction training."""
import json
from pathlib import Path
import random

import cv2
import numpy as np
from torch.utils.data import Dataset
from beyond_freq.blueprints import BlueprintRenderer


class PreparedBlueprintDataset(Dataset):
    """Original training interface: PNG blueprint + JPEG target + filename;prompt."""
    def __init__(self, prompts, blueprints, images, size=512, flip_probability=.5):
        self.root = Path(images)
        self.blueprints = Path(blueprints)
        self.samples = []
        for number, line in enumerate(Path(prompts).read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            name, separator, prompt = line.partition(';')
            path = Path(name)
            if not separator or not prompt or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Invalid filename;prompt at line {number} of {prompts}')
            self.samples.append((path, prompt))
        if not self.samples:
            raise ValueError(f'Empty prompt manifest: {prompts}')
        self.size = size
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, prompt = self.samples[index]
        source_path, target_path = self.blueprints / name, self.root / name.with_suffix('.jpg')
        source, target = cv2.imread(str(source_path)), cv2.imread(str(target_path))
        if source is None or target is None:
            raise FileNotFoundError(source_path if source is None else target_path)
        source = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), (self.size, self.size))
        target = cv2.resize(cv2.cvtColor(target, cv2.COLOR_BGR2RGB), (self.size, self.size))
        if random.random() < self.flip_probability:
            source, target = source[:, ::-1].copy(), target[:, ::-1].copy()
        return {'jpg': target.astype(np.float32) / 127.5 - 1.,
                'hint': source.astype(np.float32) / 255., 'txt': prompt}


class CocoBlueprintDataset(Dataset):
    def __init__(self, annotations, images, size=512, flip_probability=.5):
        coco = json.loads(Path(annotations).read_text())
        self.images = coco['images']
        self.root = Path(images)
        self.renderer = BlueprintRenderer(coco['categories'])
        self.annotations = {image['id']: [] for image in self.images}
        for ann in coco['annotations']:
            if not ann.get('iscrowd', False):
                self.annotations[ann['image_id']].append(ann)
        self.size = size
        self.flip_probability = flip_probability

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.root / info['file_name']
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(cv2.resize(image, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        anns = self.annotations[info['id']]
        blueprint = self.renderer.render(anns, info['width'], info['height'], self.size)
        if random.random() < self.flip_probability:
            image, blueprint = image[:, ::-1].copy(), blueprint[:, ::-1].copy()
        return {'jpg': image.astype(np.float32) / 127.5 - 1.,
                'hint': blueprint.astype(np.float32) / 255., 'txt': self.renderer.prompt(anns)}
