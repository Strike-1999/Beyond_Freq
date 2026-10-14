"""COCO class-color blueprints compatible with the released generator weights."""
from collections import Counter
import colorsys

import cv2
import inflect
import numpy as np


class BlueprintRenderer:
    def __init__(self, categories):
        # Preserve COCO category order. Reordering the palette changes conditioning.
        self.names = {cat['id']: cat['name'] for cat in categories}
        self.hues = {cat['id']: (i + 1) / len(categories) for i, cat in enumerate(categories)}
        self.plural = inflect.engine()

    def render_native(self, annotations, width, height):
        """Render at COCO image dimensions, as in the training PNG preparation."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        counts = Counter()
        areas = [a.get('area', a['bbox'][2] * a['bbox'][3]) for a in annotations]
        for index in np.argsort(areas)[::-1]:
            ann = annotations[index]
            category = ann['category_id']
            value = max(0., 1 - counts[category] * .02)
            rgb = colorsys.hsv_to_rgb(self.hues[category], 1., value)
            bgr = tuple(int(channel * 255) for channel in rgb[::-1])
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr, -1)
            counts[category] += 1
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def render(self, annotations, width, height, output_size=512):
        canvas = self.render_native(annotations, width, height)
        return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    def prompt(self, annotations):
        counts = Counter()
        areas = [a.get('area', a['bbox'][2] * a['bbox'][3]) for a in annotations]
        for index in np.argsort(areas):
            counts[self.names[annotations[index]['category_id']]] += 1
        phrases = [f'{count} {self.plural.plural(name) if count > 1 else name}' for name, count in counts.items()]
        return 'a photograph with ' + ', '.join(phrases) if phrases else 'a photograph'
