"""Strict loading for the original checkpoint and lossless safetensors release."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_weights(path, verify=True):
    path = Path(path)
    if path.is_file() and path.suffix in {'.ckpt', '.pt', '.pth'}:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        return saved.get('state_dict', saved)
    if path.is_file() and path.suffix == '.safetensors':
        return load_file(str(path), device='cpu')
    if path.is_dir():
        indexes = list(path.glob('*.safetensors.index.json'))
        if len(indexes) != 1:
            raise ValueError(f'Expected exactly one weight index in {path}; found {len(indexes)}')
        index_path = indexes[0]
    else:
        index_path = path
        path = index_path.parent
    index = json.loads(index_path.read_text())
    weight_map = index['weight_map']
    if verify:
        manifest = json.loads((path / 'weights_manifest.json').read_text())
        expected = {f['filename']: f for f in manifest['files']}
        assert set(weight_map.values()) == set(expected), 'Shard manifest does not match weight index'
        for filename, record in expected.items():
            file = path / filename
            assert file.resolve().parent == path.resolve(), 'Unsafe shard path'
            assert file.stat().st_size == record['bytes'], f'Incorrect shard size: {filename}'
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(chunk)
            assert digest.hexdigest() == record['sha256'], f'Incorrect shard checksum: {filename}'
    state = {}
    for filename in sorted(set(weight_map.values())):
        file = path / filename
        if file.resolve().parent != path.resolve():
            raise ValueError('Weight shards must be inside the model directory')
        shard = load_file(str(file), device='cpu')
        if set(state) & set(shard):
            raise ValueError(f'Duplicate tensors in {filename}')
        if any(weight_map.get(key) != filename for key in shard):
            raise ValueError(f'Weight index does not match tensors in {filename}')
        state.update(shard)
    assert set(state) == set(weight_map), 'Missing weight tensors'
    return state
