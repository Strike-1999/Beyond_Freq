"""Export the released L2I checkpoint without changing any tensor or its dtype."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-shard-bytes', type=int, default=2_000_000_000)
    parser.add_argument('--prefix', help='Optional shard filename stem; defaults to the saved update count')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = saved.get('state_dict', saved)
    step = saved.get('global_step')
    prefix = args.prefix or ('beyondfreq_l2i_coco512_100k' if step == 100000 else
                            f'beyondfreq_l2i_coco512_{step}steps' if step is not None else
                            'beyondfreq_l2i_coco512_trained')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', prefix):
        parser.error('--prefix must contain only letters, digits, underscores and hyphens')
    groups, group, size = [], {}, 0
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'Non-tensor state entry: {key}')
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes > args.max_shard_bytes:
            raise ValueError(f'Tensor exceeds shard limit: {key}')
        if group and size + tensor_bytes > args.max_shard_bytes:
            groups.append(group)
            group, size = {}, 0
        group[key] = tensor.detach().cpu().contiguous()
        size += tensor_bytes
    if group:
        groups.append(group)
    weight_map, files = {}, []
    for i, tensors in enumerate(groups, 1):
        filename = f'{prefix}-{i:05d}-of-{len(groups):05d}.safetensors'
        path = args.output / filename
        save_file(tensors, str(path), metadata={'format': 'pt', 'model': 'BeyondFreq-L2I',
                  'conversion': 'lossless; all original dtypes and tensor names preserved'})
        with safe_open(path, framework='pt', device='cpu') as exported:
            assert set(exported.keys()) == set(tensors)
            for key, tensor in tensors.items():
                actual = exported.get_tensor(key)
                assert actual.dtype == tensor.dtype and torch.equal(actual, tensor), key
                weight_map[key] = filename
        assert path.stat().st_size < 2 * 1024**3
        record = {'filename': filename, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        files.append(record)
        print(json.dumps({'event': 'verified_weight_shard', **record}), flush=True)
    total_size = sum(t.numel() * t.element_size() for t in state.values())
    (args.output / f'{prefix}.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': total_size, 'model': 'BeyondFreq-L2I'}, 'weight_map': weight_map
    }, indent=2))
    manifest = {'model': 'BeyondFreq-L2I', 'dataset': 'COCO', 'resolution': [512, 512],
                'source_checkpoint_bytes': args.checkpoint.stat().st_size,
                'source_checkpoint_sha256': sha256(args.checkpoint),
                'global_step': saved.get('global_step'), 'epoch': saved.get('epoch'),
                'tensor_count': len(state), 'total_tensor_bytes': total_size,
                'all_tensors_bitwise_verified': True, 'dtype_conversion': 'none', 'files': files}
    (args.output / 'weights_manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'SHA256SUMS').write_text(''.join(f"{f['sha256']}  {f['filename']}\n" for f in files))
    print(json.dumps({'event': 'lossless_export_complete', 'shards': len(files), 'tensors': len(state),
                      'bytes': total_size}), flush=True)


if __name__ == '__main__':
    main()
