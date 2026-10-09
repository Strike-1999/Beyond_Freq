"""Download a pinned BeyondFreq-L2I Hugging Face model snapshot."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='HF account/BeyondFreq-L2I-COCO512')
    parser.add_argument('--revision', help='Model commit or tag')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/beyondfreq_l2i_coco512_100k'))
    args = parser.parse_args()
    snapshot_download(args.repo, revision=args.revision, local_dir=args.output,
                      allow_patterns=['*.safetensors', '*.json', '*.yaml', 'README.md', 'LICENSE'])
    print(args.output.resolve())
