"""Upload the reviewed model folder to an explicitly selected Hugging Face repo."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Your HF account or organization/model-name')
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise RuntimeError('Authenticate with a write-capable HF token using huggingface-cli login first')
    if '/' not in args.repo:
        parser.error('--repo must contain the target account or organization')
    manifest = json.loads((args.folder / 'weights_manifest.json').read_text())
    assert manifest['all_tensors_bitwise_verified'] and manifest['dtype_conversion'] == 'none'
    for file in manifest['files']:
        path = args.folder / file['filename']
        assert path.resolve().parent == args.folder.resolve(), 'Unsafe shard path'
        assert path.stat().st_size == file['bytes']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(chunk)
        assert digest.hexdigest() == file['sha256'], file['filename']
    api = HfApi(token=token)
    account = api.whoami()['name']
    print(f'Authenticated as {account}; destination {args.repo}', flush=True)
    if api.repo_exists(args.repo):
        files = api.list_repo_files(args.repo)
        if files and 'weights_manifest.json' not in files:
            raise RuntimeError('Target repository already has unrelated contents; select a new model repo')
        if 'weights_manifest.json' in files:
            from huggingface_hub import hf_hub_download
            remote = json.loads(Path(hf_hub_download(args.repo, 'weights_manifest.json', token=token)).read_text())
            if remote.get('source_checkpoint_sha256') != manifest['source_checkpoint_sha256']:
                raise RuntimeError('Target repository contains a different checkpoint')
    api.create_repo(args.repo, repo_type='model', private=args.private, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=args.folder,
                              commit_message='Release BeyondFreq-L2I COCO512 original100k generator')
    info = api.model_info(args.repo, revision=commit.oid, files_metadata=True)
    metadata = {f.rfilename: f for f in info.siblings}
    for file in manifest['files']:
        uploaded = metadata[file['filename']]
        assert uploaded.size == file['bytes']
        if uploaded.lfs:
            sha = uploaded.lfs.sha256 if hasattr(uploaded.lfs, 'sha256') else uploaded.lfs['sha256']
            assert sha == file['sha256']
    report = {'repo_id': args.repo, 'url': f'https://huggingface.co/{args.repo}',
              'revision': commit.oid, 'uploaded_shards_verified': True}
    (args.folder.parent / 'hf_upload_complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
