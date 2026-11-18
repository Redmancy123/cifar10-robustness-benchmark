"""Prepare public data, then bind it to the archived manifest after exact content checks.

The original manifest includes a creation timestamp. Only that timestamp may differ
after download; file hashes, archive identities, counts and splits must all match.
No frozen source file or frozen configuration is modified.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import digest, verify_frozen


def validate_manifest(actual, reference):
    actual_content = {k: v for k, v in actual.items() if k != 'created'}
    reference_content = {k: v for k, v in reference.items() if k != 'created'}
    if actual_content != reference_content:
        raise RuntimeError('Prepared data differs from the reference manifest; refusing to bind')


def main(check_only=False):
    config = json.loads((ROOT / 'configs/frozen_v1_2.json').read_text(encoding='utf-8'))
    for relative, expected in config['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError('Frozen source mismatch: ' + relative)
    reference_path = ROOT / 'evidence/data_manifest.json'
    if digest(reference_path) != config['data_manifest_sha256']:
        raise RuntimeError('Archived manifest checksum mismatch')
    for key, expected in config['split_hashes'].items():
        if digest(ROOT / 'data/splits' / (key + '_indices.npy')) != expected:
            raise RuntimeError('Split checksum mismatch: ' + key)
    if check_only:
        print('Frozen source, archived manifest and split checksums match.')
        return
    (ROOT / 'runs').mkdir(exist_ok=True)
    subprocess.run([sys.executable, 'scripts/prepare_data.py'], cwd=ROOT, check=True)
    prepared = ROOT / 'data/data_manifest.json'
    actual = json.loads(prepared.read_text(encoding='utf-8'))
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    validate_manifest(actual, reference)
    # Store the original bytes only after every content field matched independently.
    prepared.write_bytes(reference_path.read_bytes())
    verify_frozen(config)
    print('Data verified; original manifest timestamp restored for frozen-config verification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Check small archived files without downloads')
    main(parser.parse_args().check_only)
