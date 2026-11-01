"""Install the pinned SAM 2 source and verify the single model checkpoint."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'
SHA256 = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    models = ROOT / 'models'
    models.mkdir(exist_ok=True)
    source = models / 'sam2'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/sam2.git', str(source)], check=True)
    current = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if current != REVISION:
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
        if dirty:
            raise RuntimeError('The model source has local edits; preserve them before running setup')
        subprocess.run(['git', '-C', str(source), 'checkout', REVISION], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '-e', str(source)],
                   env={**os.environ, 'SAM2_BUILD_CUDA': '0'}, check=True)
    path = models / 'sam2.1_hiera_base_plus.pt'
    if not path.exists():
        partial = path.with_suffix('.partial')
        try:
            with urlopen(URL, timeout=60) as response, partial.open('wb') as stream:
                while chunk := response.read(8 * 1024 * 1024):
                    stream.write(chunk)
            if checksum(partial) != SHA256:
                raise RuntimeError('Model download checksum does not match; run setup again')
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if checksum(path) != SHA256:
        raise RuntimeError(f'Model checksum does not match. Remove {path} and run setup again')
    print(f'Ready: SAM 2.1 Base+\nSHA256: {SHA256}')


if __name__ == '__main__':
    main()
