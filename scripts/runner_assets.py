#!/usr/bin/env python3
"""The in-model runner's assets for a RUNNER=1 image: download the archive config/runner-assets.release names, check its
sha256, unpack it into an empty directory and check every file against config/runner-assets.sha256 (exactly the listed
files, each at its sha256). After the DeepSeek V4.1 Spark kit's fetch_runner_assets.py and check_runner_assets.py.

  runner_assets.py DEST                   download, check and unpack into DEST (new or empty)
  runner_assets.py DEST --archive FILE    the same from an archive already downloaded (still sha256-checked)
  runner_assets.py --check DIR            check a directory only
RUNNER_ASSETS_URL names a mirror, which must serve the same bytes."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'config/runner-assets.release'
MANIFEST = ROOT / 'config/runner-assets.sha256'


class AssetError(Exception):
    pass


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def release(path: Path = RELEASE) -> tuple[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            key, _, value = line.partition(' ')
            fields[key] = value.strip()
    url, digest = fields.get('url', ''), fields.get('sha256', '')
    if not url.startswith('https://') or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise AssetError(f'{path.name} must hold an https url and a 64-digit sha256')
    return url, digest


def listed(manifest: Path = MANIFEST) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        digest, name = line.split('  ', 1)
        if len(digest) != 64 or name in pins or name.startswith('/') or '..' in PurePosixPath(name).parts:
            raise AssetError(f'bad manifest line: {line}')
        pins[name] = digest
    return pins


def check(root: Path, manifest: Path = MANIFEST) -> list[str]:
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise AssetError(f'not a regular file: {path.relative_to(root)}')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
    pins = listed(manifest)
    problems = [f'missing {n}' for n in sorted(pins.keys() - files.keys())]
    problems += [f'unlisted {n}' for n in sorted(files.keys() - pins.keys())]
    problems += [f'sha256 differs {n}' for n in sorted(pins.keys() & files.keys()) if pins[n] != files[n]]
    return problems


def download(url: str, dest: Path, attempts: int = 3, timeout: int = 120) -> None:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, dest.open('wb') as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except (urllib.error.URLError, OSError) as error:
            last = error
            time.sleep(2 * (attempt + 1))
    raise AssetError(f'cannot download {PurePosixPath(url).name} from {url}: {last}')


def keep_dir_mode(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    """tarfile's 'data' filter, keeping the archive's directory modes so the image layer is the same on every builder."""
    kept = tarfile.data_filter(member, path)
    return kept.replace(mode=member.mode & 0o755, deep=False) if member.isdir() else kept


def unpack(archive: Path, dest: Path, url: str) -> None:
    """Only plain files and directories under relative names; anything else stops before a byte is written."""
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            for m in members:
                if not (m.isfile() or m.isdir()) or m.name.startswith('/') or '..' in PurePosixPath(m.name).parts:
                    raise AssetError(f'{archive.name} from {url} holds an unsafe entry: {m.name}')
            tar.extractall(dest, members=members, filter=keep_dir_mode)
    except tarfile.TarError as error:
        raise AssetError(f'cannot unpack {archive.name} from {url}: {error}') from error


def fetch(dest: Path, archive: str | None = None) -> None:
    url, digest = release()
    url = os.environ.get('RUNNER_ASSETS_URL') or url
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise AssetError(f'{dest} must be empty')
    name = PurePosixPath(url).name
    local = Path(archive) if archive else dest.parent / f'.{dest.name}-{name}'
    try:
        if not archive:
            print(f'runner assets: downloading {name} from {url}')
            download(url, local)
        got = sha256(local)
        if got != digest:
            raise AssetError(f'sha256 mismatch for {name} from {url}: expected {digest}, got {got}')
        unpack(local, dest, url)
    finally:
        if not archive and local.exists():
            local.unlink()
    problems = check(dest)
    if problems:
        raise AssetError('; '.join(problems) + f' (in {name} from {url}; expected by {MANIFEST.name})')
    print(f'runner assets: {name} sha256 ok, {len(listed())} files match {MANIFEST.name}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dest', type=Path, nargs='?', help='a new or empty directory')
    parser.add_argument('--archive', help='an archive already downloaded (offline)')
    parser.add_argument('--check', type=Path, metavar='DIR', help='only check DIR against the manifest')
    args = parser.parse_args(argv)
    try:
        if args.check:
            problems = check(args.check)
            for problem in problems:
                print(f'runner assets: {problem}', file=sys.stderr)
            if problems:
                return 1
            print(f'runner assets: {len(listed())} files match {MANIFEST.name}')
            return 0
        if args.dest is None:
            parser.error('DEST or --check DIR')
        fetch(args.dest, archive=args.archive)
    except (AssetError, OSError, ValueError) as error:
        print(f'runner assets: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
