#!/usr/bin/env python3
"""Download the public binaries behind the README's results table, check their SHA-256, and extract the library
to open in IDA. Standard library only.

    python tools/fetch_samples.py samples/            # all of them (about 210 MB)
    python tools/fetch_samples.py samples/ lespas     # just one
    python tools/fetch_samples.py --list

Each sample ends up in samples/<name>/: the downloaded file, the extracted library, and the expected result.
F-Droid moves older app versions from /repo/ to /archive/; the script tries both.
"""
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    with open(os.path.join(HERE, 'samples.json'), encoding='utf-8') as f:
        return json.load(f)['samples']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    urls = [url]
    if '://f-droid.org/repo/' in url:
        urls.append(url.replace('/repo/', '/archive/'))
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'tensorlens-fetch-samples'})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + '.part', 'wb') as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            os.replace(dest + '.part', dest)
            return u
        except OSError as e:
            print(f'    {u}: {e}')
    raise SystemExit(f'could not download {url}')


def fetch(s, root):
    d = os.path.join(root, s['name'])
    os.makedirs(d, exist_ok=True)
    got = os.path.join(d, s['file'])
    if not (os.path.exists(got) and sha256(got) == s['sha256']):
        print(f"  downloading {s['file']} ({s['size'] / 1e6:.1f} MB)")
        download(s['url'], got)
    if sha256(got) != s['sha256']:
        raise SystemExit(f"{got}: SHA-256 mismatch, expected {s['sha256']}")
    target = got
    if s['open']:
        target = os.path.join(d, os.path.basename(s['open']))
        with zipfile.ZipFile(got) as z, z.open(s['open']) as src, open(target, 'wb') as dst:
            dst.write(src.read())
    with open(os.path.join(d, 'EXPECTED.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{s['what']}\nOpen in IDA: {os.path.basename(target)}\n")
        if s['add_models']:
            f.write(f"Then: File > Load file > TensorLens: add models..., pick {s['file']}\n")
        f.write(f"Expected: {s['expect']}\nSource: {s['url']}\nLicence: {s['licence']}\n")
    return target


def main(argv):
    samples = load()
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return
    if argv[0] == '--list':
        for s in samples:
            print(f"{s['name']:20} {s['size'] / 1e6:6.1f} MB  {s['what']}")
        return
    root, wanted = argv[0], set(argv[1:])
    unknown = wanted - {s['name'] for s in samples}
    if unknown:
        raise SystemExit(f"unknown sample(s): {', '.join(sorted(unknown))}; see --list")
    for s in samples:
        if wanted and s['name'] not in wanted:
            continue
        print(f"{s['name']}: {s['what']}")
        target = fetch(s, root)
        print(f"  open in IDA: {target}")
        if s['add_models']:
            print(f"  then add models from: {os.path.join(root, s['name'], s['file'])}")
        print(f"  expected: {s['expect']}")


if __name__ == '__main__':
    main(sys.argv[1:])
