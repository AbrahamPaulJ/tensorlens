#!/usr/bin/env python3
"""Survey F-Droid: which apps ship ML models and ML runtimes. Standard library plus tensorlens-scan.

    python tools/fdroid_survey.py --scan build/tensorlens-scan --out survey/ [--limit N] [--max-gb 8]
    python tools/fdroid_survey.py --summary survey/

For the latest version of every app in the F-Droid index, the script reads only the APK's zip central
directory (HTTP range requests), records its native libraries (which name the ML runtime: TFLite, ONNX Runtime,
ncnn, ...), downloads only the entries that could hold a model (no images, code, layouts or fonts), runs
tensorlens-scan on them and deletes them again. Results go to survey/results.jsonl, one line per app, and the
run resumes where it stopped.
"""
import argparse
import concurrent.futures
import json
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zlib

REPO = 'https://f-droid.org/repo'
UA = {'User-Agent': 'tensorlens-fdroid-survey'}

# Entries that cannot hold a model. Everything else outside lib/ is downloaded and scanned.
SKIP_EXT = {
    '.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg', '.ico', '.bmp', '.heic', '.avif',
    '.xml', '.dex', '.arsc', '.kotlin_builtins', '.kotlin_module', '.properties', '.version', '.pro',
    '.ttf', '.otf', '.woff', '.woff2', '.mp3', '.ogg', '.wav', '.m4a', '.aac', '.flac', '.opus', '.mid',
    '.mp4', '.webm', '.mkv', '.html', '.htm', '.css', '.js', '.map', '.md', '.txt', '.csv', '.sf', '.rsa',
    '.dsa', '.ec', '.mf', '.so', '.class', '.java', '.kt', '.pdf', '.license', '.notice', '.idsig', '.json',
}
MIN_ENTRY, MAX_ENTRY = 512, 200 << 20  # a larger entry would hold that much memory per worker
# Native libraries that identify an ML runtime, by file name.
RUNTIMES = [
    ('tflite', ('tensorflowlite', 'tflite', 'litert', 'mediapipe')),
    ('onnxruntime', ('onnxruntime',)),
    ('ncnn', ('ncnn',)),
    ('executorch', ('executorch',)),
    ('pytorch', ('pytorch_jni', 'torch')),
    ('mnn', ('libmnn',)),
    ('paddle', ('paddle',)),
    ('ggml', ('ggml', 'llama', 'whisper')),
    ('opencv-dnn', ('opencv_java', 'opencv_dnn')),
]


def http(url, rng=None, retries=4):
    for attempt in range(retries):
        try:
            h = dict(UA)
            if rng:
                h['Range'] = 'bytes=%d-%d' % rng
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return r.read()
        except OSError as e:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def central_directory(url, size):
    tail_len = min(size, 1 << 16)
    tail = http(url, (size - tail_len, size - 1))
    i = tail.rfind(b'PK\x05\x06')
    if i < 0:
        raise ValueError('no end of central directory')
    n, cd_size, cd_off = struct.unpack_from('<HII', tail, i + 10)
    if cd_off == 0xFFFFFFFF or n == 0xFFFF:  # zip64
        j = tail.rfind(b'PK\x06\x06', 0, i)
        if j < 0:
            raise ValueError('zip64 without record')
        n, cd_size, cd_off = struct.unpack_from('<QQQ', tail, j + 32)
    start_in_tail = cd_off - (size - tail_len)
    cd = tail[start_in_tail:start_in_tail + cd_size] if start_in_tail >= 0 else http(url, (cd_off, cd_off + cd_size - 1))
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack_from('<H', cd, p + 10)
        csize, usize = struct.unpack_from('<II', cd, p + 20)
        nlen, elen, clen = struct.unpack_from('<HHH', cd, p + 28)
        off, = struct.unpack_from('<I', cd, p + 42)
        name = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, off):  # zip64 extended information
            q = 0
            while q + 4 <= len(extra):
                tag, ln = struct.unpack_from('<HH', extra, q)
                if tag == 1:
                    vals = list(struct.unpack_from('<%dQ' % (ln // 8), extra, q + 4))
                    if usize == 0xFFFFFFFF: usize = vals.pop(0)
                    if csize == 0xFFFFFFFF: csize = vals.pop(0)
                    if off == 0xFFFFFFFF: off = vals.pop(0)
                q += 4 + ln
        entries.append(dict(name=name, method=method, csize=csize, usize=usize, off=off))
        p += 46 + nlen + elen + clen
    return entries


def fetch_entry(url, e):
    # One request for the local header and the data: the local extra field is usually the central one or
    # shorter, so over-fetch a little and fetch the rest only when it is longer.
    guess = 30 + len(e['name'].encode('utf-8')) + 128
    blob = http(url, (e['off'], e['off'] + guess + e['csize'] - 1))
    nlen, elen = struct.unpack_from('<HH', blob, 26)
    start = 30 + nlen + elen
    data = blob[start:start + e['csize']]
    if len(data) < e['csize']:
        more = e['off'] + start + len(data)
        data += http(url, (more, e['off'] + start + e['csize'] - 1))
    if e['method'] == 0:
        return data
    if e['method'] == 8:
        return zlib.decompressobj(-15).decompress(data, MAX_ENTRY)
    raise ValueError('compression method %d' % e['method'])


def candidate(name, usize):
    low = name.lower()
    if low.startswith(('lib/', 'meta-inf/', 'kotlin/', 'okhttp3/', 'org/', 'com/', 'google/')):
        return False
    if low.startswith('res/') and not low.startswith('res/raw/'):
        return False
    ext = os.path.splitext(low)[1]
    return ext not in SKIP_EXT and MIN_ENTRY <= usize <= MAX_ENTRY


def runtimes(libs):
    found = set()
    for lib in libs:
        base = os.path.basename(lib).lower()
        for rt, keys in RUNTIMES:
            if any(k in base for k in keys):
                found.add(rt)
    return sorted(found)


def survey_app(pkg, ver, args, tmp):
    url = REPO + ver['file']['name']
    size = ver['file']['size']
    entries = central_directory(url, size)
    libs = sorted({e['name'] for e in entries if e['name'].startswith('lib/') and e['name'].endswith('.so')})
    cands = [e for e in entries if candidate(e['name'], e['usize'])]
    rec = dict(pkg=pkg, version=ver['manifest'].get('versionCode'), apk_size=size, entries=len(entries),
               libs=sorted({os.path.basename(l) for l in libs}), runtimes=runtimes(libs),
               candidates=len(cands), downloaded=0, models=[], errors=[])
    budget = args.per_app_mb << 20
    files = []
    for e in sorted(cands, key=lambda e: e['csize']):
        if rec['downloaded'] + e['csize'] > budget:
            rec['errors'].append('per-app budget reached, %d candidates left' % (len(cands) - len(files)))
            break
        try:
            data = fetch_entry(url, e)
        except Exception as ex:  # one bad entry must not lose the app
            rec['errors'].append('%s: %s' % (e['name'], ex))
            continue
        rec['downloaded'] += e['csize']
        path = os.path.join(tmp, '%04d_%s' % (len(files), os.path.basename(e['name'])[-80:]))
        with open(path, 'wb') as f:
            f.write(data)
        files.append((path, e['name']))
    if files:
        out_json = os.path.join(tmp, 'scan.json')
        subprocess.run([args.scan, '-q', '--json', out_json] + [p for p, _ in files],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
        if os.path.exists(out_json):
            with open(out_json, encoding='utf-8') as f:
                rep = json.load(f)
            where = {os.path.basename(p): n for p, n in files}
            for m in rep.get('models', []):
                w = m.get('where', '')
                base = os.path.basename(w.split('!')[0])
                m['where'] = where.get(base, base) + (('!' + w.split('!', 1)[1]) if '!' in w else '')
                rec['models'].append({k: m.get(k) for k in ('where', 'format', 'ops', 'size', 'custom_ops', 'producer')})
            os.remove(out_json)
        for p, _ in files:
            os.remove(p)
    return rec


def latest(pkg_entry):
    vs = list(pkg_entry['versions'].values())
    return max(vs, key=lambda v: v['manifest'].get('versionCode', 0)) if vs else None


def run(args):
    os.makedirs(args.out, exist_ok=True)
    index = os.path.join(args.out, 'index-v2.json')
    if not os.path.exists(index):
        with open(index, 'wb') as f:
            f.write(http(REPO + '/index-v2.json'))
    with open(index, encoding='utf-8') as f:
        full = json.load(f)['packages']
    # Keep only the latest version's file and version code: the full index costs about 1 GB of Python objects.
    pkgs = {}
    for name, entry in full.items():
        v = latest(entry)
        pkgs[name] = {'versions': {'v': {'file': {'name': v['file']['name'], 'size': v['file']['size']},
                                         'manifest': {'versionCode': v['manifest'].get('versionCode', 0)}}}} if v else {'versions': {}}
    del full
    results = os.path.join(args.out, 'results.jsonl')
    done, total_dl = set(), 0
    if os.path.exists(results):
        with open(results, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                done.add(r['pkg'])
                total_dl += r.get('downloaded', 0) + 70000
    tmp = os.path.join(args.out, 'tmp')
    os.makedirs(tmp, exist_ok=True)
    names = sorted(pkgs)[:args.limit] if args.limit else sorted(pkgs)
    if args.only:
        names = [n for n in args.only.split(',') if n in pkgs]
    args.scan = os.path.abspath(args.scan)
    todo = [p for p in names if p not in done and latest(pkgs[p]) is not None]
    lock = threading.Lock()
    state = dict(dl=total_dl, n=0, stop=False)
    t0 = time.time()

    def one(pkg):
        if state['stop']:
            return
        ver = latest(pkgs[pkg])
        work = os.path.join(tmp, pkg)
        os.makedirs(work, exist_ok=True)
        try:
            rec = survey_app(pkg, ver, args, work)
        except Exception as ex:
            rec = dict(pkg=pkg, version=ver['manifest'].get('versionCode'), apk_size=ver['file']['size'],
                       fatal='%s: %s' % (type(ex).__name__, ex), models=[], runtimes=[], libs=[], downloaded=0)
        for f in os.listdir(work):
            os.remove(os.path.join(work, f))
        os.rmdir(work)
        with lock:
            state['dl'] += rec.get('downloaded', 0) + 70000
            state['n'] += 1
            with open(results, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            if rec['models'] or rec['runtimes']:
                print('%-50s runtimes=%s models=%d' % (pkg, ','.join(rec['runtimes']), len(rec['models'])), flush=True)
            if state['n'] % 100 == 0:
                print('%5d/%d  %.0f min, %.2f GB downloaded' % (state['n'], len(todo), (time.time() - t0) / 60,
                                                              state['dl'] / (1 << 30)), flush=True)
            if state['dl'] > args.max_gb * (1 << 30) and not state['stop']:
                state['stop'] = True
                print('STOP: download budget of %.1f GB reached' % args.max_gb, flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    print('DONE', flush=True)


def summary(out):
    rows = [json.loads(l) for l in open(os.path.join(out, 'results.jsonl'), encoding='utf-8')]
    ok = [r for r in rows if 'fatal' not in r]
    with_models = [r for r in ok if r['models']]
    with_rt = [r for r in ok if r['runtimes']]
    fmt, rts, custom = {}, {}, {}
    for r in with_models:
        for m in r['models']:
            fmt[m['format']] = fmt.get(m['format'], 0) + 1
            for c in m.get('custom_ops') or []:
                custom.setdefault(c, set()).add(r['pkg'])
    for r in with_rt:
        for rt in r['runtimes']:
            rts[rt] = rts.get(rt, 0) + 1
    print('apps surveyed: %d (%d unreadable)' % (len(ok), len(rows) - len(ok)))
    print('apps shipping an ML runtime library: %d  %s' % (len(with_rt), dict(sorted(rts.items(), key=lambda x: -x[1]))))
    print('apps with at least one model found: %d, models: %d  %s' % (len(with_models), sum(fmt.values()), fmt))
    print('apps with custom ops: %d' % len({p for s in custom.values() for p in s}))
    for c, ps in sorted(custom.items(), key=lambda x: -len(x[1]))[:20]:
        print('   %-40s %d app(s)' % (c, len(ps)))
    print('downloaded: %.2f GB' % (sum(r.get('downloaded', 0) for r in rows) / (1 << 30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', help='path to tensorlens-scan')
    ap.add_argument('--out', default='survey')
    ap.add_argument('--limit', type=int, default=0, help='first N apps (alphabetical), for a trial run')
    ap.add_argument('--workers', type=int, default=6, help='apps surveyed in parallel')
    ap.add_argument('--only', help='comma-separated package names')
    ap.add_argument('--max-gb', type=float, default=8.0, help='stop after downloading this much')
    ap.add_argument('--per-app-mb', type=int, default=300, help='download at most this much per app')
    ap.add_argument('--summary', metavar='DIR')
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    elif a.scan:
        run(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
