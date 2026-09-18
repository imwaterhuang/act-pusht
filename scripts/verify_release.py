"""Verify release files and optionally the downloaded inference checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256='e12c51f026eb19e56c3602aff14907ff55c0e8859606ef416bf150961bef0fb1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path);args=p.parse_args()
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(ROOT/name)!=expected: raise SystemExit(f'CHECKSUM FAILED: {name}')
    if args.checkpoint and sha(args.checkpoint)!=CHECKPOINT_SHA256:
        raise SystemExit('CHECKSUM FAILED: checkpoint')
    summary=json.loads((ROOT/'reports/act/release87/summary.json').read_text())
    assert summary['threshold']==.87 and summary['comparison']=='>'
    assert [(r['step'],r['success_count'],r['episode_count']) for r in summary['results']]==[(30000,32,50),(35000,31,50),(40000,31,50)]
    completed=json.loads((ROOT/'reports/act/training/completed.json').read_text())
    assert completed['completed'] and completed['step']==40000
    print(f'RELEASE_VERIFIED: {len(manifest["files"])} files; 87% metrics; 40000-step completion'+('; checkpoint verified' if args.checkpoint else ''))

if __name__=='__main__': main()
