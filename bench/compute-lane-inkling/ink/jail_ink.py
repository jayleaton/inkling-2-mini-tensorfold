#!/usr/bin/env python3
"""Qualify the isolated HumanEval/reference jail in the recipe's runner image (the kit's jail.qualify, unchanged).

  jail_ink.py qualify --freeze F --host NAME --out NEW_DIR
  jail_ink.py qualify --image-id sha256:... --tag TAG [--patches P] --selected FILE --out NEW_DIR   (local CPU test)

For each of the 24 frozen HumanEval tasks: the canonical solution passes (with the harness receipt), a wrong
`return None` fails, an infinite loop fails as a timeout, `os._exit(0)` fails without a receipt; plus the six
reference-executor cases (323, 55, REPL repr, one LF, no output, fresh namespace). Grading may run only after
this writes qualified=true.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import safety  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('qualify')
    p.add_argument('--freeze')
    p.add_argument('--host')
    p.add_argument('--image-id')
    p.add_argument('--tag', default=common.IMAGE_TAG)
    p.add_argument('--patches')
    p.add_argument('--selected')
    p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.freeze:
        host = common.read_json(Path(args.freeze) / 'hosts' / (args.host + '.json'))
        image_id, tag, patches = host['image_id'], common.IMAGE_TAG, host['tf_patches']
        selected = Path(args.freeze) / 'stdbench' / 'humaneval-selected.jsonl'
    else:
        common.require(args.image_id and args.selected, '--image-id and --selected, or --freeze and --host')
        image_id, tag, patches, selected = args.image_id, args.tag, args.patches, Path(args.selected)
    jail = safety.InkJail(image_id, tag, patches)
    receipt = safety.jail.qualify(jail, common.KIT / 'sources' / 'humaneval.jsonl.gz', selected, args.out)
    failed = [(r.get('id'), r['case']) for r in receipt['results'] if not r['qualified']]
    print(common.canonical({'qualified': receipt['qualified'], 'cases': len(receipt['results']), 'failed': failed,
                            'image': receipt['image'], 'out': str(Path(args.out) / 'qualification.json')}))
    if not receipt['qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
