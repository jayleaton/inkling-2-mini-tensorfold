#!/usr/bin/env python3
"""MATH-500 blinded review: preseal (at freeze), seal one packet for all replies, merge reviewer labels.

The kit's phase2/math_review.py hard-codes two arms and 80 replies. This is the same procedure for every MATH reply
of this rerun (20 tasks x 2 seeds x 3 arms x 2 passes = 240): a precollection mapping from Random(20261105)
(shuffle the sorted keys, then one 128-bit opaque ID per candidate), a packet that shows only candidate ID, problem,
official answer and the delivered final content (no arm, seed, pass, status or timing), two complete independent
label files, a third reviewer for disagreements only, majority of correct/incorrect else unresolved, and generation
failures incorrect whatever the label. Label validation, label loading and the 0600 private writer are the kit's
own functions (math_review.validate_labels / load_labels / write_private), imported unchanged.

  seal  --freeze F --responses R... --packet OUT/packet.json --private-key OUT/key.json
  merge --packet P --private-key K --key-sha256 H --first A.json --first-id a --second B.json --second-id b
        [--third C.json --third-id c] --out OUT/math-reviewed.jsonl --audit OUT/math-audit.json
  disputes --packet P --first A.json --first-id a --second B.json --second-id b --out OUT/third-packet.json
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

math_review = common.kit('math_review')
KEY_FIELDS = ('pass', 'arm', 'seed', 'suite', 'id')
PACKET_FIELDS = {'candidate_id', 'problem', 'official_answer', 'candidate_final_content'}


def key_of(row):
    return tuple(row[k] for k in KEY_FIELDS)


def math_tasks(tasks):
    selected = sorted((t for t in tasks if t.get('kind') == 'math_manual'), key=lambda t: (t['suite'], t['id']))
    common.require(len(selected) == common.STD_COUNTS['math500'] and len({t['suite'] for t in selected}) == 1,
                   'all 20 MATH tasks of one suite required')
    return selected


def build_preseal(tasks, passes=tuple(common.PASSES), arms=common.ARMS, *, _fixture=False):
    selected = tasks if _fixture else math_tasks(tasks)
    keys = sorted((p, a, s, t['suite'], t['id']) for t in selected for s in common.SEEDS for a in arms for p in passes)
    rng = random.Random(20261105)
    rng.shuffle(keys)
    mapping, order = {}, []
    for key in keys:
        opaque = 'C-' + format(rng.getrandbits(128), '032x')
        common.require(opaque not in mapping, 'opaque ID collision')
        mapping[opaque] = dict(zip(KEY_FIELDS, key))
        order.append(opaque)
    return {'version': 1, 'phase': 'before_collection', 'rng': 20261105, 'key_fields': list(KEY_FIELDS),
            'passes': list(passes), 'arms': list(arms), 'tasks': selected, 'tasks_sha256': common.digest(selected),
            'mapping': mapping, 'order': order, 'trusted_selftest_fixture': _fixture}


def validate_preseal(preseal, expected_sha):
    common.require(common.digest(preseal) == expected_sha, 'precollection mapping differs from its pinned hash')
    common.require(preseal.get('rng') == 20261105 and preseal.get('phase') == 'before_collection', 'bad preseal')
    order, mapping = preseal['order'], preseal['mapping']
    common.require(len(order) == len(set(order)) == len(mapping) and set(order) == set(mapping), 'bad preseal IDs')
    expected = {(p, a, s, t['suite'], t['id']) for t in preseal['tasks'] for s in common.SEEDS
                for a in preseal['arms'] for p in preseal['passes']}
    common.require({key_of(v) for v in mapping.values()} == expected, 'preseal must cover the full MATH grid once')
    return mapping


def build_seal(preseal, preseal_sha, rows):
    mapping = validate_preseal(preseal, preseal_sha)
    tasks = {(t['suite'], t['id']): t for t in preseal['tasks']}
    suite = preseal['tasks'][0]['suite']
    selected = {}
    for row in rows:
        if row.get('suite') != suite or row.get('diagnostic'):
            continue
        key = key_of(row)
        common.require(key not in selected, 'duplicate MATH reply: ' + repr(key))
        selected[key] = row
    expected = {key_of(v) for v in mapping.values()}
    common.require(set(selected) == expected,
                   'all scheduled MATH replies required; missing ' + repr(sorted(expected - set(selected))[:4]))
    packet, entries = [], {}
    for opaque in preseal['order']:
        row = selected[key_of(mapping[opaque])]
        task = tasks[(row['suite'], row['id'])]
        content = row.get('content') if isinstance(row.get('content'), str) else ''
        packet.append({'candidate_id': opaque, 'problem': task['prompt'], 'official_answer': task['answer'],
                       'candidate_final_content': content})
        entries[opaque] = {'row': row, 'row_sha256': common.digest(row)}
    private = {'version': 1, 'rng': 20261105, 'packet_sha256': common.digest(packet), 'preseal': preseal,
               'preseal_sha256': preseal_sha, 'entries': entries,
               'responses_sha256': common.digest([selected[k] for k in sorted(selected)])}
    return packet, private


def validate_seal(packet, private, key_sha):
    common.require(common.digest(private) == key_sha, 'private key differs from its sealed hash')
    common.require(common.digest(packet) == private['packet_sha256'], 'packet hash mismatch')
    mapping = validate_preseal(private['preseal'], private['preseal_sha256'])
    ids = [item['candidate_id'] for item in packet]
    common.require(ids == private['preseal']['order'] and set(ids) == set(private['entries']), 'packet order/IDs')
    for item in packet:
        common.require(set(item) == PACKET_FIELDS and all(isinstance(v, str) for v in item.values()),
                       'packet holds prohibited metadata')
        entry = private['entries'][item['candidate_id']]
        common.require(key_of(entry['row']) == key_of(mapping[item['candidate_id']]), 'rekeyed response')
        common.require(common.digest(entry['row']) == entry['row_sha256'], 'sealed response hash mismatch')
    return set(ids)


def merge(packet, private, first, second, first_id, second_id, *, key_sha, third=None, third_id=None):
    candidates = validate_seal(packet, private, key_sha)
    common.require(first_id != second_id and third_id not in (first_id, second_id), 'distinct reviewers required')
    a = math_review.validate_labels(first, first_id, candidates, require_all=True)
    b = math_review.validate_labels(second, second_id, candidates, require_all=True)
    disputes = {c for c in candidates if a[c]['label'] != b[c]['label']}
    if third is not None:
        c = math_review.validate_labels(third, third_id, disputes, require_all=False)
    else:
        common.require(third_id is None, 'third reviewer identity without labels')
        c = {}
    rows, audit = [], []
    for candidate in sorted(candidates):
        votes = [a[candidate], b[candidate]] + ([c[candidate]] if candidate in c else [])
        counts = Counter(v['label'] for v in votes)
        label = next((x for x in ('correct', 'incorrect') if counts[x] >= 2), 'unresolved')
        row = dict(private['entries'][candidate]['row'])
        row['grade'] = ('incorrect' if row['status'] != 'ok' else 'manual_review' if label == 'unresolved' else label)
        row['math_review'] = {'candidate_id': candidate, 'label': label, 'packet_sha256': private['packet_sha256'],
                              'key_sha256': key_sha, 'reviewers': [v['reviewer_id'] for v in votes], 'sealed': True}
        rows.append(row)
        audit.append({'candidate_id': candidate, 'labels': votes, 'decision': label, 'effective_grade': row['grade']})
    rows.sort(key=key_of)
    summary = {'packet_sha256': private['packet_sha256'], 'private_key_sha256': key_sha,
               'first_review_sha256': common.digest(first), 'second_review_sha256': common.digest(second),
               'third_review_sha256': common.digest(third) if third is not None else None,
               'reviewer_ids': [first_id, second_id] + ([third_id] if third is not None else []),
               'disputed_candidates': sorted(disputes), 'third_reviewed': len(c),
               'unresolved_after_merge': sum(x['decision'] == 'unresolved' for x in audit), 'labels': audit}
    return rows, summary


def dispute_packet(packet, first, second, first_id, second_id):
    ids = {item['candidate_id'] for item in packet}
    a = math_review.validate_labels(first, first_id, ids, require_all=True)
    b = math_review.validate_labels(second, second_id, ids, require_all=True)
    return [item for item in packet if a[item['candidate_id']]['label'] != b[item['candidate_id']]['label']]


def selftest():
    tasks = [{'suite': 'fixture_math', 'id': str(i), 'kind': 'math_manual', 'prompt': 'What is 1+1?', 'answer': '2'}
             for i in range(2)]
    preseal = build_preseal(tasks, ('c8k',), ('P', 'L'), _fixture=True)
    pin = common.digest(preseal)
    rows = [{'pass': 'c8k', 'suite': 'fixture_math', 'id': t['id'], 'seed': s, 'arm': a, 'content': '2',
             'status': 'ok'} for t in tasks for s in common.SEEDS for a in ('P', 'L')]
    packet, private = build_seal(preseal, pin, rows)
    assert (packet, private) == build_seal(preseal, pin, list(reversed(rows)))
    assert all(set(item) == PACKET_FIELDS for item in packet) and len(packet) == 8
    for bad in (rows[:-1], rows + rows[:1]):
        try:
            build_seal(preseal, pin, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('partial/duplicate MATH packet accepted')
    tampered = json.loads(common.canonical(preseal))
    tampered['mapping'][tampered['order'][0]]['arm'] = 'T'
    try:
        build_seal(tampered, pin, rows)
    except ValueError:
        pass
    else:
        raise AssertionError('mutated preseal accepted')
    first = [{'candidate_id': p['candidate_id'], 'reviewer_id': 'r-a', 'label': 'correct', 'reason': 'equals 2'}
             for p in packet]
    second = [dict(v, reviewer_id='r-b') for v in first]
    plain = next(k for k, v in private['entries'].items() if v['row']['arm'] == 'P')
    lane = next(k for k, v in private['entries'].items() if v['row']['arm'] == 'L')
    for v in second:
        if v['candidate_id'] in (plain, lane):
            v.update(label='incorrect', reason='disagree')
    key_sha = common.digest(private)
    merged, audit = merge(packet, private, first, second, 'r-a', 'r-b', key_sha=key_sha)
    assert sum(r['grade'] == 'manual_review' for r in merged) == 2 and audit['unresolved_after_merge'] == 2
    assert [p['candidate_id'] for p in dispute_packet(packet, first, second, 'r-a', 'r-b')] == \
        [p['candidate_id'] for p in packet if p['candidate_id'] in (plain, lane)]
    third = [{'candidate_id': plain, 'reviewer_id': 'r-c', 'label': 'correct', 'reason': 'equals 2'}]
    merged, _ = merge(packet, private, first, second, 'r-a', 'r-b', key_sha=key_sha, third=third, third_id='r-c')
    assert sum(r['grade'] == 'manual_review' for r in merged) == 1
    failed = [dict(r, status='timeout') if r['arm'] == 'L' and r['seed'] == 101 else r for r in rows]
    packet2, private2 = build_seal(preseal, pin, failed)
    merged, _ = merge(packet2, private2, [dict(v) for v in first], [dict(v, reviewer_id='r-b') for v in first],
                      'r-a', 'r-b', key_sha=common.digest(private2))
    assert all(r['grade'] == 'incorrect' for r in merged if r['status'] != 'ok')
    for same in (('r-a', 'r-a'),):
        try:
            merge(packet, private, first, second, *same, key_sha=key_sha)
        except ValueError:
            pass
        else:
            raise AssertionError('same reviewer accepted')
    return {'selftest': 'passed', 'fixture_candidates': len(packet)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('seal')
    p.add_argument('--freeze', required=True)
    p.add_argument('--responses', nargs='+', required=True, help='every responses.jsonl of every MATH boot')
    p.add_argument('--packet', required=True)
    p.add_argument('--private-key', required=True)
    p = sub.add_parser('disputes')
    for name in ('--packet', '--first', '--first-id', '--second', '--second-id', '--out'):
        p.add_argument(name, required=True)
    p = sub.add_parser('merge')
    for name in ('--packet', '--private-key', '--key-sha256', '--first', '--first-id', '--second', '--second-id',
                 '--out', '--audit'):
        p.add_argument(name, required=True)
    p.add_argument('--third')
    p.add_argument('--third-id')
    sub.add_parser('selftest')
    args = parser.parse_args()
    if args.command == 'selftest':
        print(common.canonical(selftest()))
    elif args.command == 'seal':
        freeze = Path(args.freeze)
        manifest = common.read_json(freeze / 'manifest.json')
        key_path = freeze / manifest['math_preseal']['file']
        common.require(stat.S_IMODE(key_path.stat().st_mode) == 0o600, 'preseal must have mode 0600')
        rows = [r for path in args.responses for r in common.read_jsonl(path)]
        packet, private = build_seal(common.read_json(key_path), manifest['math_preseal']['sha256'], rows)
        common.require(not Path(args.packet).exists() and not Path(args.private_key).exists(), 'refuse overwrite')
        math_review.write_private(args.private_key, private)
        common.write_json_new(args.packet, packet)
        print(common.canonical({'candidates': len(packet), 'packet_sha256': common.digest(packet),
                                'private_key_sha256': common.digest(private),
                                'instruction': 'give each reviewer the packet only; keep the key private'}))
    elif args.command == 'disputes':
        packet = common.read_json(args.packet)
        out = dispute_packet(packet, math_review.load_labels(args.first), math_review.load_labels(args.second),
                             args.first_id, args.second_id)
        common.write_json_new(args.out, out)
        print(common.canonical({'disputed_candidates': len(out)}))
    else:
        key_path = Path(args.private_key)
        common.require(stat.S_IMODE(key_path.stat().st_mode) == 0o600, 'private key mode must be 0600')
        common.require(not Path(args.out).exists() and not Path(args.audit).exists(), 'refuse overwrite')
        rows, audit = merge(common.read_json(args.packet), common.read_json(key_path),
                            math_review.load_labels(args.first), math_review.load_labels(args.second),
                            args.first_id, args.second_id, key_sha=args.key_sha256,
                            third=math_review.load_labels(args.third) if args.third else None, third_id=args.third_id)
        math_review.write_private(args.audit, audit)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(common.canonical(row) + '\n')
        print(common.canonical({'merged_candidates': len(rows), 'audit_sha256': common.digest(audit),
                                'unresolved': audit['unresolved_after_merge']}))


if __name__ == '__main__':
    main()
