#!/usr/bin/env python3
"""Grade every collected reply offline with the frozen DSv4.1 graders; untrusted code runs only in the jail.

  grade.py --freeze F --runs R --host NAME --jail-qualification Q.json --out NEW_DIR [--partial]

stdbench rows (grade_all.py's per-row rules, its gsm_grade imported unchanged): MATH-500 -> manual_review (every
answer goes to the blind review), generation status != ok -> incorrect, GSM8K -> grade_all.gsm_grade, HumanEval ->
the kit jail (InkJail) with correct only for status ok + harness receipt; the jail must hold a qualified
receipt from jail_ink.py for the same image ID. compute-DL rows: compute-dl/grade.py
grade() unchanged (strict JSON extraction, exact typed answer). Writes stdbench-graded.jsonl, compute-dl-graded.jsonl
and coverage.json. Without --partial every pass must hold the complete scheduled grid, each key exactly once.
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

bench = common.kit('bench')
grade_all = common.kit('grade_all')
p2analysis = common.kit('analysis')


def key(row):
    return (row['pass'], row['suite'], row['id'], row['seed'], row['arm'])


def collect(runs):
    """Every dataset row under runs/<pass>/<turn>/<boot[-rN]>/responses.jsonl, each key once."""
    rows = {}
    for path in sorted(Path(runs).glob('*/*/*/responses.jsonl')):
        for row in common.read_jsonl(path):
            common.require(not row.get('diagnostic') and row.get('included_in_cohort') is not False,
                           'diagnostic in a dataset file: ' + str(path))
            k = key(row)
            common.require(k not in rows, 'duplicate observation ' + repr(k))
            row['_source'] = str(path.relative_to(runs))
            if isinstance(row.get('tensorfold'), dict) and 'token_ids' in row['tensorfold']:
                # graded copies keep the accounting verdict, not the ID list (the raw rows keep it)
                row['tensorfold'] = {k2: v for k2, v in row['tensorfold'].items() if k2 != 'token_ids'}
            rows[k] = row
    return rows


def coverage(rows, freeze, partial):
    expected = {(e['pass'], e['suite'], e['id'], e['seed'], e['arm'])
                for e in common.read_jsonl(Path(freeze) / 'expected.jsonl')}
    extra = set(rows) - expected
    common.require(not extra, 'unexpected observations: ' + repr(sorted(extra)[:4]))
    missing = sorted(expected - set(rows))
    common.require(partial or not missing, f'{len(missing)} scheduled replies missing, e.g. {missing[:4]}')
    by_pass = {}
    for p in common.PASSES:
        exp = [k for k in expected if k[0] == p]
        by_pass[p] = {'scheduled': len(exp), 'observed': sum(k in rows for k in exp)}
    return {'missing': [list(m) for m in missing], 'by_pass': by_pass, 'complete': not missing}


def task_maps(freeze):
    std = common.read_json(Path(freeze) / 'analysis-stdbench.json')['tasks']
    cdl = common.read_json(Path(freeze) / 'analysis-compute-dl.json')['tasks']
    return {(t['suite'], t['id']): t for t in std}, {t['id']: t for t in cdl}


def grade_stdbench(row, task, jail):
    out = dict(row)
    out['censored'] = p2analysis.censored(row)
    if task.get('kind') == 'math_manual':
        result = {'grade': 'manual_review', 'reason': 'all_MATH_requires_independent_blinded_review'}
    elif row['status'] != 'ok':
        result = {'grade': 'incorrect', 'reason': 'generation_' + row['status']}
    elif task.get('kind') in ('integer', 'numeric'):
        result = grade_all.gsm_grade(task, row['content'])
    elif task.get('kind') == 'humaneval':
        evidence = jail.grade_humaneval(task, row['content'])
        passed = (evidence.get('grade') == 'correct' and evidence.get('status') == 'ok'
                  and evidence.get('harness_receipt') is True)
        result = {'grade': 'correct' if passed else 'incorrect',
                  'reason': evidence.get('reason', 'isolated_original_HumanEval_tests'), 'jail_evidence': evidence}
    else:
        raise ValueError('unsupported task kind')
    out['grading'] = result
    out['grade'] = result['grade']
    out['grader'] = {'name': 'phase2.grade_all rules', 'grade_all_sha256': common.sha_file(common.PHASE2 / 'grade_all.py')}
    return out


def grade_rows(rows, freeze, jail):
    std_tasks, cdl_tasks = task_maps(freeze)
    answers = common.read_json(common.CDL / 'dataset-v1' / 'answers-a.json')
    cdl_grade = common.cdl('grade')
    std, cdl = [], []
    for k in sorted(rows):
        row = rows[k]
        if row['suite'] == 'compute_dl':
            graded = cdl_grade.grade(row, cdl_tasks[row['id']], answers[row['id']])
            graded['censored'] = p2analysis.censored(row)
            cdl.append(graded)
        else:
            task = std_tasks[(row['suite'], row['id'])]
            common.require(row['task_sha256'] == bench.digest(task), 'row task hash differs from the freeze')
            std.append(grade_stdbench(row, task, jail))
    return std, cdl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--host', required=True, help='host binding whose image runs the HumanEval jail')
    parser.add_argument('--out', required=True)
    parser.add_argument('--partial', action='store_true', help='grade what exists (progress check only)')
    parser.add_argument('--jail-qualification', help='qualification.json written by jail_ink.py for this host')
    parser.add_argument('--selftest-fake-jail', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = Path(args.out)
    common.require(not out.exists(), 'output exists: ' + str(out))
    rows = collect(args.runs)
    cover = coverage(rows, args.freeze, args.partial)
    if args.selftest_fake_jail:
        import fake_server
        jail = fake_server.FakeJail()
    else:
        import safety
        host = common.read_json(Path(args.freeze) / 'hosts' / (args.host + '.json'))
        common.require(args.jail_qualification, '--jail-qualification is required')
        receipt = common.read_json(args.jail_qualification)
        common.require(receipt.get('qualified') is True and receipt['image']['id'] == host['image_id'],
                       'jail qualification missing, failed or for another image')
        jail = safety.InkJail(host['image_id'], common.IMAGE_TAG, host['tf_patches'])
    std, cdl = grade_rows(rows, args.freeze, jail)
    out.mkdir(parents=True)
    common.write_jsonl_new(out / 'stdbench-graded.jsonl', std)
    common.write_jsonl_new(out / 'compute-dl-graded.jsonl', cdl)
    common.write_json_new(out / 'coverage.json', cover)
    print(common.canonical({'stdbench': len(std), 'compute_dl': len(cdl), 'complete': cover['complete'],
                            'by_pass': cover['by_pass'],
                            'stdbench_sha256': hashlib.sha256((out / 'stdbench-graded.jsonl').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
