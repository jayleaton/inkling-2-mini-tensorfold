#!/usr/bin/env python3
"""CPU self-tests: no model, no Spark, no network beyond 127.0.0.1. Docker only for the optional jail controls.

  selftest.py --work DIR [--quick] [--jail-image TAG]

DIR must lie outside bench/compute-lane-inkling and its path must contain 'selftest'. Steps:
  units      body/field refusal, call success rule, runner cells, reference classification, arm gates
  kit        unchanged DSv4.1 selftests: phase2 analysis, grade_all, client, jail (command only), bench, MATH packet
  graders    GSM8K / compute-DL graders on fixed rows
  stats      bootstrap + exact sign-flip + Holm on synthetic rows through the relabelled kit analyses
  client     ink_client against the fake SSE server (ok, lane call, P+tools 400, error event, cut stream)
  freeze     create twice -> identical root hash; verify; drift refusal
  e2e        controller on the fake for every turn of both passes (--quick: c8k only), grade (fake jail),
             MATH seal + synthetic reviewers + merge, report
  faults     wrong delivered value -> STOPPED, rerun refused without --resume, --resume finishes the boot;
             P call -> STOPPED; diagnostic mismatch -> STOPPED; error event and cut stream recorded, not stops
  jail       (with --jail-image) the kit qualification in a local image holding /usr/bin/python3
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

INK = common.INK
PY = [sys.executable, '-I']
RESULTS = []


def step(name, detail):
    RESULTS.append({'step': name, **detail})
    print(common.canonical({'step': name, 'result': 'PASS', **detail}), flush=True)


def expect_error(fn, *args, contains=''):
    try:
        fn(*args)
    except (ValueError, RuntimeError, AssertionError) as error:
        assert contains in str(error), (contains, str(error))
        return str(error)
    raise AssertionError('expected a refusal: ' + contains)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run(args, env=None, ok=True):
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=3600)
    if ok and proc.returncode:
        raise AssertionError(f'{args[:4]} failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}')
    if not ok and proc.returncode == 0:
        raise AssertionError(f'{args[:4]} unexpectedly succeeded')
    return proc


# ---------------------------------------------------------------- units
def units():
    import safety
    b = common.body('Q', 101, 'L', 'c8k', 'M')
    assert b['tools'] == common.TOOLS and b['max_tokens'] == 8192 and b['deadline_s'] == 270
    assert 'tools' not in common.body('Q', 101, 'P', 'c32k', 'M')
    expect_error(common.check_body, dict(b, min_p=0.0), contains='unsupported request fields')
    expect_error(common.check_body, dict(b, chat_template_kwargs={'enable_thinking': True}), contains='kwargs')
    expect_error(common.check_body, dict(b, tools=[{'type': 'function'}]), contains='code_interpreter')
    expect_error(common.check_body, dict(b, return_token_ids=False), contains='native IDs')
    ok = {'status': 'ok', 'inserted': True, 'exit_code': 0}
    assert common.successful_call(ok) and not common.successful_call(dict(ok, exit_code=1))
    assert not common.successful_call(dict(ok, inserted=False))
    assert common.successful_call(dict(ok, exit_code=None, executor='in-model runner'))
    assert not common.successful_call(dict(ok, exit_code=None, executor='code tool'))
    assert common.runner_owned({'runner': {'inline': {'owned': 2}, 'fence': {'owned': 1}, 'x': {'owned': 9}}}) == 3
    import base64

    def ref(body, status='ok'):
        return {'status': status, 'normalized_body_base64': base64.b64encode(body.encode()).decode()}
    count = len
    assert safety.compare({'status': 'ok', 'output': '323'}, ref('323'))['verdict'] == 'match'
    assert safety.compare({'status': 'ok', 'output': '324'}, ref('323'))['verdict'] == 'mismatch'
    clip = safety.compare({'status': 'ok', 'output': 'x' * 512 + '\n' + common.TRUNCATION_MARKER}, ref('x' * 900), count)
    assert clip['verdict'] == 'clipped_match' and clip['token_rule_holds'] is True
    short = safety.compare({'status': 'ok', 'output': 'x' * 10 + '\n' + common.TRUNCATION_MARKER}, ref('x' * 20), count)
    assert short['verdict'] == 'clipped_match' and short['token_rule_holds'] is False
    assert safety.compare({'status': 'exit_error', 'output': 'E'}, ref('E', 'exit_error'))['verdict'] == 'error_match'
    assert safety.compare({'status': 'exit_error', 'output': 'E'}, ref('F', 'exit_error'))['verdict'] == 'error_differs'
    assert safety.compare({'status': 'ok', 'output': '1'}, ref('', 'timeout'))['verdict'] == 'inconclusive'
    assert safety.compare({'status': 'ok', 'output': '1'}, {'status': 'harness_error'})['verdict'] == 'inconclusive'
    call = {'status': 'ok', 'inserted': True, 'exit_code': 0, 'code': 'print(1)', 'output': '1'}
    assert safety.arm_violations('P', {'tensorfold': {'compute': {'calls': [call], 'inserted_tokens': 3}}})
    assert safety.arm_violations('L', {'tensorfold': {'runner': {'inline': {'owned': 1}}}})
    assert safety.arm_violations('T', {'tensorfold': {'compute_executed': True}})
    assert not safety.arm_violations('T', {'tensorfold': {'runner': {'inline': {'owned': 1}}}})
    assert not safety.arm_violations('P', {'tensorfold': {}})
    text = 'think print(1) <o>1</o>'
    row = {'tensorfold': {'token_ids': [ord(c) for c in text], 'compute': {'calls': [call]}}}
    detok = lambda ids: ''.join(map(chr, ids))   # noqa: E731
    assert safety.server_binding(row, detok)['qualified'] is True
    bad = {'tensorfold': {'token_ids': [ord(c) for c in 'nothing'], 'compute': {'calls': [call]}}}
    assert safety.server_binding(bad, detok)['qualified'] is False
    checked = safety.check_row('L', dict(row, status='ok', native_accounting={'complete': True}),
                               lambda code: ref('2'), detok, count)
    assert checked['stop'] and checked['mismatches'] == [0] and not checked['qualified']
    step('units', {'checks': 33})


# ---------------------------------------------------------------- kit selftests (unchanged modules)
def kit_selftests(work):
    analysis, grade_all, client, jail, bench = (common.kit(n) for n in ('analysis', 'grade_all', 'client', 'jail',
                                                                       'bench'))
    analysis.selftest()
    grade_all.selftest()
    client.selftest(work / 'kit-client-selftest.json')
    jail.selftest(work / 'kit-jail-command-selftest.json')
    # kit/bench.py's own selftest fails in the unchanged copy (its phase-1 make_report fixture asserts
    # resolved_subset lane_accuracy == 4/14; same on CPython 3.12 and 3.14). This harness uses only bench.normalize,
    # digest, usage_counts and read_jsonl from it (normalize is exercised by the freeze step). Record, do not hide.
    try:
        bench.selftest(argparse.Namespace(out=str(work / 'kit-bench-selftest'), bootstrap=20000))
        bench_status = 'ok'
    except AssertionError:
        bench_status = 'PRE-EXISTING FAILURE in unchanged kit (make_report fixture); module parts used are tested'
    import math_packet
    packet = math_packet.selftest()
    step('kit', {'phase2.analysis': 'ok', 'phase2.grade_all': 'ok', 'phase2.client': 'ok',
                 'phase2.jail(command)': 'ok', 'kit.bench': bench_status, 'ink.math_packet': packet['selftest']})


# ---------------------------------------------------------------- graders
def graders():
    grade_all = common.kit('grade_all')
    task = {'answer': 'It is 72.\n#### 1,234'}
    assert grade_all.gsm_grade(task, 'so \\boxed{1234}')['grade'] == 'correct'
    assert grade_all.gsm_grade(task, 'Final answer: 1,234')['grade'] == 'correct'
    assert grade_all.gsm_grade(task, '\\boxed{1235}')['grade'] == 'incorrect'
    assert grade_all.gsm_grade(task, '<think>\\boxed{1234}')['grade'] == 'incorrect'
    g = common.cdl('grade')
    tasks = {t['id']: t for t in common.read_jsonl(common.CDL / 'dataset-v1' / 'tasks-public.jsonl')}
    answers = common.read_json(common.CDL / 'dataset-v1' / 'answers-a.json')
    tid = 'compute-dl-v1-integer-001'
    base = {'status': 'ok', 'finish_reason': 'stop'}
    good = g.grade(dict(base, content='Final answer: ' + json.dumps(answers[tid])), tasks[tid], answers[tid])
    assert good['grade'] == 'correct'
    assert g.grade(dict(base, content=str(answers[tid]) + '.0'), tasks[tid], answers[tid])['grader']['format_rejection']
    assert g.grade(dict(base, content=json.dumps(answers[tid] + 1)), tasks[tid], answers[tid])['grade'] == 'incorrect'
    assert g.grade(dict(base, status='timeout', content=json.dumps(answers[tid])), tasks[tid],
                   answers[tid])['grade'] == 'incorrect'
    agg = 'compute-dl-v1-aggregation-001'
    fenced = '```json\n' + json.dumps(answers[agg]) + '\n```'
    assert g.grade(dict(base, content=fenced), tasks[agg], answers[agg])['grade'] == 'correct'
    dup = '{"A": 1, "A": 2}'
    assert g.grade(dict(base, content=dup), tasks[agg], answers[agg])['grader']['reason'] == 'duplicate_key'
    step('graders', {'gsm8k': 4, 'compute_dl': 6})


# ---------------------------------------------------------------- statistics on synthetic data
def stats():
    import report
    p2 = common.kit('analysis')
    tasks = [{'suite': s, 'id': f'{s}-{i:02d}', 'kind': 'x', 'prompt': 'p'} for s, n in common.STD_COUNTS.items()
             for i in range(n)]
    manifest = {'tasks': tasks, 'seeds': list(common.SEEDS)}
    rows = []
    for t in tasks:
        for seed in common.SEEDS:
            for arm in common.ARMS:
                win = (arm == 'L' and t['suite'] == 'gsm8k') or (arm != 'P' and t['suite'] == 'humaneval'
                                                                 and t['id'].endswith(('0', '2')))
                grade = 'correct' if win or (t['suite'] == 'math500' and arm == 'P') else 'incorrect'
                rows.append({'suite': t['suite'], 'id': t['id'], 'seed': seed, 'arm': arm, 'grade': grade,
                             'status': 'ok', 'elapsed_seconds': 1.0 + (arm == 'P'), 'finish_reason': 'stop',
                             'tokens': {}, 'native_accounting': {'complete': True,
                                                                 'returned_tokens': 100 if arm == 'P' else 50}})
    result = p2.analyze(report.relabel(rows, 'L', 'P', ('lane', 'plain')), manifest)['suites']
    gsm, math, he = result['gsm8k'], result['math500'], result['humaneval']
    assert gsm['accuracy_difference_lane_minus_plain'] == 1.0 and gsm['accuracy_ci95'] == [1.0, 1.0]
    assert gsm['sign_flip']['p_two_sided'] == 2 / 2 ** 48 and gsm['sign_flip']['holm_adjusted_p'] < 1e-10
    assert math['accuracy_difference_lane_minus_plain'] == -1.0 and math['sign_flip']['p_two_sided'] == 2 / 2 ** 20
    assert he['accuracy_ci95'][0] > 0 and he['tokens']['paired_sums_ratio_lane_over_plain'] == 0.5
    t_res = p2.analyze(report.relabel(rows, 'T', 'P', ('lane', 'plain')), manifest)['suites']
    assert t_res['gsm8k']['accuracy_difference_lane_minus_plain'] == 0 and t_res['gsm8k']['sign_flip']['p_two_sided'] == 1
    again = p2.analyze(report.relabel(rows, 'L', 'P', ('lane', 'plain')), manifest)['suites']['humaneval']
    assert again['accuracy_ci95'] == he['accuracy_ci95']    # RNG 20261104 draws are reproducible
    cdl_tasks = common.read_jsonl(common.CDL / 'dataset-v1' / 'tasks-public.jsonl')
    crow = []
    for t in cdl_tasks:
        for seed in common.SEEDS:
            for arm in common.ARMS:
                right = (arm != 'P' and t['family'] in ('integer', 'modular')
                         or arm == 'P' and t['family'] == 'dp' and seed == 101)
                crow.append({'suite': 'compute_dl', 'id': t['id'], 'seed': seed, 'arm': arm, 'status': 'ok',
                             'grade': 'correct' if right else 'incorrect', 'finish_reason': 'stop',
                             'elapsed_seconds': 2.0, 'native_accounting': {'complete': True, 'returned_tokens': 10}})
    cdl = common.cdl('analysis').analyze(report.relabel(crow, 'L', 'P', ('L', 'D')), cdl_tasks)
    o = cdl['overall']
    assert cdl['status'] == 'COMPLETE' and abs(o['solve_difference_L_minus_D'] - 60 / 240) < 1e-12
    assert o['solve_difference_ci95'][0] > 0 and o['sign_flip']['p_two_sided'] < 1e-3
    assert cdl['families']['dp']['solve_difference_L_minus_D'] == -0.5
    step('stats', {'stdbench_gsm_p': gsm['sign_flip']['p_two_sided'], 'humaneval_ci95': he['accuracy_ci95'],
                   'compute_dl_diff': o['solve_difference_L_minus_D'], 'compute_dl_ci95': o['solve_difference_ci95'],
                   'compute_dl_p': o['sign_flip']['p_two_sided']})


# ---------------------------------------------------------------- client against the fake server
def client(freeze, work):
    import fake_server
    import ink_client
    import safety
    lc = fake_server.FakeLifecycle('http://127.0.0.1:%d' % free_port(), work / 'client-selftest', freeze,
                                   'sha256:' + '1' * 64)
    base = 'http://%s:%d' % lc.address
    sessions = work / 'client-selftest-sessions'
    sessions.mkdir()
    seen = {}
    for arm, lane, fault in (('P', '0', None), ('L', '1', None), ('L', '1', 'error:L:1'), ('T', 'tool-full', 'cut:T:1'),
                             ('T', 'tool-full', None)):
        if fault:
            os.environ['INK_FAKE_FAULT'] = fault
        else:
            os.environ.pop('INK_FAKE_FAULT', None)
        lc.start(dict(common.COMMON_ENV, COMPUTE_LANE=lane, SESSION_DIR=str(sessions)), work / 'client-start.log')
        try:
            body = common.body(common.DIAGNOSTICS[0][1], 101, arm, 'c8k', common.DEFAULT_MODEL)
            row = ink_client.send(base, body, cap=8192, wait_s=30, raw_path=work / f'client-{arm}-{fault}.sse.gz')
            seen[(arm, fault)] = row
            if arm == 'P':
                refused = ink_client.send(base, dict(body, tools=common.TOOLS), cap=8192, wait_s=30)
                assert refused['status'] == 'error' and refused['http_status'] == 400
                assert row['status'] == 'ok' and not common.compute_calls(row['tensorfold'])
            elif fault is None:
                calls = common.compute_calls(row['tensorfold'])
                assert row['status'] == 'ok' and calls and calls[0]['output'] == '323'
                assert row['native_accounting']['complete'] and row['client_ttft_seconds'] is not None
                detok, count = safety.server_tools(base)
                check = safety.check_row(arm, row, fake_server.fake_reference, detok, count)
                assert check['qualified'] and check['references'][0]['verdict'] == 'match', check
            elif fault.startswith('error'):
                assert row['status'] == 'error' and 'server error event' in row['error']
                assert not row['native_accounting']['complete']
            else:
                assert row['status'] == 'error' and 'without [DONE]' in row['error']
        finally:
            lc.stop(work / 'client-stop.log')
    os.environ.pop('INK_FAKE_FAULT', None)
    step('client', {'cases': len(seen) + 1, 'p_tools_http': 400})


# ---------------------------------------------------------------- freeze
def make_freeze(work, name):
    proc = run(PY + [str(INK / 'freeze.py'), 'create', '--recipe', str(common.ROOT.parent.parent),
                     '--out', str(work / name)])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def bind_fake_host(freeze, host='fake'):
    manifest = common.read_json(freeze / 'manifest.json')
    common.write_json_new(freeze / 'hosts' / (host + '.json'), {
        'host': host, 'hostname': 'selftest', 'image_tag': common.IMAGE_TAG, 'image_id': 'sha256:' + '1' * 64,
        'tf_patches': manifest['recipe']['tf_patches_expected'], 'served_name': manifest['model'],
        'freeze_root_sha256': (freeze / 'file-hashes.sha256').read_text().split()[0]})


def freeze_checks(work):
    first, second = make_freeze(work, 'freeze-a'), make_freeze(work, 'freeze-b')
    assert first['root_sha256'] == second['root_sha256'], (first, second)
    assert first['scheduled'] == 2544 and first['boots'] == 48
    run(PY + [str(INK / 'freeze.py'), 'verify', '--freeze', str(work / 'freeze-a'), '--recipe',
              str(common.ROOT.parent.parent)])
    tampered = work / 'freeze-b'
    path = tampered / 'boots' / 'c8k-s01-L.json'
    path.write_text(path.read_text().replace('"seed": 101', '"seed": 103', 1))
    proc = run(PY + [str(INK / 'freeze.py'), 'verify', '--freeze', str(tampered)], ok=False)
    assert 'frozen file changed' in proc.stderr
    freeze = work / 'freeze-a'
    m = common.read_json(freeze / 'manifest.json')
    boot = common.read_json(freeze / 'boots' / 'c8k-s01-L.json')
    assert boot['env'] == {'RUNNER': '1', 'IMAGE': common.IMAGE_TAG, 'PREPARE': '0', 'COMPUTE_LANE': '1'}
    assert [b['boot'] for b in m['boots'][:3]] == ['c8k-s01-P', 'c8k-s01-L', 'c8k-s01-T']
    assert all(r['body_template']['model'] == m['model'] for r in boot['requests'])
    preseal = freeze / 'private' / 'math-preseal.json'
    assert oct(preseal.stat().st_mode & 0o777) == '0o600' and m['math_preseal']['candidates'] == 240
    bind_fake_host(freeze)
    step('freeze', {'root_sha256': first['root_sha256'], 'deterministic': True, 'tamper_refused': True,
                    'tf_patches_expected': first['tf_patches_expected'], 'scheduled': first['scheduled']})
    return freeze


# ---------------------------------------------------------------- controller e2e on the fake
def controller(freeze, runs, pass_name, turn, port, *extra, fault=None, ok=True):
    env = dict(os.environ)
    env.pop('INK_FAKE_FAULT', None)
    if fault:
        env['INK_FAKE_FAULT'] = fault
    return run(PY + [str(INK / 'controller.py'), 'run', '--freeze', str(freeze), '--host', 'fake', '--pass',
                     pass_name, '--turn', turn, '--runs', str(runs), '--base-url', f'http://127.0.0.1:{port}',
                     '--selftest-fake', *extra], env=env, ok=ok)


def verdict(runs, pass_name, turn, attempt):
    return common.read_json(runs / pass_name / turn / attempt / 'verdict.json')


def e2e(freeze, work, quick):
    runs = work / 'runs-selftest'
    port = free_port()
    passes = ('c8k',) if quick else tuple(common.PASSES)
    turns = [t[0] for t in common.STD_LAYOUT] + [t[0] for t in common.CDL_LAYOUT]
    t0 = time.monotonic()
    for pass_name in passes:
        for turn in turns:
            controller(freeze, runs, pass_name, turn, port)
    collect_s = time.monotonic() - t0
    rerun = controller(freeze, runs, 'c8k', 's01', port)     # QUALIFIED boots are skipped
    assert rerun.stdout.count('already QUALIFIED') == 3
    verdicts = [common.read_json(p) for p in runs.glob('*/*/*/verdict.json')]
    assert len(verdicts) == 24 * len(passes) and all(v['verdict'] == 'QUALIFIED' for v in verdicts)
    assert all(not v['seed_log_check']['seed_mismatch_ids'] and v['seed_log_check']['found_in_log'] ==
               v['observed'] for v in verdicts)
    graded = work / 'graded'
    args = PY + [str(INK / 'grade.py'), '--freeze', str(freeze), '--runs', str(runs), '--host', 'fake', '--out',
                 str(graded), '--selftest-fake-jail']
    if quick:
        args.append('--partial')
    summary = json.loads(run(args).stdout.strip().splitlines()[-1])
    assert summary['complete'] is (not quick)
    if not quick:
        proc = run(PY + [str(INK / 'grade.py'), '--freeze', str(freeze), '--runs', str(runs / 'c8k'), '--host', 'fake',
                         '--out', str(work / 'graded-refused'), '--selftest-fake-jail'], ok=False)
        assert 'missing' in proc.stderr
    reviewed = None
    if not quick:
        math_files = [str(p) for p in sorted(runs.glob('*/s0[25]/*/responses.jsonl'))]
        review = work / 'math-review'
        review.mkdir()
        sealed = json.loads(run(PY + [str(INK / 'math_packet.py'), 'seal', '--freeze', str(freeze), '--responses',
                                      *math_files, '--packet', str(review / 'packet.json'), '--private-key',
                                      str(review / 'key.json')]).stdout.strip().splitlines()[-1])
        packet = common.read_json(review / 'packet.json')
        assert sealed['candidates'] == 240 and all(set(p) == {'candidate_id', 'problem', 'official_answer',
                                                              'candidate_final_content'} for p in packet)

        def label(p, who, flip=False):
            right = 'fake-wrong' not in p['candidate_final_content']
            if flip:
                right = not right
            return {'candidate_id': p['candidate_id'], 'reviewer_id': who, 'label': 'correct' if right else 'incorrect',
                    'reason': 'synthetic selftest label'}
        first = [label(p, 'rev-a') for p in packet]
        second = [label(p, 'rev-b', flip=i % 40 == 0) for i, p in enumerate(packet)]
        common.write_json_new(review / 'a.json', first)
        common.write_json_new(review / 'b.json', second)
        disputes = json.loads(run(PY + [str(INK / 'math_packet.py'), 'disputes', '--packet', str(review / 'packet.json'),
                                        '--first', str(review / 'a.json'), '--first-id', 'rev-a', '--second',
                                        str(review / 'b.json'), '--second-id', 'rev-b', '--out',
                                        str(review / 'third-packet.json')]).stdout.strip().splitlines()[-1])
        third = [label(p, 'rev-c') for p in common.read_json(review / 'third-packet.json')[:-1]]  # one left open
        common.write_json_new(review / 'c.json', third)
        key_sha = sealed['private_key_sha256']
        merged = json.loads(run(PY + [str(INK / 'math_packet.py'), 'merge', '--packet', str(review / 'packet.json'),
                                      '--private-key', str(review / 'key.json'), '--key-sha256', key_sha, '--first',
                                      str(review / 'a.json'), '--first-id', 'rev-a', '--second', str(review / 'b.json'),
                                      '--second-id', 'rev-b', '--third', str(review / 'c.json'), '--third-id', 'rev-c',
                                      '--out', str(review / 'math-reviewed.jsonl'), '--audit',
                                      str(review / 'math-audit.json')]).stdout.strip().splitlines()[-1])
        assert disputes['disputed_candidates'] == 6 and merged['unresolved'] == 1, (disputes, merged)
        reviewed = review / 'math-reviewed.jsonl'
    out = work / 'report'
    args = PY + [str(INK / 'report.py'), '--freeze', str(freeze), '--runs', str(runs), '--graded', str(graded),
                 '--out', str(out)]
    if reviewed:
        args += ['--math-reviewed', str(reviewed)]
    if quick:
        return step('e2e', {'passes': passes, 'boots': len(verdicts), 'collect_seconds': round(collect_s, 1),
                            'report': 'skipped (--quick grid incomplete)'})
    run(args)
    rep = common.read_json(out / 'report.json')
    c8 = rep['passes']['c8k']
    cdl = c8['compute_dl']['comparisons']['L-P']['overall']
    assert c8['compute_dl']['arms']['L']['lane_calls_succeeded'] > 0 and c8['compute_dl']['arms']['P']['lane_calls'] == 0
    assert c8['compute_dl']['arms']['T']['runner_cells'] > 0 and c8['compute_dl']['arms']['L']['runner_cells'] == 0
    assert cdl['difference'] > 0 and cdl['sign_flip_p'] < .05
    assert rep['math_reviewed_rows'] == 240 and c8['stdbench']['arms']['math500']['P']['unresolved'] + \
        c8['stdbench']['arms']['math500']['L']['unresolved'] + c8['stdbench']['arms']['math500']['T']['unresolved'] + \
        sum(rep['passes']['c32k']['stdbench']['arms']['math500'][a]['unresolved'] for a in common.ARMS) == 1
    assert c8['diagnostics']['rows'] == 48 and c8['diagnostics']['qualified'] == 48
    step('e2e', {'passes': passes, 'boots': len(verdicts), 'collect_seconds': round(collect_s, 1),
                 'graded': summary, 'math_disputes': disputes['disputed_candidates'],
                 'math_unresolved_after_third': merged['unresolved'],
                 'c8k_compute_dl_L_minus_P': [cdl['difference'], cdl['ci95'], cdl['sign_flip_p']],
                 'report': str(out / 'REPORT.md')})


def faults(freeze, work):
    runs = work / 'runs-selftest-faults'
    port = free_port()
    # a wrong delivered value in L's third dataset request (diagnostics are requests 1-2)
    controller(freeze, runs, 'c8k', 's01', port, fault='mismatch:L:5', ok=False)
    v = verdict(runs, 'c8k', 's01', 'c8k-s01-L')
    assert v['verdict'] == 'STOPPED' and 'wrong delivered result' in v['error'] and v['observed'] == 2
    assert len(common.read_jsonl(runs / 'c8k' / 's01' / 'c8k-s01-L' / 'responses.jsonl')) == 3   # retained
    assert not (runs / 'c8k' / 's01' / 'c8k-s01-T').exists()
    refused = controller(freeze, runs, 'c8k', 's01', port, ok=False)
    assert '--resume' in refused.stderr
    controller(freeze, runs, 'c8k', 's01', port, '--resume')
    r1 = verdict(runs, 'c8k', 's01', 'c8k-s01-L-r1')
    assert r1['verdict'] == 'QUALIFIED' and r1['scheduled_this_attempt'] == 45 and r1['observed'] == 45
    # P makes a lane call -> arm violation
    controller(freeze, runs, 'c8k', 's02', port, fault='p_call:P:4', ok=False)
    v = verdict(runs, 'c8k', 's02', 'c8k-s02-P')
    assert v['verdict'] == 'STOPPED' and 'P made lane calls' in v['error']
    # a wrong diagnostic value fails the activation step before any dataset request
    controller(freeze, runs, 'c8k', 's03', port, fault='mismatch:L:1', ok=False)
    v = verdict(runs, 'c8k', 's03', 'c8k-s03-L')
    assert v['verdict'] == 'STOPPED' and v['observed'] == 0
    assert not (runs / 'c8k' / 's03' / 'c8k-s03-L' / 'responses.jsonl').exists()
    # an in-stream error event and a cut stream: recorded failures, not stops
    controller(freeze, runs, 'c8k', 's04', port, fault='error:L:6')
    controller(freeze, runs, 'c8k', 's05', port, fault='cut:T:3')
    rows = common.read_jsonl(runs / 'c8k' / 's04' / 'c8k-s04-L' / 'responses.jsonl')
    assert sum(r['status'] == 'error' for r in rows) == 1
    rows = common.read_jsonl(runs / 'c8k' / 's05' / 'c8k-s05-T' / 'responses.jsonl')
    assert sum(r['status'] == 'error' and 'without [DONE]' in r['error'] for r in rows) == 1
    # a request that started without a recorded row blocks resume
    boot = runs / 'c8k' / 's02' / 'c8k-s02-P'
    common.append_jsonl(boot / 'steps.jsonl', {'step': 'request-start', 'suite': 'math500', 'id': 'ghost', 'seed': 101,
                                               'diagnostic': False})
    blocked = controller(freeze, runs, 'c8k', 's02', port, '--resume', ok=False)
    assert 'reconcile it by hand' in blocked.stderr
    graded = work / 'graded-faults'
    summary = json.loads(run(PY + [str(INK / 'grade.py'), '--freeze', str(freeze), '--runs', str(runs), '--host', 'fake',
                                   '--out', str(graded), '--partial', '--selftest-fake-jail']).stdout.splitlines()[-1])
    assert summary['complete'] is False
    step('faults', {'mismatch_stop_and_resume': True, 'resume_without_flag_refused': True, 'p_call_stop': True,
                    'diagnostic_stop': True, 'error_event_recorded': True, 'cut_stream_recorded': True,
                    'unreconciled_start_blocks_resume': True, 'partial_grade': summary['by_pass']['c8k']})


def jail_controls(freeze, work, tag):
    text = subprocess.run(['docker', 'image', 'inspect', '-f', '{{.Id}} {{index .Config.Labels "tf.patches"}}', tag],
                          capture_output=True, text=True, check=True).stdout.split()
    image_id, patches = text[0], (text[1] if len(text) > 1 else None)
    proc = run(PY + [str(INK / 'jail_ink.py'), 'qualify', '--image-id', image_id, '--tag', tag, '--patches',
                     str(patches), '--selected', str(freeze / 'stdbench' / 'humaneval-selected.jsonl'), '--out',
                     str(work / 'jail-qualification')])
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary['qualified'] is True and summary['cases'] == 102
    step('jail', {'image': tag, 'image_id': image_id, 'cases': summary['cases'], 'qualified': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--jail-image')
    args = parser.parse_args()
    work = Path(args.work).absolute()
    common.require('selftest' in str(work) and common.ROOT not in work.parents and work != common.ROOT,
                   'work dir must be outside the harness and contain "selftest"')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    t0 = time.monotonic()
    units()
    kit_selftests(work)
    graders()
    stats()
    freeze = freeze_checks(work)
    client(freeze, work)
    faults(freeze, work)
    e2e(freeze, work, args.quick)
    if args.jail_image:
        jail_controls(freeze, work, args.jail_image)
    print(common.canonical({'selftest': 'PASSED', 'steps': [r['step'] for r in RESULTS],
                            'seconds': round(time.monotonic() - t0, 1), 'threads': threading.active_count()}))


if __name__ == '__main__':
    main()
