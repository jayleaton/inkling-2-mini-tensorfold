#!/usr/bin/env python3
"""Run one frozen turn (its three boots, in frozen order) on this Spark, one request at a time.

  controller.py run --freeze F --host NAME --pass c8k --turn s01 [--runs DIR] [--stop-existing] [--resume]

Per boot: refuse drift (freeze inventory, harness, recipe tree, host image); stop any server only with
--stop-existing; start ./start.sh with a scrubbed environment (the arm's env, RUNNER=1, the runner image,
PREPARE=0, a fresh empty SESSION_DIR); verify the container's image, engine environment and served name; run the two
diagnostics and the activation gate; run the boot's requests in frozen order, each written to disk before any check;
check each reply (arm gates, independent jail reference per delivered call, server-decoder binding); save the
server log; ./stop.sh; verdict. A safety stop or error stops the turn with everything retained. A boot whose
verdict is QUALIFIED is skipped on a rerun; a STOPPED boot continues only with --resume, in a fresh boot that
runs the diagnostics again and then only the keys no earlier attempt observed, in their frozen order.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import freeze as freeze_mod  # noqa: E402
import ink_client  # noqa: E402
import safety  # noqa: E402

bench = common.kit('bench')
DEFAULT_RECIPE = common.ROOT.parent.parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class RecipeLifecycle:
    """The recipe's own ./start.sh and ./stop.sh, plus read-only docker inspect/logs."""

    def __init__(self, recipe):
        self.recipe = Path(recipe).resolve()

    def docker(self, *args, timeout=60, check=True):
        return subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout, check=check)

    def state(self):
        names = self.docker('ps', '-a', '--format', '{{.Names}}').stdout.split()
        if common.CONTAINER not in names:
            return 'absent'
        return 'running' if self.docker('inspect', '-f', '{{.State.Running}}', common.CONTAINER).stdout.strip() == 'true' \
            else 'stopped'

    def port_free(self):
        out = subprocess.run(['ss', '-ltn', 'sport = :8888'], capture_output=True, text=True, timeout=30).stdout
        return 'LISTEN' not in out

    def clean_env(self, extra):
        env = {k: os.environ[k] for k in ('HOME', 'PATH', 'USER', 'LOGNAME', 'LANG', 'DOCKER_HOST', 'DOCKER_CONFIG')
               if k in os.environ}
        env.update(extra)
        return env

    def start(self, extra, log):
        with open(log, 'ab') as out:
            subprocess.run(['./start.sh'], cwd=self.recipe, env=self.clean_env(extra), stdout=out,
                           stderr=subprocess.STDOUT, timeout=3600, check=True)

    def stop(self, log):
        with open(log, 'ab') as out:
            subprocess.run(['./stop.sh'], cwd=self.recipe, env=self.clean_env({}), stdout=out,
                           stderr=subprocess.STDOUT, timeout=300, check=True)

    def inspect(self):
        return json.loads(self.docker('inspect', common.CONTAINER).stdout)[0]

    def save_logs(self, path):
        result = self.docker('logs', common.CONTAINER, timeout=300, check=False)
        Path(path).write_text(result.stdout + result.stderr, encoding='utf-8')
        return path


def expected_env(arm):
    """What the container's engine environment must and must not hold for each arm (start.sh, recipe defaults)."""
    must = {'TF_INKLING2_ATTN_FP8': '1', 'TF_INKLING2_PREFILL_TC': '1', 'TENSORFOLD_MEMORY_RESERVE_GIB': '9'}
    if arm in ('L', 'T'):
        must['TF_INKLING2_COMPUTE_LANE'] = '1'
    if arm == 'T':
        must.update(TF_INKLING2_RUNNER='1', TF_INKLING2_COMPUTE_LANE_RUNNER='1',
                    TF_INKLING2_RUNNER_FORMS='fence,repl,inline,trace', TF_INKLING2_RUNNER_BACKEND='graph')
    return must


def check_container(record, arm, image_id):
    problems = []
    if record.get('Image') != image_id:
        problems.append('container image differs from the bound image ID')
    env = dict(item.split('=', 1) for item in record.get('Config', {}).get('Env') or [] if '=' in item)
    for key, value in expected_env(arm).items():
        if env.get(key) != value:
            problems.append(f'{key}={env.get(key)!r}, expected {value!r}')
    engine = {k: v for k, v in env.items() if k.startswith(('TF_', 'TENSORFOLD_'))}
    if arm == 'P' and engine.get('TF_INKLING2_COMPUTE_LANE') not in (None, '0'):
        problems.append('P has the compute lane on')
    if arm in ('P', 'L') and any(k.startswith('TF_INKLING2_RUNNER') or k.startswith('TF_INKLING2_COMPUTE_LANE_RUNNER')
                                 for k in engine):
        problems.append(arm + ' has runner knobs set')
    if engine.get('TF_INKLING2_COMPUTE_LANE_DEFAULT') not in (None, '0'):
        problems.append('compute default is on')
    args = record.get('Args') or []
    for flag, value in (('--context', '131072'), ('--parallel', '2'), ('--kv-cache', 'fp8'),
                        ('--drafter', '/models/draft'), ('--prompt-cache-disk', '/sessions')):
        if flag not in args or args.index(flag) + 1 >= len(args) or args[args.index(flag) + 1] != value:
            problems.append(f'serve argument {flag} {value} missing')
    for flag in ('--vision', '--thinking'):
        if flag not in args:
            problems.append(f'serve argument {flag} missing')
    return {'problems': problems, 'engine_env': engine, 'args': args}


class Boot:
    def __init__(self, ctl, entry, boot, attempt_dir, keys):
        self.ctl, self.entry, self.boot, self.dir, self.keys = ctl, entry, boot, attempt_dir, keys
        self.arm, self.pass_name = boot['arm'], boot['pass']
        self.settings = boot['pass_settings']
        self.started = False

    def event(self, step, verdict, **detail):
        record = {'at': now(), 'step': step, 'verdict': verdict, 'boot': self.boot['boot'], **detail}
        common.append_jsonl(self.dir / 'steps.jsonl', record)
        print(common.canonical(record), flush=True)

    def measure(self, template, diagnostic, index):
        body = template['body_template']
        common.require(common.digest(body) == template['body_sha256'], 'body differs from the frozen request')
        key = (template['suite'], template['id'], template['seed'])
        self.event('request-start', 'SENT', suite=key[0], id=key[1], seed=key[2], diagnostic=diagnostic)
        name = ('diag-' if diagnostic else '') + f'{index:03d}-{key[1].replace("/", "_")}-{key[2]}.sse.gz'
        row = ink_client.send(self.ctl.base_url, body, cap=self.settings['max_tokens'],
                              wait_s=self.settings['client_wait_s'], raw_path=self.dir / 'streams' / name)
        row.update(suite=key[0], id=key[1], seed=key[2], arm=self.arm, boot=self.boot['boot'],
                   attempt=self.dir.name, task_sha256=template['task_sha256'], request=body,
                   body_sha256=template['body_sha256'], diagnostic=diagnostic, included_in_cohort=not diagnostic,
                   request_index=index, host=self.ctl.host['host'], raw_stream_file='streams/' + name,
                   study=self.boot['study'])
        row['pass'] = self.pass_name
        row['tokens'] = bench.usage_counts(row.get('usage'))
        common.append_jsonl(self.dir / ('diagnostics.jsonl' if diagnostic else 'responses.jsonl'), row)
        check = safety.check_row(self.arm, row, self.ctl.references, *self.ctl.server_tools)
        common.append_jsonl(self.dir / 'safety.jsonl', {'pass': self.pass_name, 'boot': self.boot['boot'],
                                                         'suite': key[0], 'id': key[1], 'seed': key[2],
                                                         'arm': self.arm, 'diagnostic': diagnostic, **check})
        self.event('response', 'RECORDED', suite=key[0], id=key[1], seed=key[2], status=row['status'],
                   finish=row.get('finish_reason'), elapsed=round(row['elapsed_seconds'], 3),
                   calls=len(common.compute_calls(row.get('tensorfold'))), runner_cells=check['runner_cells'],
                   native_complete=row['native_accounting']['complete'], qualified=check['qualified'])
        if check['stop']:
            raise RuntimeError('safety stop (row retained): ' + '; '.join(
                check['violations'] + [f'wrong delivered result in call {i}' for i in check['mismatches']]))
        return row, check

    def diagnostics(self):
        delivered, cells = 0, 0
        for index, template in enumerate(self.boot['diagnostics']):
            row, check = self.measure(template, True, index)
            tf = row.get('tensorfold') or {}
            common.require(row['status'] == 'ok' and row['native_accounting']['complete'],
                           'diagnostic protocol/accounting failure: ' + str(row.get('error')))
            common.require(tf.get('enable_thinking') is True and tf.get('reasoning_effort') == 'high',
                           'server did not take thinking on / reasoning_effort high')
            common.require(isinstance(tf.get('token_ids'), list), 'server did not return token_ids')
            calls = common.compute_calls(tf)
            verdicts = {r['index']: r['verdict'] for r in check['references']}
            outputs = [c.get('output') for c in calls if common.successful_call(c)]
            if self.arm != 'P' and outputs and template['reference'] not in outputs:
                raise RuntimeError('diagnostic delivered output differs from the fixed reference')
            # an actual delivered successful call whose bytes are the reference and match the independent run
            delivered += sum(1 for i, c in enumerate(calls) if common.successful_call(c)
                             and c.get('output') == template['reference'] and verdicts.get(i) == 'match')
            cells += check['runner_cells']
        ok = delivered == 0 if self.arm == 'P' else delivered >= 1
        self.event('diagnostics', 'QUALIFIED' if ok else 'FAIL', delivered_reference_successes=delivered,
                   runner_cells=cells)
        common.require(ok, 'activation gate failed for arm ' + self.arm)

    def seed_check(self, log_path, rows):
        seeds = {}
        for line in Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines():
            match = re.search(r'\bdone (\S+) .* seed=(\d+)', line)
            if match:
                seeds.setdefault(match.group(1), set()).add(int(match.group(2)))
        checked = [r for r in rows if r.get('response_id') in seeds]
        bad = [r['response_id'] for r in checked if seeds[r['response_id']] != {r['seed']}]
        return {'rows': len(rows), 'found_in_log': len(checked), 'seed_mismatch_ids': bad}

    def run(self):
        ctl, lc = self.ctl, self.ctl.lifecycle
        self.dir.mkdir(parents=True)
        (self.dir / 'streams').mkdir()
        common.write_json_new(self.dir / 'boot-ref.json', {'boot': self.boot['boot'], 'file': self.entry['file'],
                                                           'sha256': self.entry['sha256'], 'keys': len(self.keys),
                                                           'host': ctl.host, 'base_url': ctl.base_url})
        sessions = ctl.runs / 'sessions' / self.dir.name
        common.require(not sessions.exists(), 'session dir exists: ' + str(sessions))
        sessions.mkdir(parents=True)
        error, rows, t0 = None, [], time.monotonic()
        timing = {}
        try:
            self.event('start', 'BEGIN', env=self.boot['env'])
            lc.start(dict(self.boot['env'], SESSION_DIR=str(sessions)), self.dir / 'start.log')
            self.started = True
            timing['boot_seconds'] = time.monotonic() - t0
            record = lc.inspect()
            common.write_json_new(self.dir / 'docker-inspect.json', record)
            check = check_container(record, self.arm, ctl.host['image_id'])
            models = ink_client.get_json(ctl.base_url, '/v1/models')
            common.write_json_new(self.dir / 'models.json', models)
            served = models['data'][0]['id']
            self.event('boot', 'READY', served_name=served, problems=check['problems'],
                       boot_seconds=round(timing['boot_seconds'], 1))
            common.require(not check['problems'], 'container differs from the arm: ' + '; '.join(check['problems']))
            common.require(served == ctl.host['served_name'] == self.boot['requests'][0]['body_template']['model'],
                           'served name differs from the frozen model name: ' + served)
            early = lc.save_logs(self.dir / 'server-boot.log')
            lane_line = 'compute lane:' in Path(early).read_text(encoding='utf-8', errors='replace')
            common.require(lane_line == (self.arm != 'P'), 'server log lane state differs from the arm')
            t1 = time.monotonic()
            self.diagnostics()
            timing['diagnostic_seconds'] = time.monotonic() - t1
            t2 = time.monotonic()
            by_key = {(r['suite'], r['id'], r['seed']): r for r in self.boot['requests']}
            for index, key in enumerate(self.keys):
                row, _ = self.measure(by_key[key], False, index)
                rows.append(row)
            timing['dataset_seconds'] = time.monotonic() - t2
        except BaseException as exc:   # noqa: BLE001 - every stop is recorded, then cleanup runs
            error = f'{type(exc).__name__}: {exc}'
            self.event('collection', 'STOPPED', reason=error)
        finally:
            seeds = None
            try:
                present = self.started or lc.state() != 'absent'
            except BaseException:   # noqa: BLE001
                present = True
            if present:
                try:
                    lc.save_logs(self.dir / 'server.log')
                    seeds = self.seed_check(self.dir / 'server.log', rows)
                except BaseException as exc:   # noqa: BLE001
                    error = error or f'log save failed: {exc}'
            try:
                lc.stop(self.dir / 'stop.log')
                common.require(lc.state() == 'absent', 'container still present after stop.sh')
            except BaseException as exc:   # noqa: BLE001
                error = error or f'stop failed: {exc}'
            if os.environ.get('KEEP_SESSIONS') != '1':
                size = sum(p.stat().st_size for p in sessions.rglob('*') if p.is_file())
                shutil.rmtree(sessions, ignore_errors=True)
                timing['session_bytes_removed'] = size
            timing['wall_seconds'] = time.monotonic() - t0
            verdict = {'verdict': 'QUALIFIED' if error is None else 'STOPPED', 'error': error,
                       'boot': self.boot['boot'], 'attempt': self.dir.name, 'observed': len(rows),
                       'scheduled_this_attempt': len(self.keys), 'timing': timing, 'seed_log_check': seeds,
                       'freeze_root_sha256': ctl.root}
            common.write_json_new(self.dir / 'verdict.json', verdict)
            self.event('boot', verdict['verdict'], **{k: v for k, v in verdict.items() if k != 'verdict'})
        if error:
            raise SystemExit('boot stopped: ' + error)


class Controller:
    def __init__(self, args):
        self.freeze = Path(args.freeze).resolve()
        self.recipe = Path(args.recipe).resolve()
        self.runs = Path(args.runs).resolve()
        self.manifest = freeze_mod.verify_freeze(self.freeze, self.recipe, quiet=True)
        self.root = (self.freeze / 'file-hashes.sha256').read_text().split()[0]
        self.host = common.read_json(self.freeze / 'hosts' / (args.host + '.json'))
        common.require(self.host['freeze_root_sha256'] == self.root, 'host binding belongs to another freeze')
        common.require(self.host['tf_patches'] == self.manifest['recipe']['tf_patches_expected'], 'host image patches')
        self.args = args
        if args.selftest_fake:
            import fake_server
            common.require(str(self.runs).find('selftest') >= 0, 'fake lifecycle only under a selftest runs dir')
            self.base_url = args.base_url
            self.lifecycle = fake_server.FakeLifecycle(args.base_url, self.runs, self.freeze, self.host['image_id'],
                                                       self.host['served_name'])
            self.references = fake_server.fake_reference
        else:
            common.require(args.base_url == common.BASE_URL, 'production endpoint is ' + common.BASE_URL)
            self.base_url = common.BASE_URL
            self.lifecycle = RecipeLifecycle(self.recipe)
            jail = safety.InkJail(self.host['image_id'], common.IMAGE_TAG, self.host['tf_patches'])
            self.references = jail.run_reference
        self.server_tools = safety.server_tools(self.base_url)

    def attempts(self, turn_dir, name):
        return sorted(p for p in turn_dir.glob(name + '*') if p.is_dir() and re.fullmatch(re.escape(name) + r'(-r\d+)?',
                                                                                         p.name))

    def plan(self, entry, boot, turn_dir):
        keys = [(r['suite'], r['id'], r['seed']) for r in boot['requests']]
        attempts = self.attempts(turn_dir, boot['boot'])
        if not attempts:
            return turn_dir / boot['boot'], keys
        last = attempts[-1]
        verdict = last / 'verdict.json'
        common.require(verdict.exists(), f'{last} has no verdict: a controller may still be running, or it was '
                                         'killed; reconcile by hand before anything else')
        if common.read_json(verdict)['verdict'] == 'QUALIFIED':
            return None, []
        common.require(self.args.resume, f'{last.name} STOPPED; rerun with --resume to continue in a fresh boot')
        observed = set()
        for attempt in attempts:
            path = attempt / 'responses.jsonl'
            done = {(r['suite'], r['id'], r['seed']) for r in common.read_jsonl(path)} if path.exists() else set()
            started = {(s['suite'], s['id'], s['seed']) for s in common.read_jsonl(attempt / 'steps.jsonl')
                       if s.get('step') == 'request-start' and not s.get('diagnostic')} if \
                (attempt / 'steps.jsonl').exists() else set()
            common.require(started <= done, f'{attempt.name}: a request started without a recorded row; '
                                            'reconcile it by hand (never resend an observed key)')
            observed |= done
        remaining = [k for k in keys if k not in observed]
        if not remaining:
            return None, []
        return turn_dir / f'{boot["boot"]}-r{len(attempts)}', remaining

    def run_turn(self, pass_name, turn):
        entries = [t for t in self.manifest['turns'] if t['pass'] == pass_name and t['turn'] == turn]
        common.require(len(entries) == 1, f'unknown turn {pass_name} {turn}')
        boot_map = {b['boot']: b for b in self.manifest['boots']}
        turn_dir = self.runs / pass_name / turn
        lc = self.lifecycle
        for name in entries[0]['boots']:
            entry = boot_map[name]
            common.require(common.sha_file(self.freeze / entry['file']) == entry['sha256'], 'boot file drift')
            boot = common.read_json(self.freeze / entry['file'])
            attempt_dir, keys = self.plan(entry, boot, turn_dir)
            if attempt_dir is None:
                print(common.canonical({'boot': name, 'skip': 'already QUALIFIED or fully observed'}), flush=True)
                continue
            state = lc.state()
            if state != 'absent':
                common.require(self.args.stop_existing, f'{common.CONTAINER} is {state}; pass --stop-existing')
                (turn_dir).mkdir(parents=True, exist_ok=True)
                lc.stop(turn_dir / f'pre-stop-{int(time.time())}.log')
            common.require(lc.state() == 'absent' and lc.port_free(), 'port 8888 or the container is still in use')
            freeze_mod.verify_freeze(self.freeze, self.recipe, quiet=True)
            Boot(self, entry, boot, attempt_dir, keys).run()
        print(common.canonical({'turn': turn, 'pass': pass_name, 'verdict': 'COMPLETE'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--freeze', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--pass', dest='pass_name', required=True, choices=sorted(common.PASSES))
    p.add_argument('--turn', required=True)
    p.add_argument('--runs', default=str(Path.home() / 'work' / 'ink-runs' / 'runs'))
    p.add_argument('--recipe', default=str(DEFAULT_RECIPE))
    p.add_argument('--stop-existing', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--base-url', default=common.BASE_URL)
    p.add_argument('--selftest-fake', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    Controller(args).run_turn(args.pass_name, args.turn)


if __name__ == '__main__':
    main()
