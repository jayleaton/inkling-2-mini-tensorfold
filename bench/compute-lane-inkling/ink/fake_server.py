"""CPU self-test doubles: a stdlib SSE server shaped like the TensorFold Inkling wire, a fake recipe lifecycle,
a fake reference executor and a fake HumanEval jail. Never used outside a runs dir whose path contains 'selftest'.

Wire shape copied from ink-disk-tf zig/src/server (stream.zig / openai.zig): role chunk, reasoning_content and
content deltas, one choices[0].delta.compute report per lane call, a final chunk with finish_reason and the
top-level `tensorfold` runtime (enable_thinking, reasoning_effort, timings, token_ids, token_sha256, compute
{calls, inserted_tokens, prefix_tokens, runner_calls, tool_calls}, runner receipt under S3), a usage chunk with
`choices: []`, then `data: [DONE]`. A request that declares code_interpreter while the lane is off gets 400.
/tokenize counts characters and /detokenize maps IDs to chr(), so binding and the clipping count are exercised.

Faults (environment INK_FAKE_FAULT=kind:arm:n, the n-th chat request of a boot of that arm, diagnostics included):
  mismatch  delivered lane output differs from the reference (a safety stop)
  p_call    arm P reports a lane call (an arm violation)
  error     an in-stream {"error": ...} event, then [DONE] (a recorded protocol error, not a stop)
  cut       the stream ends with no [DONE]
"""
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import common

LANE = {'0': 'P', '1': 'L', 'tool-full': 'T'}
CORRECT = {'gsm8k': (.80, .85, .85), 'math500': (.70, .75, .75), 'humaneval': (.85, .90, .90),
           'compute_dl': (.45, .80, .85)}
USE_TOOL = {'gsm8k': .25, 'math500': .25, 'humaneval': .10, 'compute_dl': .85, 'diagnostic': 1.0}


def unit(*parts):
    """Deterministic uniform [0, 1) from the request identity."""
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def fake_reference(code):
    """The kit jail_worker's reference()/normalized() in a local CPython subprocess (fake-generated code only)."""
    script = ('import sys; sys.path.insert(0, sys.argv[1]); import jail_worker; '
              'jail_worker.reference(sys.stdin.read())')
    try:
        proc = subprocess.run([sys.executable, '-I', '-S', '-c', script, str(common.PHASE2)], input=code.encode(),
                              capture_output=True, timeout=10)
        status = 'ok' if proc.returncode == 0 else 'exit_error'
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        status, returncode, stdout, stderr = 'timeout', None, b'', b''
    worker = common.kit('jail_worker')
    body = worker.normalized(stdout, stderr, status, returncode)
    return {'status': status, 'exit_code': returncode, 'normalized_body': body.decode('utf-8', 'replace'),
            'normalized_body_base64': base64.b64encode(body).decode(), 'fake': True}


class FakeJail:
    """grade_humaneval: correct iff the fake reply carries the fake pass marker; run_reference: fake_reference."""

    image_record = {'tag': 'fake', 'id': 'sha256:' + '0' * 64, 'tf_patches': 'fake'}

    def grade_humaneval(self, task, content):
        passed = '# fake-pass' in content
        return {'status': 'ok' if passed else 'exit_error', 'grade': 'correct' if passed else 'incorrect',
                'harness_receipt': passed, 'reason': 'fake jail', 'fake': True,
                'task_sha256': common.digest(task)}

    run_reference = staticmethod(fake_reference)


class Tasks:
    """prompt -> (suite, id, gold) from the freeze (fake server only: it plays a model that sometimes knows)."""

    def __init__(self, freeze):
        self.by_prompt = {}
        for task in common.read_json(Path(freeze) / 'analysis-stdbench.json')['tasks']:
            self.by_prompt[task['prompt']] = (task['suite'], task['id'], task)
        answers = common.read_json(common.CDL / 'dataset-v1' / 'answers-a.json')
        for task in common.read_json(Path(freeze) / 'analysis-compute-dl.json')['tasks']:
            self.by_prompt[task['prompt']] = ('compute_dl', task['id'], answers[task['id']])
        for diag_id, prompt, reference in common.DIAGNOSTICS:
            self.by_prompt[prompt] = ('diagnostic', diag_id, reference)


def final_text(suite, gold, right):
    if suite == 'gsm8k':
        value = gold['answer'].rsplit('####', 1)[-1].strip().replace(',', '')
        return 'The answer is \\boxed{%s}.' % (value if right else '-987654321')
    if suite == 'math500':
        return 'Final: \\boxed{%s}' % (gold.get('answer') if right else 'fake-wrong')
    if suite == 'humaneval':
        return '```python\n' + gold['code_prefix'] + ('    # fake-pass\n    pass\n' if right else '    pass\n') + '```'
    if suite == 'compute_dl':
        return 'Final answer: ' + json.dumps(gold if right else 'wrong')
    return 'The result is %s.' % gold


class Model:
    """One fake reply: segments, lane calls (outputs from fake_reference), runtime and accounting."""

    def __init__(self, server, body, number):
        self.server, self.body, self.number = server, body, number
        prompt = body['messages'][-1]['content']
        self.suite, self.id, self.gold = server.tasks.by_prompt.get(prompt, ('unknown', 'unknown', None))
        self.seed, self.arm = body.get('seed'), server.arm
        self.fault = server.fault_for(number)

    def calls(self):
        arm = self.arm
        if arm == 'P' and self.fault != 'p_call':
            return []
        if arm != 'P' and self.fault != 'mismatch' and unit('tool', self.id, self.seed, arm) >= USE_TOOL.get(
                self.suite, 0):
            return []
        if self.suite == 'diagnostic':
            codes = ['print(17*19)' if self.gold == '323' else 'print(sum(range(11)))']
        else:
            pick = unit('kind', self.id, self.seed, arm)
            n = int(unit('n', self.id, self.seed) * 1000)
            if pick < .06:
                codes = ["print('x' * 2000)"]                 # clipped at the 512-token cap
            elif pick < .12:
                codes = ['1/0']                              # an attributed exit_error
            elif pick < .2:
                codes = [f'{n} * 3', f'print(sum(range({n})))']   # two calls, REPL bare expression first
            else:
                codes = [f'print(sum(range({n})))']
        out = []
        for index, code in enumerate(codes):
            ref = fake_reference(code)
            output = ref['normalized_body']
            if len(output) > 512:
                output = output[:512] + '\n' + common.TRUNCATION_MARKER
            if self.fault == 'mismatch' and index == 0:
                output = output + '0'
            runner = arm == 'T' and unit('exec', self.id, self.seed, index) < .5 and ref['status'] == 'ok'
            call = {'name': 'python', 'id': None if index else f'call_{self.number}_{index}', 'code': code,
                    'status': ref['status'], 'exit_code': None if runner else ref['exit_code'],
                    'exec_ms': 3, 'output': output, 'inserted': True}
            if arm == 'T':
                call.update(call_ms=4, executor='in-model runner' if runner else 'code tool',
                            program=None, steps=1 if runner else 0, runner_abstain=None if runner else 'not owned',
                            verify=None)
            out.append(call)
        return out

    def reply(self):
        arm, suite = self.arm, self.suite
        index = ('P', 'L', 'T').index(arm)
        right = suite == 'diagnostic' or unit('right', self.id, self.seed, arm) < CORRECT.get(suite, (0, 0, 0))[index]
        calls = self.calls()
        pieces = [('reasoning', 'Thinking about %s seed %s. ' % (self.id, self.seed))]
        inserted = 0
        for call in calls:
            pieces.append(('reasoning', '<python>' + call['code'] + '</python>'))
            result = '<output>' + call['output'] + '</output>'
            pieces.append(('inserted', result))
            inserted += len(result)
        pieces.append(('reasoning', ' Done.'))
        pieces.append(('content', final_text(suite, self.gold, right)))
        ids = [ord(ch) for _, text in pieces for ch in text]
        completion = len(ids) - inserted
        roll = unit('finish', self.id, self.seed, arm, self.body['max_tokens'])
        finish = 'length' if roll < .04 else ('deadline' if roll < .06 else 'stop')
        runtime = {'enable_thinking': True, 'reasoning_effort': 'high', 'engine': 'fake-inkling2',
                   'tokens_per_second': 60.0, 'seconds': .01, 'prefill_seconds': .001, 'time_to_first_token': .002,
                   'sampling': {'temperature': self.body['temperature'], 'top_p': self.body['top_p'],
                                'top_k': self.body['top_k'], 'seed': self.seed},
                   'drafts': {'proposed': 0, 'accepted': 0}, 'token_sha': common.sha_bytes(
                       ','.join(map(str, ids)).encode())[:16],
                   'token_sha256': hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest(), 'token_ids': ids}
        if calls:
            runtime['compute'] = {'calls': calls, 'inserted_tokens': inserted,
                                  'prefix_tokens': len(self.body['messages'][-1]['content']),
                                  'runner_calls': sum(c.get('executor') == 'in-model runner' for c in calls),
                                  'tool_calls': sum(c.get('executor') != 'in-model runner' for c in calls)}
        if arm == 'T':
            owned = int(unit('cells', self.id, self.seed) * 3)
            runtime['runner'] = {form: {'closed': owned if form == 'inline' else 0,
                                        'owned': owned if form == 'inline' else 0,
                                        'abstained': 0, 'owned_tokens': 4 * owned if form == 'inline' else 0}
                                 for form in common.RUNNER_FORMS}
        return pieces, calls, runtime, finish, completion


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def send_json(self, code, value):
        blob = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def read_body(self):
        return json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')

    def do_GET(self):
        if self.path == '/health':
            self.send_json(200, {'status': 'ok'})
        elif self.path == '/v1/models':
            self.send_json(200, {'object': 'list', 'data': [{'id': self.server.model, 'object': 'model'}]})
        else:
            self.send_json(404, {'error': {'message': 'not found'}})

    def do_POST(self):
        body = self.read_body()
        if self.path == '/tokenize':
            self.send_json(200, {'tokens': [ord(c) for c in body['prompt']], 'count': len(body['prompt'])})
        elif self.path == '/detokenize':
            self.send_json(200, {'prompt': ''.join(chr(i) for i in body['tokens'])})
        elif self.path == '/v1/chat/completions':
            self.chat(body)
        else:
            self.send_json(404, {'error': {'message': 'not found'}})

    def frame(self, value):
        self.wfile.write(b'data: ' + json.dumps(value, separators=(',', ':')).encode() + b'\n\n')
        self.wfile.flush()
        time.sleep(.0005)

    def chat(self, body):
        server = self.server
        if body.get('tools') and server.arm == 'P':
            self.send_json(400, {'error': {'message': 'code_interpreter requires the compute lane'}})
            return
        with server.lock:
            server.requests += 1
            number = server.requests
        model = Model(server, body, number)
        pieces, calls, runtime, finish, completion = model.reply()
        rid = 'chatcmpl-fake%06d' % number
        base = {'id': rid, 'object': 'chat.completion.chunk', 'model': server.model}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.frame(dict(base, choices=[{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]))
        call_iter = iter(calls)
        for kind, text in pieces:
            if kind == 'reasoning':
                self.frame(dict(base, choices=[{'index': 0, 'delta': {'reasoning_content': text},
                                                'finish_reason': None}]))
            elif kind == 'inserted':
                call = next(call_iter)
                self.frame(dict(base, choices=[{'index': 0, 'delta': {'compute': call}, 'finish_reason': None}]))
            else:
                if model.fault == 'cut':
                    self.close_connection = True
                    return
                self.frame(dict(base, choices=[{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]))
        if model.fault == 'error':
            self.frame({'error': {'message': 'fake engine fault', 'type': 'server_error'}})
            self.wfile.write(b'data: [DONE]\n\n')
            server.log('error %s seed=%s' % (rid, model.seed))
            return
        self.frame(dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': finish}], tensorfold=runtime))
        prompt_tokens = len(body['messages'][-1]['content'])
        self.frame(dict(base, choices=[], usage={'prompt_tokens': prompt_tokens, 'completion_tokens': completion,
                                                 'total_tokens': prompt_tokens + completion}))
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        server.log('done %s tokens=%d finish=%s seed=%s' % (rid, completion, finish, model.seed))


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, arm, model, tasks, fault):
        super().__init__(address, Handler)
        self.arm, self.model, self.tasks, self.fault = arm, model, tasks, fault
        self.requests, self.lock, self.lines = 0, threading.Lock(), []

    def log(self, line):
        self.lines.append('%.3f %s' % (time.time(), line))

    def fault_for(self, number):
        if not self.fault:
            return None
        kind, arm, n = self.fault.split(':')
        return kind if arm == self.arm and int(n) == number else None


class FakeLifecycle:
    """start.sh/stop.sh stand-in: one in-process FakeServer per boot on the base URL's port."""

    def __init__(self, base_url, runs, freeze, image_id, model=common.DEFAULT_MODEL):
        common.require('selftest' in str(runs), 'fake lifecycle only for selftest runs')
        url = urlparse(base_url)
        self.address = (url.hostname, url.port)
        self.tasks, self.image_id, self.model = Tasks(freeze), image_id, model
        self.server, self.thread, self.env, self.boot_lines = None, None, None, []

    def state(self):
        return 'running' if self.server else 'absent'

    def port_free(self):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(self.address)
            except OSError:
                return False
        return True

    def start(self, extra, log):
        common.require(self.server is None, 'fake server already running')
        common.require(extra.get('RUNNER') == '1' and extra.get('PREPARE') == '0' and extra.get('IMAGE')
                       == common.IMAGE_TAG and Path(extra['SESSION_DIR']).is_dir(), 'boot env differs')
        arm = LANE[extra['COMPUTE_LANE']]
        self.env = dict(extra)
        self.server = FakeServer(self.address, arm, self.model, self.tasks, os.environ.get('INK_FAKE_FAULT'))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.server.log('fake server listening arm=%s' % arm)
        if arm != 'P':
            self.server.log('compute lane: on (python, 512-token results)')
        with open(log, 'a', encoding='utf-8') as out:
            out.write('fake start.sh COMPUTE_LANE=%s\n' % extra['COMPUTE_LANE'])

    def inspect(self):
        import controller
        arm = self.server.arm
        env = dict(controller.expected_env(arm))
        if arm == 'P':
            env['TF_INKLING2_COMPUTE_LANE'] = '0'
        args = ['serve', '--context', '131072', '--parallel', '2', '--kv-cache', 'fp8', '--drafter', '/models/draft',
                '--prompt-cache-disk', '/sessions', '--vision', '--thinking']
        return {'Image': self.image_id, 'Name': '/' + common.CONTAINER, 'Args': args,
                'Config': {'Env': [f'{k}={v}' for k, v in sorted(env.items())]}, 'fake': True}

    def save_logs(self, path):
        lines = self.server.lines if self.server else []
        Path(path).write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
        return path

    def stop(self, log):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=10)
        self.server = None
        with open(log, 'a', encoding='utf-8') as out:
            out.write('fake stop.sh\n')
