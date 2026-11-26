"""Shared constants and helpers for the Inkling 2 Mini compute-lane rerun (stdlib only).

Everything protocol-shaped lives here once: arms and their boot environments, the two passes, the turn layouts,
the request body, the server fields the body may use, and loaders for the unchanged DSv4.1 kit modules.
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # bench/compute-lane-inkling
INK = ROOT / 'ink'
DSV = ROOT / 'dsv41'                                   # unchanged DSv4.1 kit copies (dsv41/COPIED.sha256)
PHASE2 = DSV / 'phase2'
KIT = DSV / 'kit'
CDL = DSV / 'compute-dl'

ARMS = ('P', 'L', 'T')
SEEDS = (101, 102)
STD_SUITES = ('gsm8k', 'math500', 'humaneval')
STD_COUNTS = {'gsm8k': 48, 'math500': 20, 'humaneval': 24}
CDL_FAMILIES = ('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')
IMAGE_TAG = 'tensorfold-inkling2:c4-dc2ca8a-runner'
CONTAINER = 'inkling2-mini-tf'
BASE_URL = 'http://127.0.0.1:8888'
DEFAULT_MODEL = 'Inkling-2-Mini'

# Every boot: the same runner image for all arms, no prepare step. Everything else is the recipe default.
COMMON_ENV = {'RUNNER': '1', 'IMAGE': IMAGE_TAG, 'PREPARE': '0'}
ARM_ENV = {'P': {'COMPUTE_LANE': '0'}, 'L': {'COMPUTE_LANE': '1'}, 'T': {'COMPUTE_LANE': 'tool-full'}}
TOOLS = [{'type': 'code_interpreter'}]

# Two separately labelled passes; only the cap, the server deadline and the client wait differ.
PASSES = {
    'c8k': {'max_tokens': 8192, 'deadline_s': 270, 'client_wait_s': 360},
    'c32k': {'max_tokens': 32768, 'deadline_s': 300, 'client_wait_s': 390},
}

# The fixed request settings (DSv4.1: thinking/high, T1, top_p .95, top_k 20, streamed with usage and native IDs).
# Inkling's template takes reasoning_effort through chat_template_kwargs (tools/client.py thinking_kwargs).
SETTINGS = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'stream': True,
            'stream_options': {'include_usage': True}, 'return_token_ids': True,
            'chat_template_kwargs': {'reasoning_effort': 'high'}}

# The only body fields sent, each read by the server (ink-disk-tf zig/src/server: openai.zig plan() copies
# temperature/top_p/top_k/seed/return_token_ids/deadline_s into the sampling fields; max_tokens, stream,
# stream_options.include_usage, messages, tools, model are read there too; fields.zig thinkingFields reads
# chat_template_kwargs.reasoning_effort). The server ignores unknown fields silently, so the client refuses them.
ACCEPTED_FIELDS = {'model', 'messages', 'temperature', 'top_p', 'top_k', 'seed', 'max_tokens', 'deadline_s',
                   'stream', 'stream_options', 'return_token_ids', 'chat_template_kwargs', 'tools'}
ACCEPTED_KWARGS = {'reasoning_effort'}
ACCEPTED_STREAM_OPTIONS = {'include_usage'}

DIAGNOSTICS = (
    ('python-17x19', 'Use the Python tool to print 17*19, then report the result.', '323'),
    ('python-sum11', 'Use the Python tool to print sum(range(11)), then report the result.', '55'),
)

# stdbench six-turn layout: DSv4.1's P/L order per turn, T placed at the pre-registered rotating position
# 3,1,2,1,2,3 (each position twice; the two blocks of a suite put T at different positions).
STD_LAYOUT = (
    ('s01', 'gsm8k', 1, ('P', 'L', 'T')),
    ('s02', 'math500', 1, ('T', 'L', 'P')),
    ('s03', 'humaneval', 1, ('P', 'T', 'L')),
    ('s04', 'gsm8k', 2, ('T', 'L', 'P')),
    ('s05', 'math500', 2, ('P', 'T', 'L')),
    ('s06', 'humaneval', 2, ('L', 'P', 'T')),
)
# compute-DL: DSv4.1's two turns (slots 1-10 comparator first, slots 11-20 lane first), P in the comparator's
# place, T last in the first turn and first in the second.
CDL_LAYOUT = (
    ('c01', 0, ('P', 'L', 'T')),
    ('c02', 10, ('T', 'L', 'P')),
)

TRUNCATION_MARKER = '[output truncated at 512 tokens]'
RUNNER_FORMS = ('fence', 'repl', 'inline', 'trace')


def canonical(value):
    """The kit's canonical JSON (sorted keys, compact, UTF-8, no NaN)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, text, mode=0o644):
    """Create a file that must not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(text)


def write_json_new(path, value, mode=0o644):
    write_new(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', mode)


def write_jsonl_new(path, rows, mode=0o644):
    write_new(path, ''.join(canonical(r) + '\n' for r in rows), mode)


def append_jsonl(path, row):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(canonical(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def kit_path():
    """Put the unchanged phase2 modules and kit/bench.py on sys.path, as their own imports expect."""
    for path in (str(KIT), str(PHASE2)):
        if path not in sys.path:
            sys.path.insert(0, path)


def kit(name):
    """Import an unchanged phase2/kit module by its own name (analysis, grade_all, jail, client, bench...)."""
    kit_path()
    return __import__(name)


def cdl(name):
    """Load an unchanged compute-dl module under a distinct name (its analysis.py would clash with phase2's)."""
    key = 'cdl_' + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, CDL / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def body(prompt, seed, arm, pass_name, model):
    """The exact request body: same for every arm except the declared tool."""
    require(arm in ARMS and pass_name in PASSES and type(seed) is int, 'bad arm/pass/seed')
    p = PASSES[pass_name]
    value = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'seed': seed,
             'max_tokens': p['max_tokens'], 'deadline_s': p['deadline_s']}
    value.update(copy.deepcopy(SETTINGS))
    if arm != 'P':
        value['tools'] = copy.deepcopy(TOOLS)
    check_body(value)
    return value


def check_body(value):
    """Refuse any field the server would not read (it drops unknown fields without an error)."""
    require(set(value) <= ACCEPTED_FIELDS, 'unsupported request fields: ' + repr(sorted(set(value) - ACCEPTED_FIELDS)))
    require(set(value.get('chat_template_kwargs', {})) <= ACCEPTED_KWARGS, 'unsupported chat_template_kwargs')
    require(set(value.get('stream_options', {})) <= ACCEPTED_STREAM_OPTIONS, 'unsupported stream_options')
    require(value.get('stream') is True and value.get('return_token_ids') is True, 'stream + native IDs required')
    require(value.get('tools') in (None, TOOLS), 'only the built-in code_interpreter tool may be declared')
    return value


def boot_name(pass_name, turn, arm):
    return f'{pass_name}-{turn}-{arm}'


def runner_owned(tf):
    """Runner cells the engine wrote a value for (runtime.runner.<form>.owned), 0 when the runner is off."""
    runner = (tf or {}).get('runner') or {}
    total = 0
    for form in RUNNER_FORMS:
        item = runner.get(form)
        if isinstance(item, dict) and type(item.get('owned')) is int:
            total += item['owned']
    return total


def compute_calls(tf):
    compute = (tf or {}).get('compute')
    calls = compute.get('calls') if isinstance(compute, dict) else None
    return calls if isinstance(calls, list) else []


def successful_call(call):
    """A delivered successful lane call: inserted, status ok and exit 0. A call the in-model runner ran (S3, arm T)
    has no process and reports exit_code null; it counts as successful with status ok."""
    if call.get('status') != 'ok' or call.get('inserted') is not True:
        return False
    if call.get('executor') == 'in-model runner':
        return call.get('exit_code') in (None, 0)
    return call.get('exit_code') == 0
