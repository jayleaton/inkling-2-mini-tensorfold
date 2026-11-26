"""Streaming client for one frozen request against the recipe's server; the kit's parse/accounting unchanged.

Transport differences from the kit's phase2/client.py Transport (whose 8192/270 asserts and DSv4.1 endpoint pin
do not fit): the cap/deadline come from the frozen pass, the raw SSE bytes are kept, the client records its own
monotonic time to the first streamed token, and an in-stream {"error": ...} event is a protocol error.
Parsing (parse_stream) and native accounting (accounting: token_ids/token_sha256/usage/inserted reconciliation,
aggregate cap) are the kit's own functions, imported unchanged.
"""
import gzip
import json
import signal
import socket
import time
import urllib.error
import urllib.request

import common

kclient = common.kit('client')


class Result(dict):
    pass


def first_token_frame(frame):
    for choice in frame.get('choices') or []:
        delta = choice.get('delta') or {}
        for key in ('content', 'reasoning_content', 'reasoning'):
            if isinstance(delta.get(key), str) and delta[key]:
                return True
        if delta.get('tool_calls'):
            return True
    return False


def send(base_url, body, *, cap, wait_s, raw_path=None):
    """POST one canonical body; return the row (status ok/error/timeout) with kit parsing and accounting."""
    common.check_body(body)
    common.require(body['max_tokens'] == cap, 'body cap differs from the pass cap')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError('unexpected active alarm')
    wire = common.canonical(body).encode('utf-8')
    lines, stamps = [], []
    row = {'status': 'error', 'raw_response': None, 'sse_done': False, 'content': '', 'reasoning': '',
           'usage': None, 'tensorfold': {}, 'finish_reason': None, 'http_status': None, 'error': None,
           'client_ttft_seconds': None}

    def expired(_sig, _frame):
        raise TimeoutError('whole-request client wait exceeded')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, wait_s)
    start = time.monotonic()
    try:
        request = urllib.request.Request(base_url + '/v1/chat/completions', data=wire, method='POST',
                                         headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        with urllib.request.urlopen(request, timeout=wait_s) as response:
            row['http_status'] = response.status
            for raw in response:
                now = time.monotonic()
                lines.append(raw)
                stamps.append(now - start)
                text = raw.decode('utf-8', 'replace').strip()
                if row['client_ttft_seconds'] is None and text.startswith('data:') and text[5:].strip() != '[DONE]':
                    try:
                        if first_token_frame(json.loads(text[5:].strip())):
                            row['client_ttft_seconds'] = now - start
                    except ValueError:
                        pass
                if text == 'data: [DONE]':
                    break
        row['status'] = 'ok'
    except urllib.error.HTTPError as error:
        row['http_status'] = error.code
        try:
            row['error_body'] = error.read().decode('utf-8', 'replace')[:65536]
        except (OSError, TimeoutError):
            pass
        row['error'] = 'HTTPError: ' + str(error)
    except (TimeoutError, socket.timeout) as error:
        row['status'] = 'timeout'
        row['error'] = type(error).__name__ + ': ' + str(error)
    except (OSError, ValueError) as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        row['elapsed_seconds'] = time.monotonic() - start
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    blob = b''.join(lines)
    row['raw_stream_sha256'] = common.sha_bytes(blob)
    row['raw_stream_bytes'] = len(blob)
    if raw_path is not None:
        with gzip.open(raw_path, 'xb') as stream:
            stream.write(blob)
    frames = []
    try:
        parsed = kclient.parse_stream(lines, start, frames)
    except ValueError as error:   # malformed JSON frame: a recorded protocol failure
        parsed = None
        row['status'] = 'error' if row['status'] == 'ok' else row['status']
        row['error'] = (row['error'] or '') + '; malformed SSE frame: ' + str(error)
    if parsed is not None:
        parsed.pop('sse_frames', None)   # the raw stream file keeps every frame byte
        row.update(parsed)
        if row['status'] == 'ok' and not row['sse_done']:
            row.update(status='error', error='stream ended without [DONE]')
        errors = [f['error'] for f in frames if isinstance(f, dict) and 'error' in f]
        if errors:
            row.update(status='error', error='server error event: ' + common.canonical(errors[-1])[:4096])
        if row.get('external_tool_calls'):
            row.update(status='error', error='unexpected external client tool calls')
        row['response_id'] = next((f.get('id') for f in frames if isinstance(f, dict) and f.get('id')), None)
        row['frames'] = len(frames)
    row['frame_offsets_seconds_last'] = stamps[-1] if stamps else None
    row['native_accounting'] = kclient.accounting(row, cap)
    usage = row.get('usage') or {}
    completion = usage.get('completion_tokens')
    ttft = row['client_ttft_seconds']
    span = row['elapsed_seconds'] - ttft if ttft is not None else None
    row['client_decode_tps'] = (completion / span if type(completion) is int and span and span > 0 else None)
    return row


def get_json(base_url, path, timeout=10):
    with urllib.request.urlopen(base_url + path, timeout=timeout) as response:
        return json.load(response)


def post_json(base_url, path, value, timeout=60):
    request = urllib.request.Request(base_url + path, data=common.canonical(value).encode('utf-8'), method='POST',
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
