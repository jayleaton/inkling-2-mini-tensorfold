"""Per-reply safety: arm gates, independent CPython references for delivered calls, native-ID binding.

The reference executor is the kit's unchanged jail (phase2/jail.py + jail_worker.py) in the recipe's runner image;
InkJail only replaces the DSv4.1 image tag/revision-label check with this image's tag and tf.patches label.
jail_worker.reference() is the same REPL convention as the Inkling lane's prelude (core/compute_runner.zig runner:
random.seed(0), `<code>`, final bare expression printed as repr) and the same body normalization (runner.body).
"""
import base64
import json
import re

import common
import ink_client

jail = common.kit('jail')


class InkJail(jail.Jail):
    """The kit Jail, pinned to the recipe image by full ID and its tf.patches label instead of the DSv4.1 tag."""

    def __init__(self, image_id, tag=common.IMAGE_TAG, patches=None, docker_prefix=('docker',), job_timeout=7200):
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise ValueError('full requested image ID required')
        if not isinstance(docker_prefix, tuple) or not docker_prefix or docker_prefix[-1] != 'docker':
            raise ValueError('docker_prefix must be a tuple ending with docker')
        if type(job_timeout) is not int or job_timeout < 7200:
            raise ValueError('whole Docker job timeout must be >=7200 seconds')
        self.image_id, self.docker_prefix, self.job_timeout = image_id, docker_prefix, job_timeout
        self.cpu_job = None
        record = json.loads(self._docker(['image', 'inspect', tag]).stdout)[0]
        if record['Id'] != image_id:
            raise ValueError('requested image ID differs from local pinned tag')
        label = (record.get('Config', {}).get('Labels') or {}).get('tf.patches')
        if patches is not None and label != patches:
            raise ValueError('image tf.patches label differs from the frozen recipe hash')
        self.image_record = dict(tag=tag, id=image_id, tf_patches=label)


def clipped_parts(output):
    """(prefix, True) when the delivered body carries the lane's 512-token cut marker (compute_segments capTokens)."""
    if output == common.TRUNCATION_MARKER:
        return '', True
    tail = '\n' + common.TRUNCATION_MARKER
    if output.endswith(tail):
        return output[:-len(tail)], True
    return output, False


def compare(call, reference, count_tokens=None):
    """Classify one delivered call against its independent reference run.

    match           : native ok, reference ok, bytes equal
    clipped_match   : native body is the reference body cut at the 512-token marker (prefix bytes equal; with
                      count_tokens, the reference exceeds 512 server tokens and the prefix fits)
    mismatch        : native ok and reference ok but the delivered bytes differ (a wrong delivered value: stop)
    error_match / error_differs : native non-ok (an attributed execution error; never a value)
    inconclusive    : the reference did not complete (jail's 3 s / 512 MiB / no site-packages limits are tighter
                      than the lane's), so the delivered value is unchecked
    """
    try:
        body = base64.b64decode(reference['normalized_body_base64'], validate=True).decode('utf-8')
    except (KeyError, TypeError, ValueError):
        return {'verdict': 'inconclusive', 'reason': 'reference produced no body'}
    output = call.get('output') if isinstance(call.get('output'), str) else ''
    native, ref = call.get('status'), reference.get('status')
    if native != 'ok':
        return {'verdict': 'error_match' if (ref == native and body == output) else 'error_differs',
                'native_status': native, 'reference_status': ref}
    if ref != 'ok':
        return {'verdict': 'inconclusive', 'native_status': native, 'reference_status': ref,
                'reason': 'reference did not complete under the jail limits'}
    if output == body:
        return {'verdict': 'match'}
    prefix, clipped = clipped_parts(output)
    if clipped and body.startswith(prefix) and body != prefix:
        detail = {'verdict': 'clipped_match', 'delivered_prefix_bytes': len(prefix.encode()),
                  'reference_bytes': len(body.encode())}
        if count_tokens is not None:
            try:
                full, kept = count_tokens(body), count_tokens(prefix) if prefix else 0
                detail.update(reference_tokens=full, prefix_tokens=kept,
                              token_rule_holds=bool(full > 512 and kept <= 512))
            except (OSError, ValueError, KeyError) as error:
                detail.update(token_rule_holds=None, token_error=str(error))
        return detail
    return {'verdict': 'mismatch', 'delivered_sha256': common.sha_bytes(output.encode()),
            'reference_sha256': common.sha_bytes(body.encode())}


def server_binding(row, detokenize):
    """Bind each inserted call's source and result to the reply's own token IDs, decoded by the server.

    This is the engine's decoder (POST /detokenize, special tokens kept), not an independent CPU tokenizer, which
    this harness does not have for Inkling; it is reported as server-decoder binding."""
    tf = row.get('tensorfold') or {}
    ids = tf.get('token_ids')
    calls = [c for c in common.compute_calls(tf) if c.get('inserted') is True]
    if not calls:
        return {'calls': [], 'qualified': True, 'method': 'no inserted calls'}
    if not isinstance(ids, list):
        return {'calls': [], 'qualified': False, 'method': 'server /detokenize', 'reason': 'no native token IDs'}
    try:
        text = detokenize(ids)
    except (OSError, ValueError, KeyError) as error:
        return {'calls': [], 'qualified': False, 'method': 'server /detokenize', 'reason': str(error)}
    bound = []
    for index, call in enumerate(calls):
        code, output = call.get('code') or '', call.get('output') or ''
        escaped = json.dumps(code, ensure_ascii=False)[1:-1]
        bound.append({'index': index, 'id': call.get('id'),
                      'source_sha256': common.sha_bytes(code.encode()),
                      'output_sha256': common.sha_bytes(output.encode()),
                      'source_in_ids': code in text or escaped in text,
                      'output_in_ids': output in text})
    return {'calls': bound, 'method': 'server /detokenize of tensorfold.token_ids',
            'decoded_sha256': common.sha_bytes(text.encode()),
            'qualified': all(b['source_in_ids'] and b['output_in_ids'] for b in bound)}


def arm_violations(arm, row):
    """Effective-OFF assertions per arm (raise in the controller after the row is durable)."""
    tf = row.get('tensorfold') or {}
    calls, cells = common.compute_calls(tf), common.runner_owned(tf)
    problems = []
    if arm == 'P':
        if calls or (tf.get('compute') or {}).get('inserted_tokens'):
            problems.append('P made lane calls')
        if cells:
            problems.append('P wrote runner cells')
    if arm in ('P', 'L'):
        if cells:
            problems.append(arm + ' wrote runner cells')
        if tf.get('runner') not in (None, {}):
            problems.append(arm + ' reply carries a runner receipt')
    if tf.get('runner_guide') is not None:
        problems.append('reply ran with a runner guide')
    if 'compute_executed' in tf:
        problems.append('compute default is on (runtime.compute_executed present)')
    return sorted(set(problems))


def check_row(arm, row, references, detokenize=None, count_tokens=None):
    """References for every inserted call, server binding, arm gates. Returns the safety record."""
    tf = row.get('tensorfold') or {}
    refs = []
    for index, call in enumerate(common.compute_calls(tf)):
        if call.get('inserted') is not True:
            continue
        result = references(call.get('code') or '')
        refs.append({'index': index, 'id': call.get('id'), 'executor': call.get('executor'),
                     'status': call.get('status'), 'exit_code': call.get('exit_code'),
                     'result': result, **compare(call, result, count_tokens)})
    binding = (server_binding(row, detokenize) if detokenize is not None
               else {'calls': [], 'qualified': None, 'method': 'not run'})
    violations = arm_violations(arm, row)
    mismatch = [r['index'] for r in refs if r['verdict'] == 'mismatch']
    unchecked = [r['index'] for r in refs if r['verdict'] in ('inconclusive', 'error_differs')]
    clip_unproven = [r['index'] for r in refs if r['verdict'] == 'clipped_match' and r.get('token_rule_holds') is False]
    accounting = row.get('native_accounting') or {}
    qualified = bool(not violations and not mismatch and not unchecked and not clip_unproven
                     and binding.get('qualified') is True
                     and (row.get('status') != 'ok' or accounting.get('complete') is True))
    return {'references': refs, 'binding': binding, 'violations': violations, 'mismatches': mismatch,
            'unchecked': unchecked, 'clip_token_rule_failed': clip_unproven,
            'runner_cells': common.runner_owned(tf), 'qualified': qualified,
            'stop': bool(violations or mismatch)}


def server_tools(base_url):
    """The live server's /detokenize and /tokenize (no generation), for binding and the clipping count."""
    def detokenize(ids):
        return ink_client.post_json(base_url, '/detokenize', {'tokens': ids})['prompt']

    def count_tokens(text):
        return ink_client.post_json(base_url, '/tokenize', {'prompt': text, 'add_special_tokens': False})['count']
    return detokenize, count_tokens
