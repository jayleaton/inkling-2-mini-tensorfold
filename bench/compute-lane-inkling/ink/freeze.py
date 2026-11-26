#!/usr/bin/env python3
"""Freeze the cohort, request bodies, boot order and every input hash before the first request.

  create    --recipe RECIPE --out NEW_DIR [--model Inkling-2-Mini]
  bind-host --freeze DIR --host NAME [--served-name NAME]   (docker image inspect only; per Spark)
  verify    --freeze DIR [--recipe RECIPE]                   (inventory + drift; prints the root hash)

create is deterministic (no timestamps, no paths in hashed content), so two Sparks that run it on the same tree get
the same root hash. Cohort selection is the kit's own bench.py normalize (SHA256("stdbench-v1:"+id) order, first
48/20/24 per suite), blocks are contiguous halves, each block's sorted (taskID, seed) pairs get a fresh
Random(20261104).shuffle; compute-DL turns are DSv4.1's runtime_stage.ordered_pairs (slots 1-10 / 11-20, a fresh
Random(20261105).shuffle per family, families concatenated in the fixed order).
"""
import argparse
import gzip
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import math_packet  # noqa: E402

bench = common.kit('bench')
RECIPE_FILES = ('start.sh', 'stop.sh', 'scripts/config.sh', 'scripts/prepare.sh', 'scripts/banner.sh',
                'scripts/fetch.sh', 'scripts/runner_assets.py')
SKIP_PARTS = {'__pycache__', 'runs', 'selftest-out'}


def copied_files():
    """dsv41/COPIED.sha256: '<sha256>  <path under dsv41>  <source path in the DSv4.1 export>'."""
    rows = {}
    for line in (common.DSV / 'COPIED.sha256').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            sha, rel, source = line.split(None, 2)
            rows[rel] = (sha, source)
    for rel, (sha, _) in rows.items():
        common.require(common.sha_file(common.DSV / rel) == sha, 'copied kit file changed: dsv41/' + rel)
    return rows


def harness_files():
    """Every file of bench/compute-lane-inkling except caches and run outputs, by relative path."""
    out = {}
    for path in sorted(common.ROOT.rglob('*')):
        rel = path.relative_to(common.ROOT)
        if path.is_file() and not SKIP_PARTS & set(rel.parts) and path.suffix != '.pyc':
            out[str(rel)] = common.sha_file(path)
    return out


def recipe_state(recipe):
    recipe = Path(recipe).resolve()
    common.require((recipe / 'start.sh').is_file(), 'recipe tree not found: ' + str(recipe))
    common.require(not (recipe / '.env').exists(), 'recipe has a .env; the arms must run on the recipe defaults')
    files = {rel: common.sha_file(recipe / rel) for rel in RECIPE_FILES if (recipe / rel).exists()}
    for sub in ('patches', 'config'):
        for path in sorted((recipe / sub).glob('*')):
            if path.is_file():
                files[str(path.relative_to(recipe))] = common.sha_file(path)
    env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/tmp'), 'RUNNER': '1'}
    expected = subprocess.run(['bash', '-c', 'source ./scripts/config.sh && image_hash'], cwd=recipe, env=env,
                              capture_output=True, text=True, timeout=120, check=True).stdout.strip()
    common.require(len(expected) == 12 and all(c in '0123456789abcdef' for c in expected), 'bad image_hash')
    return {'files_sha256': files, 'tf_patches_expected': expected, 'image_tag': common.IMAGE_TAG}


def normalized(suite, limit):
    with tempfile.TemporaryDirectory(prefix='ink-normalize-') as tmp:
        out = Path(tmp) / 'tasks.jsonl'
        bench.normalize(argparse.Namespace(command='normalize', lock=str(common.KIT / 'datasets.lock.json'),
                                           suites=suite, fetch_network=False, out=str(out), limit=limit))
        return bench.read_jsonl(out)


def check_humaneval(tasks):
    with gzip.open(common.KIT / 'sources' / 'humaneval.jsonl.gz', 'rt', encoding='utf-8') as stream:
        originals = {r['task_id']: r for r in (json.loads(x) for x in stream if x.strip())}
    for task in tasks:
        original = originals[task['id']]
        common.require(task['test'] == original['test'] and task['entry_point'] == original['entry_point']
                       and task['code_prefix'] == original['prompt'], 'normalized HumanEval differs from original')


def cdl_tasks():
    tasks = common.read_jsonl(common.CDL / 'dataset-v1' / 'tasks-public.jsonl')
    answers = common.read_json(common.CDL / 'dataset-v1' / 'answers-a.json')
    common.require(len(tasks) == 120 and {t['id'] for t in tasks} == set(answers), 'compute-DL task/answer coverage')
    grid = {(t['family'], t['slot']) for t in tasks}
    common.require(grid == {(f, s) for f in common.CDL_FAMILIES for s in range(20)}, 'compute-DL family/slot grid')
    for t in tasks:
        common.require(t['suite'] == 'compute_dl' and t['id'] == f"compute-dl-v1-{t['family']}-{t['slot'] + 1:03d}"
                       and 'answer' not in t, 'compute-DL public task schema')
    return tasks


def cdl_order(tasks, low):
    """runtime_stage.ordered_pairs, verbatim in effect."""
    result = []
    for family in common.CDL_FAMILIES:
        units = [(t, seed) for t in tasks if t['family'] == family and low <= t['slot'] < low + 10
                 for seed in common.SEEDS]
        units.sort(key=lambda pair: (pair[0]['id'], pair[1]))
        random.Random(20261105).shuffle(units)
        result.extend(units)
    common.require(len(result) == 120, 'compute-DL turn must have 120 task/seed pairs')
    return result


def request_row(task, seed, arm, pass_name, model):
    b = common.body(task['prompt'], seed, arm, pass_name, model)
    return {'suite': task['suite'], 'id': task['id'], 'seed': seed, 'task': task, 'task_sha256': bench.digest(task),
            'body_template': b, 'body_sha256': common.digest(b)}


def diagnostics(arm, pass_name, model):
    rows = []
    for diag_id, prompt, reference in common.DIAGNOSTICS:
        task = {'suite': 'diagnostic', 'id': diag_id, 'prompt': prompt, 'kind': 'numeric', 'answer': reference}
        rows.append(dict(request_row(task, 101, arm, pass_name, model), reference=reference, included_in_cohort=False))
    return rows


def create(args):
    out = Path(args.out).absolute()
    common.require(not out.exists() and not out.is_symlink(), 'output exists: ' + str(out))
    common.require(common.ROOT not in out.parents, 'write the freeze outside bench/compute-lane-inkling')
    copied = copied_files()
    recipe = recipe_state(args.recipe)
    selected = {suite: normalized(suite, common.STD_COUNTS[suite]) for suite in common.STD_SUITES}
    for suite, tasks in selected.items():
        common.require(len(tasks) == common.STD_COUNTS[suite], 'selected count differs: ' + suite)
    check_humaneval(selected['humaneval'])
    compute = cdl_tasks()
    model = args.model
    tmp = Path(tempfile.mkdtemp(prefix='.ink-freeze-', dir=str(out.parent)))
    try:
        emit = lambda rel, value: common.write_json_new(tmp / rel, value)   # noqa: E731
        blocks, orders = {}, {}
        for suite in common.STD_SUITES:
            tasks = selected[suite]
            common.write_jsonl_new(tmp / 'stdbench' / f'{suite}-selected.jsonl', tasks)
            cut = (len(tasks) + 1) // 2
            for block, block_tasks in enumerate((tasks[:cut], tasks[cut:]), 1):
                pairs = sorted((t['id'], seed) for t in block_tasks for seed in common.SEEDS)
                random.Random(20261104).shuffle(pairs)
                blocks[(suite, block)] = {t['id']: t for t in block_tasks}
                orders[(suite, block)] = pairs
                common.write_jsonl_new(tmp / 'stdbench' / f'{suite}-block{block}-ordered-pairs.jsonl',
                                       [{'suite': suite, 'id': i, 'seed': s} for i, s in pairs])
        turns, boots, expected = [], [], []
        for pass_name in common.PASSES:
            layout = [(turn, 'stdbench', suite, block, order) for turn, suite, block, order in common.STD_LAYOUT]
            layout += [(turn, 'compute_dl', 'compute_dl', low, order) for turn, low, order in common.CDL_LAYOUT]
            for turn, study, suite, part, order in layout:
                if study == 'stdbench':
                    by_id = blocks[(suite, part)]
                    units = [(by_id[i], s) for i, s in orders[(suite, part)]]
                else:
                    units = cdl_order(compute, part)
                names = []
                for arm in order:
                    boot = common.boot_name(pass_name, turn, arm)
                    rows = [request_row(t, s, arm, pass_name, model) for t, s in units]
                    value = {'boot': boot, 'pass': pass_name, 'turn': turn, 'study': study, 'suite': suite,
                             'arm': arm, 'pass_settings': common.PASSES[pass_name],
                             'env': dict(common.COMMON_ENV, **common.ARM_ENV[arm]),
                             'diagnostics': diagnostics(arm, pass_name, model), 'requests': rows}
                    rel = f'boots/{boot}.json'
                    emit(rel, value)
                    boots.append({'boot': boot, 'pass': pass_name, 'turn': turn, 'arm': arm, 'file': rel,
                                  'sha256': common.sha_file(tmp / rel), 'requests': len(rows)})
                    expected += [{'pass': pass_name, 'study': study, 'suite': r['suite'], 'id': r['id'],
                                  'seed': r['seed'], 'arm': arm, 'boot': boot} for r in rows]
                    names.append(boot)
                turns.append({'pass': pass_name, 'turn': turn, 'study': study, 'suite': suite,
                              'block' if study == 'stdbench' else 'slot_from': part, 'arm_order': list(order),
                              'boots': names})
        keys = [(e['pass'], e['suite'], e['id'], e['seed'], e['arm']) for e in expected]
        common.require(len(keys) == len(set(keys)), 'duplicate scheduled reply')
        per_pass = len(keys) // len(common.PASSES)
        common.require(per_pass == (92 + 120) * 2 * 3, 'grid size differs: ' + str(per_pass))
        common.write_jsonl_new(tmp / 'expected.jsonl', expected)
        all_std = [t for suite in common.STD_SUITES for t in selected[suite]]
        emit('analysis-stdbench.json', {'tasks': all_std, 'tasks_sha256': bench.digest(all_std),
                                        'seeds': list(common.SEEDS)})
        emit('analysis-compute-dl.json', {'tasks': compute, 'tasks_sha256': bench.digest(compute),
                                          'seeds': list(common.SEEDS)})
        preseal = math_packet.build_preseal(all_std)
        (tmp / 'private').mkdir(mode=0o700)
        math_packet.math_review.write_private(tmp / 'private' / 'math-preseal.json', preseal)
        manifest = {
            'version': 1, 'study': 'compute-lane-inkling: DSv4.1 stdbench + compute-DL rerun on Inkling 2 Mini',
            'specs_sha256': {rel: common.sha_file(common.ROOT / rel) for rel in
                             ('PROTOCOL.md', 'dsv41/spec/T5-STDBENCH-SPEC.md',
                              'dsv41/compute-dl/spec/T5-COMPUTE-DL-SPEC.md',
                              'dsv41/compute-dl/spec/T5-COMPUTE-DL-SPEC-A1.md',
                              'dsv41/compute-dl/spec/T5-COMPUTE-DL-SPEC-A2.md')},
            'harness_sha256': harness_files(), 'dsv41_copied': {k: v[0] for k, v in sorted(copied.items())},
            'dataset_lock_sha256': common.sha_file(common.KIT / 'datasets.lock.json'),
            'datasets_sha256': {p.name: common.sha_file(p) for p in sorted((common.KIT / 'sources').glob('*'))},
            'compute_dl_public_sha256': common.sha_file(common.CDL / 'dataset-v1' / 'tasks-public.jsonl'),
            'compute_dl_answers_sha256': common.sha_file(common.CDL / 'dataset-v1' / 'answers-a.json'),
            'recipe': recipe, 'model': model, 'arms': list(common.ARMS), 'seeds': list(common.SEEDS),
            'passes': common.PASSES, 'settings': common.SETTINGS, 'tools': common.TOOLS,
            'accepted_fields': sorted(common.ACCEPTED_FIELDS), 'arm_env': common.ARM_ENV,
            'common_env': common.COMMON_ENV, 'diagnostics': [list(d) for d in common.DIAGNOSTICS],
            'stdbench_counts': common.STD_COUNTS, 'turns': turns, 'boots': boots,
            'scheduled_replies': {'per_pass': per_pass, 'total': len(keys)},
            'math_preseal': {'file': 'private/math-preseal.json', 'sha256': common.digest(preseal),
                             'candidates': len(preseal['order'])},
            'primary': {'stdbench': 'L-P at c8k, per suite, Holm over the three suites',
                        'compute_dl': 'L-P at c8k, overall task sign-flip'},
            'host_slots': {'fill_with': 'freeze.py bind-host', 'image_id': None, 'served_name': model,
                           'tf_patches_expected': recipe['tf_patches_expected']},
        }
        emit('manifest.json', manifest)
        inventory = {str(p.relative_to(tmp)): common.sha_file(p) for p in sorted(tmp.rglob('*')) if p.is_file()}
        emit('file-hashes.json', inventory)
        root = common.sha_file(tmp / 'file-hashes.json')
        common.write_new(tmp / 'file-hashes.sha256', root + '  file-hashes.json\n')
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(common.canonical({'freeze': str(out), 'root_sha256': root, 'manifest_sha256':
                            common.sha_file(out / 'manifest.json'), 'boots': len(boots), 'scheduled': len(keys),
                            'tf_patches_expected': recipe['tf_patches_expected']}))


def verify_freeze(freeze, recipe=None, *, quiet=False):
    """Inventory, harness drift and (optionally) recipe drift; returns the manifest."""
    freeze = Path(freeze)
    root = (freeze / 'file-hashes.sha256').read_text().split()[0]
    common.require(common.sha_file(freeze / 'file-hashes.json') == root, 'freeze inventory hash differs')
    for rel, sha in common.read_json(freeze / 'file-hashes.json').items():
        common.require(common.sha_file(freeze / rel) == sha, 'frozen file changed: ' + rel)
    manifest = common.read_json(freeze / 'manifest.json')
    now = harness_files()
    for rel, sha in manifest['harness_sha256'].items():
        common.require(now.get(rel) == sha, 'harness file changed since freeze: ' + rel)
    copied_files()
    if recipe is not None:
        state = recipe_state(recipe)
        common.require(state == manifest['recipe'], 'recipe tree differs from the freeze')
    if not quiet:
        print(common.canonical({'root_sha256': root, 'manifest_sha256': common.sha_file(freeze / 'manifest.json'),
                                'verified': True}))
    return manifest


def bind_host(args):
    manifest = verify_freeze(args.freeze, quiet=True)
    out = Path(args.freeze) / 'hosts' / (args.host + '.json')
    common.require(not out.exists(), 'host already bound: ' + str(out))
    text = subprocess.run(['docker', 'image', 'inspect', '-f', '{{.Id}} {{index .Config.Labels "tf.patches"}}',
                           common.IMAGE_TAG], capture_output=True, text=True, timeout=60, check=True).stdout.split()
    common.require(len(text) == 2, 'unexpected docker image inspect output')
    image_id, patches = text
    common.require(patches == manifest['recipe']['tf_patches_expected'],
                   f'image tf.patches {patches} differs from the frozen recipe hash '
                   f'{manifest["recipe"]["tf_patches_expected"]}: rebuild the runner image from this tree')
    served = args.served_name or manifest['model']
    common.require(served == manifest['model'], 'the frozen bodies name the model ' + manifest['model'])
    value = {'host': args.host, 'hostname': socket.gethostname(), 'image_tag': common.IMAGE_TAG,
             'image_id': image_id, 'tf_patches': patches, 'served_name': served,
             'freeze_root_sha256': (Path(args.freeze) / 'file-hashes.sha256').read_text().split()[0]}
    common.write_json_new(out, value)
    print(common.canonical(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    p.add_argument('--recipe', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', default=common.DEFAULT_MODEL)
    p = sub.add_parser('bind-host')
    p.add_argument('--freeze', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--served-name')
    p = sub.add_parser('verify')
    p.add_argument('--freeze', required=True)
    p.add_argument('--recipe')
    args = parser.parse_args()
    if args.command == 'create':
        create(args)
    elif args.command == 'bind-host':
        bind_host(args)
    else:
        verify_freeze(args.freeze, args.recipe)


if __name__ == '__main__':
    main()
