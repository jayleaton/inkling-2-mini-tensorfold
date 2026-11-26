#!/usr/bin/env python3
"""Per-arm/per-suite results, the registered paired analyses (L-P, T-P) and the DSv4.1 comparison table.

  report.py --freeze F --runs R --graded GRADED_DIR [--math-reviewed math-reviewed.jsonl] --out NEW_DIR

Paired statistics are the unchanged kit analyses, fed relabeled rows: stdbench -> phase2/analysis.py analyze()
(treatment as 'lane', P as 'plain': 20,000 task-cluster draws RNG 20261104, exact sign flip, Holm over the three
suites, MATH adverse/favourable bounds); compute-DL -> compute-dl/analysis.py analyze() (treatment as 'L', P as 'D':
20,000 family-stratified draws RNG 20261105, overall exact sign flip). Without --math-reviewed every MATH reply is
unresolved and the MATH bounds span everything.
"""
import argparse
from collections import Counter
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

p2analysis = common.kit('analysis')

# DSv4.1 published numbers on the same sets (dsv41/published/*, byte-identical copies; COPIED.sha256).
DSV41 = {
    'source_stdbench': 'dsv41/published/decl-control-RESULTS.md (declaration-control A1, P/D/L; L-P is its '
                       'secondary contrast; all three lanes DECLARED BUT UNUSED: zero dataset calls)',
    'source_compute_dl': 'dsv41/published/compute-dl-a2-RESULTS.md (A2 completion; L-P amended primary)',
    'stdbench': {
        'gsm8k': {'P': '54/96', 'L': '52/96', 'L_minus_P_pp': -2.0833, 'ci95_pp': [-8.3333, 4.1667], 'p': 0.7539,
                  'returned_ids_ratio_L_over_P': 0.7630, 'wall_ratio_L_over_P': 0.7526, 'dataset_calls_L': 0},
        'humaneval': {'P': '47/48', 'L': '48/48', 'L_minus_P_pp': 2.0833, 'ci95_pp': [0.0, 6.25], 'p': 1.0,
                      'returned_ids_ratio_L_over_P': 0.7265, 'wall_ratio_L_over_P': 0.6949, 'dataset_calls_L': 0},
        'math500': {'P': '40/40', 'L': '40/40', 'L_minus_P_pp': 0.0, 'ci95_pp': [0.0, 0.0], 'p': 1.0,
                    'returned_ids_ratio_L_over_P': 0.7808, 'wall_ratio_L_over_P': 0.7683, 'dataset_calls_L': 0},
    },
    'compute_dl': {'P': '120/240', 'L': '187/240', 'L_minus_P_pp': 27.9167, 'ci95_pp': [22.9167, 32.9167],
                   'p': 6.382590167485932e-07, 'returned_ids_ratio_L_over_P': 0.0724, 'wall_ratio_L_over_P': 0.0732,
                   'P_per_family': {'integer': '0/40', 'modular': '0/40', 'division': '0/40', 'dp': '40/40',
                                    'graph': '40/40', 'aggregation': '40/40'},
                   'L_per_family': {'integer': '28/40', 'modular': '21/40', 'division': '40/40', 'dp': '31/40',
                                    'graph': '28/40', 'aggregation': '39/40'}},
    'settings': 'DeepSeek-V4.1-Flash-1017 EXL3 2.80bpw, image r-3a429d6f122d, two Sparks TP2; thinking/high, T1, '
                'top_p .95, top_k 20, 8192 returned-ID cap, 270 s server deadline, client wait 360 s',
}

CAVEATS = [
    'Different model: Inkling 2 Mini (26.4B total / 3.4B active, looped middle layers) vs DeepSeek V4.1 Flash; '
    'the comparison is of each model\'s own lane gain on the same task sets, not of the two models\' accuracy.',
    'Different engine image and serving stack: tensorfold-inkling2:c4-dc2ca8a-runner on one Spark (fp8 KV, '
    'DFlash2 drafts, disk states) vs r-3a429d6f122d in TP2 on two Sparks.',
    'DSv4.1 compute-DL L declared a function named python; here L and T declare code_interpreter, which the server '
    'expands to the identical python function (compute_turns.zig definition, same description text).',
    'DSv4.1 compute-DL P was collected after L in one later boot (an outcome-informed amendment); here P, L and T '
    'run interleaved in pre-registered boots.',
    'DSv4.1 stdbench numbers are from the declaration-control study: its L made zero dataset calls (DECLARED BUT '
    'UNUSED), so its L-P is a declaration contrast, not an execution effect.',
    'The 32,768-token pass has no DSv4.1 counterpart.',
]


def keyed(rows):
    return {(r['pass'], r['suite'], r['id'], r['seed'], r['arm']): r for r in rows}


def safety_rows(runs):
    out, diags = {}, []
    for path in sorted(Path(runs).glob('*/*/*/safety.jsonl')):
        for row in common.read_jsonl(path):
            if row.get('diagnostic'):
                diags.append(row)
            else:
                out[(row['pass'], row['suite'], row['id'], row['seed'], row['arm'])] = row
    return out, diags


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def arm_summary(rows, safety, tasks):
    n = len(rows)
    calls = [c for r in rows for c in common.compute_calls(r.get('tensorfold'))]
    ok = [c for c in calls if common.successful_call(c)]
    usage = [(r.get('usage') or {}).get('completion_tokens') for r in rows]
    usage = [u for u in usage if type(u) is int]
    walls = [r['elapsed_seconds'] for r in rows]
    spans = [(r['elapsed_seconds'] - r['client_ttft_seconds']) for r in rows
             if r.get('client_ttft_seconds') is not None and type((r.get('usage') or {}).get('completion_tokens')) is int]
    decoded = sum((r.get('usage') or {}).get('completion_tokens') for r in rows
                  if r.get('client_ttft_seconds') is not None and type((r.get('usage') or {}).get('completion_tokens')) is int)
    correct = sum(r['status'] == 'ok' and r['grade'] == 'correct' for r in rows)
    unresolved = sum(r['status'] == 'ok' and r['grade'] == 'manual_review' for r in rows)
    native = [r['native_accounting']['returned_tokens'] for r in rows if (r.get('native_accounting') or {}).get('complete')]
    srows = [safety.get((r['pass'], r['suite'], r['id'], r['seed'], r['arm'])) for r in rows]
    verdicts = Counter(ref['verdict'] for s in srows if s for ref in s['references'])
    return {
        'scheduled_observed': n, 'correct': correct, 'unresolved': unresolved,
        'correct_bounds': [correct, correct + unresolved], 'rate_bounds': [correct / n, (correct + unresolved) / n] if n else None,
        'lane_calls': len(calls), 'lane_calls_succeeded': len(ok),
        'replies_using_tool': sum(bool(common.compute_calls(r.get('tensorfold'))) for r in rows),
        'executors': dict(Counter(str(c.get('executor', 'code tool (lane only)')) for c in calls)),
        'runner_cells': sum(common.runner_owned(r.get('tensorfold')) for r in rows),
        'replies_with_runner_cells': sum(common.runner_owned(r.get('tensorfold')) > 0 for r in rows),
        'cut_at_cap': sum(r.get('finish_reason') == 'length' for r in rows),
        'cut_at_deadline': sum(r.get('finish_reason') == 'deadline' for r in rows),
        'client_timeouts': sum(r['status'] == 'timeout' for r in rows),
        'errors': sum(r['status'] == 'error' for r in rows),
        'mean_completion_tokens': mean(usage), 'completion_usage_rows': len(usage),
        'mean_native_returned_ids': mean(native), 'native_complete_rows': len(native),
        'seconds_per_reply_mean': mean(walls), 'seconds_per_task': sum(walls) / tasks if tasks else None,
        'decode_tps_aggregate': decoded / sum(spans) if spans and sum(spans) > 0 else None,
        'decode_tps_median_reply': statistics.median([r['client_decode_tps'] for r in rows if r.get('client_decode_tps')])
        if any(r.get('client_decode_tps') for r in rows) else None,
        'safety_qualified_rows': sum(bool(s and s['qualified']) for s in srows),
        'safety_rows_missing': sum(s is None for s in srows),
        'reference_verdicts': dict(verdicts),
        'binding_qualified_rows': sum(bool(s and s['binding'].get('qualified')) for s in srows),
    }


def relabel(rows, treatment, comparator, names):
    out = []
    for r in rows:
        if r['arm'] in (treatment, comparator):
            out.append(dict(r, arm=names[0] if r['arm'] == treatment else names[1], arm_original=r['arm']))
    return out


def std_comparison(result):
    out = {}
    for suite, s in result['suites'].items():
        out[suite] = {
            'treatment_solved': s['lane']['solved'], 'P_solved': s['plain']['solved'],
            'scheduled_per_arm': s['scheduled_per_arm'],
            'difference': s['accuracy_difference_lane_minus_plain'], 'ci95': s['accuracy_ci95'],
            'sign_flip_p': s['sign_flip']['p_two_sided'], 'holm_p': s['sign_flip']['holm_adjusted_p'],
            'accuracy_label': s['accuracy_label'], 'unresolved': s['bounds']['unresolved'],
            'difference_adverse': s['bounds']['difference_adverse'],
            'difference_favorable': s['bounds']['difference_favorable'],
            'adverse_holm_p': s['bounds']['adverse_sign_flip']['holm_adjusted_p'],
            'gains': len(s['gains']), 'losses': len(s['losses']),
            'native_ids_ratio': s['tokens']['paired_sums_ratio_lane_over_plain'], 'native_ids_ratio_ci95':
                s['tokens']['ratio_ci95'], 'native_ids_claim_withheld': s['tokens']['full_cohort_claim_withheld'],
            'wall_ratio': s['time']['paired_sums_ratio_lane_over_plain'], 'wall_ratio_ci95': s['time']['ratio_ci95'],
            'primary_statistical_benefit_supported': s['primary_statistical_benefit_supported'],
        }
    return out


def cdl_comparison(result):
    def block(r):
        costs = r['costs']
        return {'treatment_solved': r['arms']['L']['solved'], 'P_solved': r['arms']['D']['solved'],
                'scheduled_per_arm': r['arms']['L']['scheduled'], 'difference': r['solve_difference_L_minus_D'],
                'ci95': r['solve_difference_ci95'],
                'sign_flip_p': (r['sign_flip'] or {}).get('p_two_sided'),
                'gains': len(r['gains']), 'losses': len(r['losses']),
                'native_ids_ratio': costs['returned_ids'].get('ratio_L_over_D'),
                'native_ids_ratio_ci95': costs['returned_ids'].get('ratio_ci95'),
                'wall_ratio': costs['wall_seconds'].get('ratio_L_over_D'),
                'wall_ratio_ci95': costs['wall_seconds'].get('ratio_ci95'),
                'format_rejections_treatment': r['arms']['L']['format_rejections'],
                'format_rejections_P': r['arms']['D']['format_rejections']}
    return {'status': result['status'], 'overall': block(result['overall']),
            'families': {f: block(v) for f, v in result['families'].items()}}


def activation(rows):
    calls = [c for r in rows for c in common.compute_calls(r.get('tensorfold'))]
    per_seed = {str(s): sum(1 for r in rows if r['seed'] == s for c in common.compute_calls(r.get('tensorfold'))
                            if common.successful_call(c)) for s in common.SEEDS}
    return {'calls': len(calls), 'delivered_successes': sum(map(common.successful_call, calls)),
            'replies_using_tool': sum(bool(common.compute_calls(r.get('tensorfold'))) for r in rows),
            'replies': len(rows), 'per_seed_successes': per_seed,
            'distinct_tasks_with_success': len({r['id'] for r in rows if any(
                common.successful_call(c) for c in common.compute_calls(r.get('tensorfold')))}),
            'label': 'DECLARED BUT UNUSED' if not calls else 'USED'}


def build(args):
    freeze = Path(args.freeze)
    manifest = common.read_json(freeze / 'manifest.json')
    std_manifest = common.read_json(freeze / 'analysis-stdbench.json')
    cdl_tasks = common.read_json(freeze / 'analysis-compute-dl.json')['tasks']
    graded = Path(args.graded)
    std = keyed(common.read_jsonl(graded / 'stdbench-graded.jsonl'))
    reviewed = 0
    if args.math_reviewed:
        for row in common.read_jsonl(args.math_reviewed):
            k = (row['pass'], row['suite'], row['id'], row['seed'], row['arm'])
            common.require(k in std and std[k]['suite'] == 'math500', 'reviewed row outside the MATH grid')
            std[k] = dict(row, censored=p2analysis.censored(row))
            reviewed += 1
    cdl_rows = common.read_jsonl(graded / 'compute-dl-graded.jsonl')
    safety, diags = safety_rows(args.runs)
    cdl_analysis = common.cdl('analysis')
    report = {'freeze_root_sha256': (freeze / 'file-hashes.sha256').read_text().split()[0],
              'model': manifest['model'], 'math_reviewed_rows': reviewed, 'passes': {},
              'primary': manifest['primary'], 'dsv41_published': DSV41, 'caveats': CAVEATS,
              'missing_evidence': [
                  'Native call/result binding uses the server\'s own /detokenize of the reply IDs (no independent '
                  'CPU Inkling tokenizer or frame parser): reported as server-decoder binding.',
                  'Clipped lane results are checked as a byte prefix of the independent reference plus the 512-token '
                  'marker, with token counts from the server\'s /tokenize; not DSv4.1 A1\'s CPU-tokenizer verifier.',
                  'No CPU-rendered prompt arrays: initial-prompt and processed-input ID totals are missing evidence '
                  '(compute-DL analysis reports them as unknown, never zero).',
                  'In-thinking runner cells (arm T) have no independent oracle here (DSv4.1 used owned-oracle r7); '
                  'they are counted, not verified.',
              ]}
    for pass_name in common.PASSES:
        prow = [r for r in std.values() if r['pass'] == pass_name]
        crow = [r for r in cdl_rows if r['pass'] == pass_name]
        entry = {'stdbench': {'arms': {}, 'activation': {}, 'comparisons': {}},
                 'compute_dl': {'arms': {}, 'activation': {}, 'comparisons': {}}}
        for suite in common.STD_SUITES:
            for arm in common.ARMS:
                rows = [r for r in prow if r['suite'] == suite and r['arm'] == arm]
                entry['stdbench']['arms'].setdefault(suite, {})[arm] = arm_summary(rows, safety, common.STD_COUNTS[suite])
                if arm != 'P':
                    entry['stdbench']['activation'].setdefault(suite, {})[arm] = activation(rows)
        for arm in common.ARMS:
            rows = [r for r in crow if r['arm'] == arm]
            entry['compute_dl']['arms'][arm] = arm_summary(rows, safety, 120)
            if arm != 'P':
                entry['compute_dl']['activation'][arm] = activation(rows)
        for treatment in ('L', 'T'):
            name = treatment + '-P'
            std_result = p2analysis.analyze(relabel(prow, treatment, 'P', ('lane', 'plain')), std_manifest)
            entry['stdbench']['comparisons'][name] = std_comparison(std_result)
            cdl_result = cdl_analysis.analyze(relabel(crow, treatment, 'P', ('L', 'D')), cdl_tasks)
            entry['compute_dl']['comparisons'][name] = cdl_comparison(cdl_result)
        entry['diagnostics'] = {'rows': sum(d['pass'] == pass_name for d in diags),
                                'qualified': sum(d['pass'] == pass_name and d['qualified'] for d in diags)}
        report['passes'][pass_name] = entry
    return report


def pct(x):
    return '—' if x is None else f'{100 * x:+.2f}'


def ci(v):
    return '—' if not v else f'[{100 * v[0]:+.2f}, {100 * v[1]:+.2f}]'


def ratio(x, c=None):
    if x is None:
        return '—'
    return f'{x:.3f}' + (f' [{c[0]:.3f}, {c[1]:.3f}]' if c else '')


def num(x, fmt='{:.1f}'):
    return '—' if x is None else fmt.format(x)


def markdown(report):
    lines = ['# Inkling 2 Mini compute-lane rerun (DSv4.1 stdbench + compute-DL)', '',
             f"Freeze root `{report['freeze_root_sha256']}` · model `{report['model']}` · MATH rows reviewed: "
             f"{report['math_reviewed_rows']}", '',
             f"Primary: stdbench {report['primary']['stdbench']}; compute-DL {report['primary']['compute_dl']}. "
             'T-P and the 32k pass are secondary.', '']
    for pass_name, entry in report['passes'].items():
        cap = common.PASSES[pass_name]
        lines += [f"## Pass {pass_name}: max_tokens {cap['max_tokens']}, deadline_s {cap['deadline_s']}", '',
                  '| Set | Arm | Correct | Unresolved | Calls (ok) | Replies using tool | Runner cells | Cap | Deadline '
                  '| Mean completion tok | s/task | Decode tok/s | Safety-qualified |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        sets = [(s, entry['stdbench']['arms'][s]) for s in common.STD_SUITES] + [('compute_dl', entry['compute_dl']['arms'])]
        for name, arms in sets:
            for arm, a in arms.items():
                lines.append(f"| {name} | {arm} | {a['correct']}/{a['scheduled_observed']} | {a['unresolved']} | "
                             f"{a['lane_calls']} ({a['lane_calls_succeeded']}) | {a['replies_using_tool']} | "
                             f"{a['runner_cells']} | {a['cut_at_cap']} | {a['cut_at_deadline']} | "
                             f"{num(a['mean_completion_tokens'])} | {num(a['seconds_per_task'])} | "
                             f"{num(a['decode_tps_aggregate'])} | {a['safety_qualified_rows']}/{a['scheduled_observed']} |")
        lines += ['', '| Set | Contrast | Treatment / P solved | Δ pp | CI95 pp | exact p | Holm p | adverse / favourable '
                  'Δ pp | native-ID ratio [CI] | wall ratio [CI] |', '|---|---|---|---:|---|---:|---:|---|---|---|']
        for contrast in ('L-P', 'T-P'):
            for suite, s in entry['stdbench']['comparisons'][contrast].items():
                lines.append(f"| {suite} | {contrast} | {s['treatment_solved']} / {s['P_solved']} of "
                             f"{s['scheduled_per_arm']} | {pct(s['difference'])} | {ci(s['ci95'])} | "
                             f"{num(s['sign_flip_p'], '{:.4g}')} | {num(s['holm_p'], '{:.4g}')} | "
                             f"{pct(s['difference_adverse'])} / {pct(s['difference_favorable'])} | "
                             f"{ratio(s['native_ids_ratio'], s['native_ids_ratio_ci95'])} | "
                             f"{ratio(s['wall_ratio'], s['wall_ratio_ci95'])} |")
            o = entry['compute_dl']['comparisons'][contrast]['overall']
            lines.append(f"| compute_dl | {contrast} | {o['treatment_solved']} / {o['P_solved']} of "
                         f"{o['scheduled_per_arm']} | {pct(o['difference'])} | {ci(o['ci95'])} | "
                         f"{num(o['sign_flip_p'], '{:.4g}')} | — | — | {ratio(o['native_ids_ratio'], o['native_ids_ratio_ci95'])} | "
                         f"{ratio(o['wall_ratio'], o['wall_ratio_ci95'])} |")
        lines.append('')
    d = report['dsv41_published']
    lines += ['## Against DSv4.1\'s published L-P on the same sets', '',
              '| Set | DSv4.1 P → L | DSv4.1 L-P pp [CI] | Inkling c8k L-P pp [CI] | Inkling c8k T-P pp [CI] |'
              ' Inkling c32k L-P pp | Inkling c32k T-P pp |', '|---|---|---|---|---|---|---|']
    for suite in common.STD_SUITES + ('compute_dl',):
        pub = d['compute_dl'] if suite == 'compute_dl' else d['stdbench'][suite]
        cells = []
        for pass_name in ('c8k', 'c32k'):
            for contrast in ('L-P', 'T-P'):
                comp = report['passes'][pass_name]['compute_dl' if suite == 'compute_dl' else 'stdbench']['comparisons'][contrast]
                s = comp['overall'] if suite == 'compute_dl' else comp[suite]
                cells.append(f"{pct(s['difference'])} {ci(s['ci95'])}" if pass_name == 'c8k' else pct(s['difference']))
        lines.append(f"| {suite} | {pub['P']} → {pub['L']} | {pub['L_minus_P_pp']:+.2f} "
                     f"[{pub['ci95_pp'][0]:+.2f}, {pub['ci95_pp'][1]:+.2f}] | " + ' | '.join(cells) + ' |')
    lines += ['', 'Sources: ' + d['source_stdbench'] + '; ' + d['source_compute_dl'] + '. DSv4.1 settings: '
              + d['settings'] + '.', '', '### Caveats', ''] + ['- ' + c for c in report['caveats']]
    lines += ['', '### Missing evidence', ''] + ['- ' + c for c in report['missing_evidence']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--graded', required=True)
    parser.add_argument('--math-reviewed')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    common.require(not out.exists(), 'output exists: ' + str(out))
    report = build(args)
    out.mkdir(parents=True)
    common.write_json_new(out / 'report.json', report)
    common.write_new(out / 'REPORT.md', markdown(report))
    print(common.canonical({'report': str(out / 'REPORT.md'), 'report_sha256': common.sha_file(out / 'report.json')}))


if __name__ == '__main__':
    main()
