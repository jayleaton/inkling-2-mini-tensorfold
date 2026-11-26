# Inkling 2 Mini compute-lane rerun: protocol amendment to the DSv4.1 stdbench and compute-DL specs

This file is frozen by `ink/freeze.py` (manifest `specs_sha256`) together with the unchanged DSv4.1 specs under
`dsv41/`. Where this file is silent, the DSv4.1 spec governs. Every departure is listed in section 9.

## 1. Question

Does the TensorFold compute lane improve Inkling 2 Mini on the same task sets where DSv4.1 measured its lane?
The contrasts are L−P (the code_interpreter lane against no tool) and T−P (the lane plus the in-model runner,
`COMPUTE_LANE=tool-full`, against no tool). The **primary** contrasts are L−P at the 8,192-token pass: per suite
for stdbench (Holm over its three suites) and overall for compute-DL (one exact task sign-flip). T−P and the
32,768-token pass are secondary. All are reported.

## 2. Sets (unchanged)

- stdbench: GSM8K test (48), MATH-500 test (20) and HumanEval (24), chosen by the kit's `bench.py normalize`
  (the first n by SHA256("stdbench-v1:"+id)), with seeds 101 and 102. Blocks are contiguous halves of each
  suite's selection, the first block taking the extra task. Each block's sorted (task, seed) pairs get one fresh
  `random.Random(20261104).shuffle`.
- compute-DL v1: 120 public tasks with private answers A, seeds 101 and 102. Turn 1 holds slots 1–10 and turn 2
  slots 11–20. Per family, the pairs are sorted by (id, seed) and get a fresh `random.Random(20261105).shuffle`;
  the families are concatenated in the order integer, modular, division, dp, graph, aggregation.

## 3. Serving

Every request goes to one DGX Spark running the TensorFold recipe (`./start.sh`, container `inkling2-mini-tf`,
127.0.0.1:8888) on recipe defaults: context 131,072, parallel 2, fp8 KV, the DFlash2 drafter, disk states, vision,
thinking, and no `.env`. Every boot uses `RUNNER=1 IMAGE=tensorfold-inkling2:c4-dc2ca8a-runner PREPARE=0`, so all
three arms run one image. Each image's `tf.patches` label must equal the recipe's `image_hash` at freeze time.

| Arm | Environment | Request tools |
|---|---|---|
| P | `COMPUTE_LANE=0` | none |
| L | `COMPUTE_LANE=1` | `[{"type":"code_interpreter"}]` |
| T | `COMPUTE_LANE=tool-full` (lane + in-model runner, S3) | `[{"type":"code_interpreter"}]` |

Each boot gets a fresh, empty `SESSION_DIR`, so no disk state crosses arms or boots. The removed directory's size
is recorded in the verdict. start.sh's own smoke request ("Reply with OK.") is the only warmup, and it is the
same for every arm.

## 4. Requests

The body is the frozen `boots/<boot>.json` `body_template` exactly; its sha256 is checked before sending:

- `model`: the served name, frozen at freeze time and asserted against `/v1/models` at every boot.
- `messages`: one user turn with the task prompt.
- `seed`.
- `temperature` 1.0, `top_p` 0.95, `top_k` 20.
- `chat_template_kwargs.reasoning_effort` `"high"`.
- `stream` true, with `stream_options.include_usage`.
- `return_token_ids` true.
- `max_tokens` and `deadline_s` from the pass.
- `tools` (L and T only).

The client refuses any other field, because the server drops unknown fields silently. Requests run one at a
time (concurrency 1).

| Pass | max_tokens | deadline_s (server, from receipt) | client wait |
|---|---|---|---|
| c8k | 8,192 | 270 | 360 s |
| c32k | 32,768 | 300 | 390 s |

## 5. Turns and boots

Each turn is three boots of the recipe, one per arm, in the frozen order. P and L keep DSv4.1's positions, and T
sits at a pre-registered rotating position.

| Turn | Set | P/L/T order | T position |
|---|---|---|---|
| s01 | GSM8K block 1 | P, L, T | 3 |
| s02 | MATH block 1 | T, L, P | 1 |
| s03 | HumanEval block 1 | P, T, L | 2 |
| s04 | GSM8K block 2 | T, L, P | 1 |
| s05 | MATH block 2 | P, T, L | 2 |
| s06 | HumanEval block 2 | L, P, T | 3 |
| c01 | compute-DL slots 1–10 | P, L, T | 3 |
| c02 | compute-DL slots 11–20 | T, L, P | 1 |

Both passes run all eight turns: 48 boots and 1,272 scheduled replies per pass. Every boot sends its requests in
the block's frozen order.

## 6. Gates (every boot)

1. Before start:
   - the freeze inventory, the harness files, the copied DSv4.1 files and the recipe tree (start.sh, stop.sh,
     scripts, patches, config, `image_hash`) must equal the freeze;
   - the host binding (image ID and `tf.patches`) must belong to this freeze;
   - port 8888 must be free and no container may be present (`--stop-existing` stops one explicitly).
2. After start:
   - the container's image ID must equal the bound ID;
   - the engine environment must match the arm: the lane on for L and T only, the runner knobs for T only, and
     no compute default;
   - the serve arguments must be the recipe defaults;
   - the served name must equal the frozen one;
   - the server log's `compute lane:` line must be present exactly for L and T.
3. Two diagnostics, both at seed 101: "Use the Python tool to print 17*19, then report the result." (323) and
   "Use the Python tool to print sum(range(11)), then report the result." (55). Each must:
   - complete with full native accounting;
   - report `enable_thinking=true` and `reasoning_effort="high"`;
   - return token IDs.
   - Activation gate: P must deliver zero successful calls. L and T must deliver at least one successful call
     whose output is exactly the reference and matches the independent jail run.
   - Runner cells in T diagnostics are recorded.
4. Each reply is written to disk before any check. Then:
   - Arm gates: P makes no calls, inserts nothing and has no runner cells; P and L have no runner receipt; no arm
     has a runner guide or `compute_executed`.
   - Every inserted call is rerun in the isolated jail and classified as match, clipped_match, mismatch,
     error_match, error_differs or inconclusive.
   - The call's code and output are bound to the reply's own token IDs through the server's `/detokenize`.
   - A **stop** is an arm violation or a mismatch (a wrong ok/ok delivered value). Inconclusive results and
     error differences are recorded as unchecked, and the row is not safety-qualified, but collection continues.
5. After the last request: the server log is saved, the response seeds are checked against the log's `done … seed=`
   lines, `./stop.sh` runs, and the container must be absent. The verdict is QUALIFIED or STOPPED.
6. A STOPPED boot continues only with `--resume`. That runs a fresh boot, including the diagnostics, then only the
   keys no attempt observed, in their frozen order. A key whose request started without a recorded row blocks the
   resume until it is reconciled by hand. No observed key is ever resent.

## 7. Grading (unchanged graders)

- GSM8K uses `phase2/grade_all.py` `gsm_grade`.
- HumanEval uses the original tests, run only in the kit jail (`phase2/jail.py` + `jail_worker.py`) inside the runner
  image, pinned by full ID and `tf.patches`. The jail qualification (`ink/jail_ink.py`, the kit's `qualify`) must
  pass first: canonical passes, wrong fails, timeout fails, `os._exit(0)` fails, plus the six reference cases.
- MATH-500 answers all go to blind review:
  - Opaque IDs come from a preseal written at freeze time: `random.Random(20261105)` shuffles the sorted keys, and
    each key gets `C-` plus 128 random bits.
  - The packet shows only the problem, the official answer and the final content.
  - Two independent reviewers label every candidate. A third labels only the disputes, and the majority decides.
  - Anything still unresolved is reported as adverse/favourable bounds. Generation failures are incorrect.
- compute-DL uses `compute-dl/grade.py`, unchanged: strict JSON extraction and an exact typed answer.

## 8. Analysis (unchanged code, relabelled arms)

- stdbench: `phase2/analysis.py analyze`, with the treatment (L or T) as `lane` and P as `plain`. It draws 20,000
  paired task-cluster bootstraps with RNG 20261104 and runs an exact sign flip per suite, Holm over the three
  suites. Each contrast and pass is a separate family. MATH adverse/favourable bounds are included.
- compute-DL: `compute-dl/analysis.py analyze`, with the treatment as `L` and P as `D`. It draws 20,000
  family-stratified task bootstraps with RNG 20261105 and runs the overall exact sign flip.
- `ink/report.py` also reports, per arm, set and pass:
  - correct counts with bounds;
  - lane calls and successes, and the executors used;
  - runner cells;
  - replies cut at the cap or the deadline;
  - mean completion tokens and native returned IDs;
  - seconds per task;
  - decode tok/s, as completion tokens / (wall − client TTFT);
  - safety-qualified rows.
- The comparison table against DSv4.1's published L−P carries the caveats in report.json.

## 9. Departures from the DSv4.1 specs

1. Model, engine and topology differ: Inkling 2 Mini on one Spark with a single rank, instead of DSv4.1 Flash in
   TP2 on two Sparks.
2. There is a third arm, T, at the pre-registered positions in section 5.
3. The second pass (32k/300 s) is new.
4. L and T declare `code_interpreter`. The server expands it to the built-in `python` function, whose description
   is identical to DSv4.1's explicit declaration.
5. compute-DL puts P in DSv4.1's D slot, interleaved within turns. DSv4.1's P was a later single boot.
6. There is no pilot. The per-boot gates and the reported timings replace it.
7. Native binding uses the server's own decoder (`/detokenize`). There is no independent CPU tokenizer or frame
   parser for Inkling, and this is reported as missing evidence.
8. Clipping is checked as a byte prefix of the reference plus the marker, with token counts from `/tokenize`.
   DSv4.1 A1 used an exact CPU-tokenizer verifier.
9. The jail reference is bounded at 3 s, 512 MiB and no site-packages, which is tighter than the lane's 10 s and
   1 GiB. Inconclusive references are recorded, not stopped.
10. Runner cells (T) have no independent value oracle. They are counted only.
11. No CPU prompt arrays exist, so initial-prompt and processed-input totals are missing evidence (never zero).
12. A fresh per-boot SESSION_DIR is used (section 3).
13. **Void first attempt (2026-11-26, freeze root `8bedc088…`).** The harness counted a lane call as successful only
    with `exit_code` 0. This engine reports `exit_code` only when it is non-zero (patch 0020), so a clean sandbox
    exit has `exit_code` null. The first L boot (`c8k-s01-L`) therefore failed its activation gate although its
    diagnostic call ran and delivered the reference bytes (`55`, matched by the jail). Collection stopped on both
    Sparks before any L or T dataset request. The rule now accepts `status` ok with `exit_code` null or 0. The
    harness was re-frozen under a new root and every turn was collected again from the start. The void attempt's
    rows are kept on each Spark under `~/work/ink-runs/void-8bedc088/` and are not part of any result.
