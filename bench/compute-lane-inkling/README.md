# compute-lane-inkling: the DSv4.1 compute-lane studies, rerun on Inkling 2 Mini

This reruns DSv4.1's stdbench (GSM8K 48, MATH-500 20, HumanEval 24, seeds 101 and 102) and compute-DL (120 tasks,
2 seeds) on Inkling 2 Mini, served by this recipe on one DGX Spark. There are three arms:

- P: no tool.
- L: `COMPUTE_LANE=1` with `code_interpreter`.
- T: `COMPUTE_LANE=tool-full` with `code_interpreter`.

It runs two passes, 8k/270 s and 32k/300 s. The protocol is `PROTOCOL.md`. The unchanged DSv4.1 kit files live in
`dsv41/`, and their hashes are in `dsv41/COPIED.sha256`. Everything is stdlib Python 3.12 plus bash, and nothing
here talks to anything but 127.0.0.1:8888 and the local docker.

## Files

| Path | Role |
|---|---|
| `PROTOCOL.md` | the frozen amendment: arms, passes, turns, gates, grading, analysis, departures |
| `freeze.sh`, `ink/freeze.py` | `create`: cohort, boots, bodies, hashes, MATH preseal. `bind`: the image ID. `verify` |
| `run.sh`, `ink/controller.py` | one turn = three boots through `./start.sh` / `./stop.sh`, with gates, diagnostics, requests and verdicts |
| `ink/ink_client.py` | streaming client: raw SSE kept; the kit's `parse_stream` / `accounting` |
| `ink/safety.py` | arm gates, jail references per delivered call, server-decoder binding, the `InkJail` image pin |
| `jail-controls.sh`, `ink/jail_ink.py` | HumanEval/reference jail qualification (the kit's `qualify`) |
| `ink/grade.py` | GSM8K, HumanEval (jail) and compute-DL graders, plus coverage against the frozen grid |
| `ink/math_packet.py` | MATH blinded packet (`seal`), third-review packet (`disputes`), `merge` |
| `ink/report.py` | per-arm tables, L−P / T−P (unchanged kit analyses) and the DSv4.1 comparison |
| `ink/common.py` | constants and shared helpers |
| `selftest.sh`, `ink/selftest.py`, `ink/fake_server.py` | CPU self-tests with a fake SSE server; no model |

## On each Spark

The tree is a copy of this recipe at `~/work/inkling-recipe`, with no `.env`. The runner image
`tensorfold-inkling2:c4-dc2ca8a-runner` is built from that same tree (`RUNNER=1`). Port 8888 is free.

```bash
cd ~/work/inkling-recipe/bench/compute-lane-inkling
./selftest.sh ~/work/ink-selftest --quick            # optional, CPU only, about 3 minutes
./freeze.sh create ~/work/ink-runs/freeze            # print root_sha256: it must be identical on both Sparks
./freeze.sh bind ~/work/ink-runs/freeze spark1       # on the second Spark: spark2
./jail-controls.sh ~/work/ink-runs/freeze spark1 ~/work/ink-runs/jail-spark1   # 102 cases; about 4 min
```

### Collection: one pass per Spark, run in parallel

spark1 runs c8k (the primary pass):

```bash
cd ~/work/inkling-recipe/bench/compute-lane-inkling
nohup ./run.sh ~/work/ink-runs/freeze spark1 c8k s01 s02 s03 s04 s05 s06 > ~/work/ink-runs/c8k-stdbench.log 2>&1
nohup ./run.sh ~/work/ink-runs/freeze spark1 c8k c01 c02 > ~/work/ink-runs/c8k-compute-dl.log 2>&1
```

spark2 runs c32k:

```bash
cd ~/work/inkling-recipe/bench/compute-lane-inkling
nohup ./run.sh ~/work/ink-runs/freeze spark2 c32k s01 s02 s03 s04 s05 s06 > ~/work/ink-runs/c32k-stdbench.log 2>&1
nohup ./run.sh ~/work/ink-runs/freeze spark2 c32k c01 c02 > ~/work/ink-runs/c32k-compute-dl.log 2>&1
```

Run the two lines of a Spark one after the other, joined with `&&` (for example `nohup bash -c '… && …' &`),
never at the same time. `run.sh` exits non-zero when a boot ends `STOPPED`, so `&&` keeps the next pass from starting. Each turn is three boots in the frozen order, all three arms, for example:

- s01: P, L, T.
- c01: P, L, T.
- c02: T, L, P.

A boot that ends `STOPPED` stops the turn and keeps everything. Read `runs/<pass>/<turn>/<boot>/verdict.json` and
`steps.jsonl`, then continue with `./run.sh … <pass> <turn> -- --resume`: it starts a fresh boot, runs the
diagnostics again, then runs only the unobserved keys. If another server is up, `-- --stop-existing` stops it with
`./stop.sh`.

A single Spark can also run both passes: run all four lines one after another.

### Bring the runs together (on spark1)

```bash
rsync -a spark2:work/ink-runs/runs/c32k/ ~/work/ink-runs/runs/c32k/
scp spark2:work/ink-runs/freeze/hosts/spark2.json ~/work/ink-runs/freeze/hosts/
python3 -I ink/freeze.py verify --freeze ~/work/ink-runs/freeze --recipe ~/work/inkling-recipe
```

### Grading

```bash
python3 -I ink/grade.py --freeze ~/work/ink-runs/freeze --runs ~/work/ink-runs/runs --host spark1 \
  --jail-qualification ~/work/ink-runs/jail-spark1/qualification.json --out ~/work/ink-runs/graded
```

Add `--partial --out <new dir>` for a progress check while collection is still running.

### MATH blind review

```bash
mkdir -m 700 ~/work/ink-runs/math
python3 -I ink/math_packet.py seal --freeze ~/work/ink-runs/freeze \
  --responses ~/work/ink-runs/runs/*/s0[25]/*/responses.jsonl \
  --packet ~/work/ink-runs/math/packet.json --private-key ~/work/ink-runs/math/key.json
```

The seal prints `private_key_sha256`; keep it. Then:

1. Give `packet.json` (only that file) to two reviewers. Each returns a JSON array of
   `{"candidate_id", "reviewer_id", "label": "correct"|"incorrect"|"unresolved", "reason"}`, one entry for every
   candidate, saved as `a.json` and `b.json`.
2. Build the third-review packet from the disagreements:

   ```bash
   python3 -I ink/math_packet.py disputes --packet ~/work/ink-runs/math/packet.json \
     --first ~/work/ink-runs/math/a.json --first-id REVIEWER_A --second ~/work/ink-runs/math/b.json \
     --second-id REVIEWER_B --out ~/work/ink-runs/math/third-packet.json
   ```

3. A third reviewer labels `third-packet.json`, saved as `c.json`.
4. Merge:

   ```bash
   python3 -I ink/math_packet.py merge --packet ~/work/ink-runs/math/packet.json \
     --private-key ~/work/ink-runs/math/key.json --key-sha256 <private_key_sha256 from seal> \
     --first ~/work/ink-runs/math/a.json --first-id REVIEWER_A --second ~/work/ink-runs/math/b.json \
     --second-id REVIEWER_B --third ~/work/ink-runs/math/c.json --third-id REVIEWER_C \
     --out ~/work/ink-runs/math/math-reviewed.jsonl --audit ~/work/ink-runs/math/math-audit.json
   ```

### Report

```bash
python3 -I ink/report.py --freeze ~/work/ink-runs/freeze --runs ~/work/ink-runs/runs \
  --graded ~/work/ink-runs/graded --math-reviewed ~/work/ink-runs/math/math-reviewed.jsonl \
  --out ~/work/ink-runs/report
```

This writes `report/REPORT.md` and `report/report.json`. Without `--math-reviewed`, every MATH reply counts as
unresolved, so the MATH bounds span everything.
