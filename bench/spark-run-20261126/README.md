# Spark run, 2026-11-26 (dgx-02)

One DGX Spark (GB10, `sm_121`), the `Inkling-2-Mini` checkpoint (shard `65e60f51aa12`, MTP `7c3b651317f0`) and the
DFlash2 drafter (`217980ab00d1`).

**Images:**
- **ours:** `tensorfold-inkling2:c4-dc2ca8a-runner`, image ID `sha256:5bf8f0300351…`. It is TensorFold `dc2ca8a1` plus
  `patches/0001`–`0026` (label `tf.patches=43e0fb792314`), built with `RUNNER=1` from branch `feat/tool-full`.
- **main:** `tensorfold-inkling2:main-920fe22`, TensorFold main `920fe22c97995d3dae9bed3e320f08aaa64c25ae` with no patches,
  built by the same `scripts/prepare.sh`.

The boots ran one after another with [`spark-run.sh`](spark-run.sh), then the reruns with [`spark-run3.sh`](spark-run3.sh).
Every boot runs `./start.sh` and the checks, then `./stop.sh`. Each boot's folder keeps:
- the logs: `start.log` and `engine.log`;
- each check's output, as `<check>.txt`;
- the lane rows, as `lane-*.jsonl`;
- `memory`: the largest drop in `MemAvailable` during the boot, sampled every 2 s.

## Boots

| Boot | Image | Settings |
| --- | --- | --- |
| `main`, `main-2` | main | `DRAFTS=0 KV_DTYPE=bf16 VISION=0 PROMPT_CACHE_DISK=0`: main refuses a drafter, vision, fp8 KV and the disk tier |
| `ours-plain` | ours | the same settings as main: engine against engine |
| `ours-off`, `ours-off-2` | ours | the recipe defaults: 2 × 131,072, fp8 KV, DFlash2, vision, states on disk, lane off |
| `ours-lane`, `ours-lane-2` | ours | defaults plus `COMPUTE_LANE=1` |
| `ours-toolfull-2` | ours | defaults plus `COMPUTE_LANE=tool-full` |

The first `features.py` and `speed.py` runs, in `main` and `ours-off`, failed to start: the files lacked execute permission.
`main-2` and `ours-off-2` reran them. `ours-toolfull` was stopped before it started.

## Step 0: TensorFold main against this recipe's engine

Decode is the sum over streams in tok/s, median of 3, 512 tokens. "sampled" is temperature 0.7, top_k 20, thinking on;
"greedy" is temperature 0, thinking off. Prefill is a fresh prompt with 1 token out: prompt tokens / time to first token,
median of 3. Measured at the client by [`tools/speed.py`](../../tools/speed.py).

| | main 920fe22c | ours, main's settings | ours, defaults (DFlash2) |
| --- | ---: | ---: | ---: |
| sampled prose ×1 / ×2 | 25.7 / 41.2 | 40.5 / 65.0 | **54.9 / 89.0** |
| sampled code ×1 / ×2 | 25.6 / 39.5 | 40.4 / 62.2 | **72.0 / 101.2** |
| greedy prose ×1 / ×2 | 25.8 / 47.6 | 44.0 / 84.0 | **70.6 / 127.6** |
| greedy code ×1 / ×2 | 25.7 / 47.2 | 43.6 / 83.2 | **105.6 / 181.3** |
| prefill 8k | 256 | 3,280 | 3,085 |
| prefill 32k | 256 | 3,080 | 2,842 |
| prefill 128k | 251 | 2,560 | 2,144 |
| peak memory (GiB) | 36.1 / 60.7 | 59.4 | 41.8 / 49.9 |

- **Engine against engine** (the same settings, no drafter): ours decodes about 1.6× faster at one stream and 1.6–1.8× at
  two. It prefills 10–13× faster: main's prompt path stays at about 256 tok/s at every length.
- **Our defaults** add DFlash2, which gives 2.1–4.1× main's decode.
- **Prefill on the defaults** is 6–16% slower than at main's settings. The defaults add fp8 KV, the DFlash2 drafter (which
  runs its own pass over the prompt) and the vision tower; the drafter's pass is the likely cause. Not investigated.
- **Main's peak** differs between its two boots: 36.1 GiB in the first, which had no 128k prefill, and 60.7 GiB with it.

**Feature checks** ([`tools/features.py`](../../tools/features.py)):

| Check | main | ours |
| --- | --- | --- |
| `json_schema` (greedy, sampled, thinking), `json_object`, `guided_regex`, `guided_choice` | **HTTP 500** ("Inkling 2 Mini serves only unconstrained, non-branching text drafts") | pass |
| `tool_choice` required, named | **HTTP 500** | pass |
| `tool_choice` auto, none | pass | pass |
| vision | refused at boot (`Inkling2LoopTextOnly`) | corrupt image → 400 and drafted = `"draft": false` pass; the two colour checks returned empty content (see below) |

The two vision colour checks used `max_tokens` 16. Inkling reasons even with reasoning off, so the reply ended before any
content. This is a fault in the check, not a measured vision failure, but it has not been rerun.

**Exactness** (`tools/exact.py`):
- ours, defaults: PASS at no pad and at 3k. Drafted equals `"draft": false`, and 6 concurrent replies equal serial ones.
- ours, with the lane on and with tool-full: PASS.
- main and ours at main's settings: UNCHECKED. Neither has a drafter, so drafted against serial was not compared. 6
  concurrent replies equal serial ones.

**Prompt reuse** (`tools/prompt_reuse.py`, 30k, 4 turns, another conversation in between): PASS on both, worst 99.4%.

**Base decision:** this recipe stays on `dc2ca8a1` + patches. On CUDA, main has:
- no drafter, no vision, no fp8 KV and no disk tier;
- no structured output, and no forced (`required` or named) tool calls;
- a prompt path 10× slower.

## Lane A/B (`tools/lane_ab.py`), recipe defaults

These are 20 generated questions (seed 1): modular powers, long products, factorial digit sums, Collatz step counts and
prime counts, each with an exact answer. Thinking is on, at temperature 0.7, top_p 0.95 and top_k 20. Every arm had the
same budget as AB3: `max_tokens` 32,768 and `deadline_s` 300. Two streams ran at once.

| Arm | Correct | Cut at deadline | Completion tokens / question | Seconds / question | Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| P, no tools | 7/20 | 13 | 12,741 | 242.2 | 0 |
| D, `python` declared, each call answered "execution disabled" | 6/20 | 13 | 14,136 | 256.2 | 20 client rounds |
| L, `COMPUTE_LANE=1` | **20/20** | 0 | 547 | 13.2 | 39 lane calls |
| T, `COMPUTE_LANE=tool-full` | **20/20** | 0 | 547 | 13.2 | 39 |

- **T against L:** T's replies equal L's, token for token. No runner cell was written in a T reply, because the model
  called Python at once on every question, and the call results were the same.
- **The runner does work in this image:** it passed its probe at boot (6 cells, 534 ms) and wrote cells in 3 replies of
  the boot's other checks (`ours-toolfull-2/engine.log`).
- **Executors:** `lane_ab.py` does not record whether S3 or the sandbox ran a call. The benchmark in
  `bench/compute-lane-inkling/` does.
- **Earlier capped run:** `ours-off` and `ours-lane` first ran 30 questions at 6,144 tokens with no deadline. P got 0/30 and
  D 2/30, every reply cut at the cap; L got 30/30. That cap made the P and D arms meaningless, so the run was redone above
  with AB3's budget.
