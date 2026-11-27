# Does the Compute Lane break Inkling 2 Mini's loop?

Engine under test: Mia's dc2ca8a1 + patches 0001–0026, image `tensorfold-inkling2:c4-dc2ca8a-runner`.
The plan is [PLAN.md](PLAN.md), frozen before any measurement.

## Answers

**Q1, state integrity: no verdict.** JJ stopped Q1 on this engine on 2026-11-26; it will be redone on the new
recipe's engine. What was measured supports "no sign that the lane breaks the loop" and does not prove it:
every lane variant gives the lane's exact tokens (30 of 30 under exact and under shipped kernels), and under exact
kernels every state sum of the lane equals its drafter-off and warm repeats. The frozen gate (lane = client round
trip = cold prefill) **fails as written**, and it fails the same way with no lane in the path (0 of 48 cuts of a
plain reply), because this engine's prompt pass and its decode rows are not bit-identical. The check that would
clear the lane, against a lane-free decode of the same rows, was built and not run. Details, numbers and what is
pending: [Q1-STATUS.md](Q1-STATUS.md).

**Q2, thinking continuity: the lane does not cut thinking short by the plan's rule, and it is not free.**

- *Does the tool turn close thinking?* Yes, by Inkling's own format, not by the engine: a call is its own message
  after a closed thinking message and the model's EOS, and the result is a tool turn after that. The engine forces
  nothing but the tool turn and cannot force a thinking close on Inkling. The model reopens thinking itself after
  339 of 340 results.
- *Is the call made inside thinking, like DSv4.1's runner?* No: 0 of 344 calls. In the tool-full arm the in-model
  runner writes into whatever message is open. T's tokens differ from L's in 42 of 120 reasoning-heavy replies (the
  runner acted in 24 of them; a sandbox result can also differ from run to run) and in none of the controls.
- *Does accuracy on reasoning-heavy questions drop?* L 104/120 against P 109/120: −4.2 points (95% interval −10.8
  to +1.7; 5 wins, 10 losses; exact McNemar p = 0.30). The plan's thresholds (p < 0.05, or −7.5 points) are not
  met. The sample cannot rule out a loss of about 5 points; tool-full is no better (102/120, −5.8, p = 0.12).
- *Is the lane working?* Yes: on the compute controls L is 44/44 against P 22/44.

No fix patch was built and no shipped default changed. The losses have named causes in the sandbox contract
(below), none of them in the loop or the thinking format.

## Q2 method as run

Arms P (no tools), L (`tools: [{"type":"code_interpreter"}]`, lane on) and T (the same request, tool-full), each
its own boot of the shipped image with shipped kernels on dgx-02. 164 frozen questions × 3 arms = 492 replies, one
request at a time, `max_tokens 32768`, `deadline_s 300`, temperature 1.0, top_p 0.95, top_k 20, effort high,
seed 101, in 14 blocks with the arm order rotating. No request failed. Collected 2026-11-26 12:19Z to
2026-11-27 02:52Z.

## Q2 results

| Set | n | P | L | T | L − P (95%) | p | T − P | T − L | cut P/L/T | median thinking P/L/T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATH-500 level 5 | 30 | 29 | 29 | 29 | +0.0 (+0.0, +0.0) | 1.000 | +0.0 | +0.0 | 0/0/0 | 1598/1982/1896 |
| AIME 2025 | 30 | 27 | 25 | 24 | −6.7 (−20.0, +6.7) | 0.625 | −10.0 | −3.3 | 3/3/5 | 6998/4099/4400 |
| ZebraLogic | 30 | 28 | 25 | 25 | −10.0 (−23.3, +3.3) | 0.375 | −10.0 | +0.0 | 2/4/4 | 5765/2134/2134 |
| GPQA diamond | 30 | 25 | 25 | 24 | +0.0 (−16.7, +16.7) | 1.000 | −3.3 | −3.3 | 4/5/6 | 8745/10578/10578 |
| **Reasoning-heavy** | 120 | 109 (90.8%) | 104 (86.7%) | 102 (85.0%) | −4.2 (−10.8, +1.7) | 0.302 | −5.8 | −1.7 | 9/12/15 | 5576/3202/3722 |
| Control: arithmetic | 20 | 8 | 20 | 20 | +60.0 (+40.0, +80.0) | <0.001 | +60.0 | +0.0 | 12/0/0 | 19052/240/240 |
| Control: compute-DL | 24 | 14 | 24 | 24 | +41.7 (+20.8, +62.5) | 0.002 | +41.7 | +0.0 | 10/0/0 | 13175/236/236 |
| **Controls** | 44 | 22 (50.0%) | 44 (100%) | 44 (100%) | +50.0 (+36.4, +63.6) | <0.001 | +50.0 | +0.0 | 22/0/0 | 19052/236/236 |

"Cut" is a reply ended by the 300 s deadline or by `max_tokens`. Thinking is in tokens, counted from the token ids.

**Thinking, reasoning-heavy, arm L** (from token ids, split on Inkling's message tokens):

| Measure | Value |
| --- | --- |
| Replies that made a call | 83 of 120 |
| Calls, and calls inside an open thinking message | 344, 0 |
| Thinking tokens before the first call, median | 828 |
| Results followed by a new thinking message | 339 of 340 |
| Thinking tokens after a result, median | 214 |
| Thinking tokens a reply, median (mean): P, L, T | 5,576 (7,375); 3,202 (5,685); 3,722 (5,899) |
| Forced tokens that are not the tool turn | 0 |

On the 83 questions where L made a call, P thought a median 6,617 tokens and L 4,198; P got 75 right and L 69. On
the 37 where L made no call the two arms think alike (1,401 and 1,441) and score alike (34 and 35).

**Where L lost.** Of the 319 reasoning-heavy replies that ended with an answer, 315 were right, so the arms differ
almost only by replies that did not finish. L's 10 losses against P:

- 8 hit the 300 s deadline. 7 of them made calls first, and 35 of their 65 calls did not return a result.
- 2 ended early on a call the lane did not run: the model wrote the `code` argument twice (duplicate key). A third
  reply ended the same way on an invalid JSON escape. The lane ends such a reply as plain text with
  `finish_reason: stop` and no answer; it gives the model no error to retry on.

What the 340 lane calls on reasoning-heavy questions returned: 199 a result; 70 a `NameError`, the model reusing a
variable from an earlier call (each call is a fresh process and the tool's description does not say so); 24 the
call-limit text (`max_calls` is 8 a reply); 16 a numpy import that failed in the sandbox
(`OpenBLAS … pthread_create failed`); 15 the 10 s timeout; 16 other errors.

## What to change, for JJ to decide

None of these is built. Each would be its own patch with its own off-parity and a re-measure on these frozen sets.

1. A malformed call (duplicate key, bad escape) should come back to the model as a tool error and the reply resume,
   as a client would do, instead of ending the reply.
2. Say in the built-in tool's description that nothing persists between calls, or keep the interpreter's state for
   the reply.
3. Let numpy import in the sandbox (thread limit, or `OPENBLAS_NUM_THREADS=1`).
4. Reconsider 8 calls a reply for long problems.

## Q2 static analysis

Checked against the patched tree (paths under `zig/src/`), then on the 492 replies' token ids.

- The template writes thinking, each call and the tool result as separate messages; the assistant turn ends with
  `<|content_model_end_sampling|>` (`families/inkling2/fixtures/chat_template.jinja:71`, `112–124`, `75`). It adds
  no generation prompt, so after a tool turn the model opens its own next message.
- The lane holds the stream at that EOS (`core/lanes/stream.zig:523`) and forces exactly the template's render of
  the tool turns (`server/compute_segments.zig:275–290`, `71`). Measured: forced tokens other than tool turns, 0.
- Inkling's tokenizer has no `</think>`, so `think_close` is empty and the budget and loop-guard close do nothing
  (`server/server.zig:272`, `server/chat.zig:243`). `think_open` after a resume sets a flag and no token
  (`core/lanes/hold.zig:36`).
- Side finding: that flag is false through an Inkling reply's whole first segment and true only after the first
  resume, so `TF_INKLING2_RUNNER_SPANS=reasoning` could not fire in the first segment. Not the shipped default.

## Departures from the plan

- Q1 was stopped before its state-sums gate and has no verdict ([Q1-STATUS.md](Q1-STATUS.md)).
- Q2 ran on dgx-02 alone; the frozen order, seeds and sizes were kept and all 14 blocks completed.
- The AIME grader crashed on one reference answer (`336^\circ`); it now reads the integer from the reference and
  accepts a degree sign. Fixed before any grade was joined to an arm.
- 40 MATH answers the normaliser could not match went to two blind reviewers (model reviewers, shown the problem,
  the official answer and the final answer only). They agreed on all 40 (37 equivalent, 3 not), so no third.
- Runner activity is read from `tensorfold.runner.<form>.owned` and `tensorfold.compute.runner_calls`.
- GPQA text is withheld: its 90 rows in `q2/rows.jsonl.gz` carry no content, reasoning or token ids (hashes and
  counts instead); their measures and grades are complete. The full rows' sha256 is `8b4a8848…f81b6bf`.

## Files

- `PLAN.md`, `Q1-STATUS.md`, this file.
- Q2: `sets/`, `sets.lock.json`, `q2_sets.py`, `q2.py`, `q2_all.sh`, `q2_report.py`; `q2/rows.jsonl.gz` (492
  replies with token ids), `q2/boots/` (each boot's `docker run` line, image id and settings), `q2/collect.log`,
  `q2/marks.json`; `q2/grading/`: `packet.jsonl` and `grades.jsonl` (blind, committed before the join),
  `review*.jsonl`, `key.jsonl`, `measures.jsonl`, `summary.json`, `summary.md`.
- Q1: `boot.sh`, `lli.py`, `q1.py`, `q1_all.sh`, `q1_sums.py`, `q1_report.py`, `engine/` (the test-only state-sums
  patch, off by default, and its build script). Q1's raw receipts are on dgx-01 and are not committed.

Reproduce Q2's tables: `python3 q2_report.py join --rows <full rows> --out q2/grading --marks q2/marks.json --sets
sets,<private GPQA folder>` (the committed rows reproduce everything but GPQA's thinking measures).
