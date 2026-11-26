# Lane loop integrity: frozen test plan

Frozen 2026-11-26, before any measured request. Engine: TensorFold `dc2ca8a1` + `patches/0001`–`0026` (branch
`feat/tool-full`). Anything done differently from this file is listed as a departure in the README, with its reason.

## Questions

- **Q1, state integrity.** After the lane holds a stream at a call's EOS, runs the sandbox, forces the template's tool
  turn in and resumes, is the stream's state what a fresh prefill of the same token history leaves? Yes or no.
- **Q2, thinking continuity.** Does the lane close or cut Inkling's thinking, and does that cost accuracy on
  problems where reasoning, not computing, is the bottleneck? Yes or no.

## What the state is

From `mini_snap.zig`: a stream carries only the 48 visits' K and V (8 whole-context visits, 40 sliding rings of
1,536 rows read over 1,024) and each visit's four conv states. The gates and the cross-pass messages are rebuilt for
every token, so they show up in that token's K/V rows, taps and logits, not as carried state. The drafter adds its
own rings (DFlash2: 2,048 positions) fed by the taps at visits 5, 23, 41 and the final norm.

Passes by visit: 0–5 before the core, 6–17 pass 1, 18–29 pass 2, 30–41 pass 3, 42–47 after it. Whole-context
visits: 5, 11, 17, 23, 29, 35, 41, 47.

## Q1 method

**Engine settings**, each its own boot:

| Name | KV | `ATTN_FP8` | `PREFILL_TC` | Role |
| --- | --- | --- | --- | --- |
| X | bf16 | 0 | 0 | exact kernels: the bit-exact gate (upstream's "same engine, same settings") |
| S | fp8 | 1 | 1 | the shipped defaults |
| S0 | fp8 | 1 | 0 | only to localise an S mismatch to the tensor-core prefill |

**Arms**, on one token history `H = prompt + seg1 + ins1 + seg2 + ins2 + …` taken from arm A with
`return_token_ids` (`seg`: the model's tokens through its EOS; `ins`: the forced tool turn):

- **A**: the in-engine lane reply (`tools: [{"type":"code_interpreter"}]`).
- **B**: a client round trip with the identical ids as a raw token prompt, one request a round, each extending the
  last one's prompt and reply so the prompt cache resumes it (cached tokens > 0 required).
- **C**: the same requests on a fresh boot with an empty RAM cache and an empty session folder (cached tokens = 0
  required).
- **D**: B, with the container restarted before a middle round, so that round resumes from the NVMe state
  (`0019`; cached tokens > 0 after the restart required). Also a second lane turn on top of a disk-resumed turn.
- **N** (no-lane baseline): a plain reply with no tools, replayed as B and C from cut points at the same position
  classes. It measures prefill-against-decode with no lane in the path.

Arm A variants, all of which must give A's exact tokens: `"draft": false`; a repeat; a second concurrent stream
that is another lane reply; a second concurrent stream that is a plain reply decoding through the hold.

**Samplers**: greedy; sampled `temperature 1.0` (the shipped default, the lazy head's keyed draws); sampled
`temperature 0.7, top_k 20, top_p 0.95` (the full head). Seed 20261126. The draws are keyed by seed and absolute
position, so a replay from a longer prompt draws the same noise.

**Drafters**: DFlash2 (default) and off on every case; DSpark (`DRAFTER=dspark`) on cases 1, 4, 7 and 10, greedy
and sampled 0.7.

**Cases**: 10 scripted prompts that make the model call `python` in sequence, each call needing the last result.
A calibration request per case sets the padding; a case counts only when its realised positions meet its class,
and the realised positions are recorded.

| # | Class | Rounds |
| --- | --- | --- |
| 1 | every insert ends before position 1,024 | 2 |
| 2 | an insert straddles 1,024 (the sliding window) | 2 |
| 3 | inserts between 1,024 and 1,536 | 3 |
| 4 | an insert straddles 1,536 (the ring wraps) | 2 |
| 5 | an insert straddles 2,048 (DFlash2's ring) | 2 |
| 6 | an insert straddles 4,096 (a prompt chunk's rows) | 2 |
| 7 | 3 results of 400+ tokens: over 1,024 forced tokens in a row of rounds | 3 |
| 8 | short results: inserts of at most 8 tokens, and of 8k and 8k±1 tokens (the 8-row verify window) | 4 |
| 9 | two calls in one segment (two tool turns in one insert) | 2 |
| 10 | 6 rounds from position 3,000 on | 6 |

**State sums.** The engine has no state dump, so a test-only patch
(`bench/lane-loop-integrity/engine/`, not in `patches/`) adds `TF_INKLING2_STATE_SUMS=<file>`: after each
forward it writes, per position and per visit, a hash of the K row, the V row and the taps, the conv states at the
round's end, each drawn row's logits, and at a stream's end a hash of the whole snapshot (`mini_snap.each`'s bytes)
and of the drafter's rings. Unset, the patched image must give the shipped image's exact tokens on every case, and
set it must give the same tokens as unset. Sums are compared by position between A and C (all 48 visits, grouped by
pass), and A against B where B's rows are its own.

**Gates.**

- G1 (X): for every case, sampler and round, the tokens after each resume are equal in A, B, C and D, and every A
  variant equals A. No mismatch allowed.
- G2 (X, patched image): every K, V, conv, tap and logit sum at a common position is equal in A and C, on all 48
  visits; the end snapshot and the drafter rings are equal.
- G3 (S): every A variant equals A, exactly. A against B and C is reported; an inequality passes only if N shows
  the same inequality with no lane in the path and S0 is equal, which puts it on the tensor-core prefill, a
  documented lossy default that every multi-turn conversation already crosses.
- A mismatch is localised to its round, position, visit and kind from the sums, then diagnosed to a root cause.

**Planted controls**, each must go red:

1. One token of an inserted result changed in the replay.
2. The held EOS dropped from the replay history.
3. A sampled replay with seed + 1.
4. Patched image, `TF_INKLING2_STATE_SUMS_PLANT=<visit>@<position>`: one bit of that visit's conv state flipped
   after that position. The comparison must name that position and visit as the first difference, with earlier
   visits clean.
5. One sum changed in a copy of a dump: the comparer must name exactly that position, visit and kind.

**Q1 verdict.** "The lane does not break the loop state" needs G1, G2 and G3 to pass and all five controls red.
Anything else is "it breaks", with the localised cause.

## Q2 method

**Static.** From `0020`–`0026` and `chat_template.jinja`: where the tool turn lands relative to the thinking
message, what the engine forces besides the tool turn, what `think_open` does after a resume, and whether the
runner can see Inkling's thinking. Each claim cites file and line and is then checked on token ids.

**Arms**, each its own boot from Doug's `docker run` lines (images and ids recorded): **P** no tools,
`COMPUTE_LANE=0`; **L** `tools: [{"type":"code_interpreter"}]`, `COMPUTE_LANE=1`; **T** the same request,
`COMPUTE_LANE=tool-full`.

**Request**, the compute-lane-inkling `c32k` pass: one user message, no system prompt, `max_tokens 32768`,
`deadline_s 300`, `temperature 1.0`, `top_p 0.95`, `top_k 20`, `reasoning_effort high`, seed 101, streamed, with
`return_token_ids`. One request at a time a Spark.

**Sets** (selection: ascending `sha256("20261126:" + id)`, first n):

| Set | Source | n | Grader |
| --- | --- | --- | --- |
| MATH-500 level 5 | the kit's `math500.jsonl` (134 rows) | 30 | normalised match, else blind review |
| AIME 2025 | `opencompass/AIME2025` @ `a6ad95f6` | 30 (all) | integer, exact |
| ZebraLogic | `WildEval/ZebraLogic` @ `0a473f5a`, `mc_mode`, puzzles of 4+ houses and 4+ attributes, one question a puzzle | 30 | the answer string, exact |
| GPQA diamond | the authors' `dataset.zip` (198 rows); question text is not committed | 30 | the option letter; options shuffled once a question |
| Control: arithmetic | `tools/lane_ab.py` tasks, seed 1, the 20 Doug ran | 20 | `ANSWER:` integer, exact |
| Control: compute-DL | the kit's `dataset-v1`, the first 4 slots of each of the 6 families | 24 | the kit's `grade.py` |

120 reasoning-heavy questions and 44 compute controls, 3 arms each: 492 replies. The question files, their
hashes, the order and the harness are committed as a freeze before the first measured request.

**Order.** Blocks of 12 questions; a block runs P, L and T on one Spark, the arm order rotating by block, so any
prefix of whole blocks is balanced. If a Spark has to go back, collection stops at a block boundary and only
complete triples are analysed. No stopping on outcomes.

**Thinking measures**, from the token ids, split on Inkling's message tokens: thinking tokens before the first
call; thinking tokens after each result; whether a thinking message opens after a result before the answer; calls
inside a thinking message against calls after it closed; for T, the cells the runner ran
(`runtime.compute_executed`); any forced token that is not the tool turn.

**Grading, blind.** A script extracts the final answer and writes rows under opaque ids, shuffled, with no arm.
Grades are committed, then joined to the arms. MATH answers the normaliser cannot match go to two reviewers who see
the problem, the official answer and the final answer only; a third settles a split. A reply cut by the deadline
or the length with no answer is wrong, and counted separately.

**Q2 verdicts.**

- *Closes thinking*: yes if the tool turn can only follow a closed thinking message, or the engine forces a close.
- *Cuts thinking short*: yes if, on the 120 reasoning-heavy questions, L is below P with exact McNemar p < 0.05, or
  L − P is −7.5 points or worse (9 of 120) whatever p is, reported then as a possible loss the sample cannot
  settle. Reported with L − P, T − P and T − L, their 95% intervals, each set alone, cuts and thinking tokens by
  arm. The controls must show the lane's gain (L above P) for the run to count as a working lane.
- **Fix.** If the lane closes or cuts thinking, the fix is a separate patch with its own on/off parity (off: the
  shipped tokens exactly), re-measured on the same frozen sets and seeds. The shipped defaults do not change
  without a report.

## Receipts

`bench/lane-loop-integrity/`: this plan, the freeze, the harness, raw rows (token ids, sums, engine logs, image
ids, `docker run` lines), the comparer's output, the grades, and a README that answers Q1 and Q2.
