# Q1 (state integrity): where it stands

Written 2026-11-26 about 19:15Z, when JJ stopped Q1 on this engine; the queued phases that had finished unattended
by then were added 2026-11-27 01:30Z from their receipts. Q1 has **no verdict**. It will be redone on the
new recipe's engine (upstream main + `jayleaton/inkling-2-mini-tensorfold-recipe`); nothing below should be read as
one. Engine tested: Mia's dc2ca8a1 + patches 0001–0026, image `tensorfold-inkling2:c4-dc2ca8a-runner`
(dgx-01 `sha256:2a5409bc…`).

## Measured

Token checks, `q1.py` on dgx-01, DFlash2 drafter, 10 position classes × 3 samplers. A cell is "replies whose tokens
equal the lane reply's".

| Check | X (exact) | S (shipped) | S0 (shipped, exact prefill) |
| --- | --- | --- | --- |
| Lane again from its kept state | 30/30 | 30/30 | 30/30 |
| Lane, `"draft": false` | 30/30 | 30/30 | 30/30 |
| Lane beside a concurrent lane stream (and the peer's own reply) | 30/30 | 30/30 | not run |
| Lane beside a concurrent plain stream (and the peer's own reply) | 30/30 | 30/30 | not run |
| Client round trip, a round | 102/117 | 93/117 | 96/117 |
| Cold prefill, a round | 132/147 | 123/147 | 126/147 |
| No lane: a plain reply prefilled cold from a cut | 0/48 (1 differs in length only) | 0/48 | 0/48 (6 in length only) |

Other boots, cases × greedy and sampled 0.7: drafter off, X 20/20 warm and 20/20 `"draft": false`, round trip 68/78,
cold 88/98; drafter off, S 20/20 and 20/20, 64/78, 84/98; DSpark, X (cases 1, 4, 7, 10) 8/8 and 8/8, 31/34, 39/42.

Disk resume (`X-disk`, `S-disk`: cases 2 and 7, greedy and sampled 0.7; the harness waits for the state files, 2–3 a
conversation, before the restart): every round after the restart reused the NVMe state (602–1,466 cached tokens),
and a second lane turn on top of a disk-resumed lane turn gave the tokens of a fresh run in 4 of 4 under X and 4 of
4 under S. Rounds against the lane's tokens: 12 of 14 under each, the 2 misses of the same kind as the round trip's.

State sums (`engine/0001-state-sums.patch`, image `…-runner-lli` `sha256:bbdf9f0d…`), exact kernels, 20 lane replies
(greedy and sampled 0.7), `q1_sums.py pairs`:

- Lane against `"draft": false`: every sum equal in 20 of 20 (1,555,064 K and as many V row sums over all 48
  visits, 161,392 tap sums, 8,297 logit sums, 277,488 conv sums a site, the end snapshots).
- Lane against the lane again from its kept state: every sum equal in 20 of 20, end snapshots included.
- Lane against a cold prefill of the same history: every prompt position equal; every model-written and every
  inserted position unequal, first at visit 1 in 70 of 78 pairs and at visit 5 in 8.
- No decode row ran twice in a lane stream.

Decode against prompt pass with no lane, the patch's author's check on one plain reply, 200 positions: 0 bit-equal
final-norm taps under X and under S (mean |Δ| 0.168 and 0.415 against a mean |value| of about 3.2). The same reply
run through decode rows only (`TF_INKLING2_STATE_SUMS_DECODE_PROMPT`) is bit-equal to the decoded reply on K and V
of all 48 visits, taps and logits, at any window width, under X and S.

## What that supports, and what it does not

- The plan's gate G1 (lane = round trip = cold prefill under exact kernels) **fails as written**.
- The failure is not specific to the lane: a plain reply shows it at every cut. In this engine a row computed by the
  prompt pass and the same row computed by decode are not bit-identical, in any of the three kernel settings. Read
  from the code, not isolated by a test: `mini_visit.zig:318-335` runs prompt rows through `gx.prompt` with a bf16
  expert output and decode rows through `gx.decode` with an fp32 one. S0 mismatching like S also shows the plan's
  G3 assumption (tensor-core prefill explains S) was wrong.
- The lane is self-consistent across every decode-path variant. It is **not yet checked against a lane-free
  computation of the same rows**, which is the check that would clear it.

## Pending (not run, or run and invalid)

1. **Lane against the decode-rows reference.** First attempt (`X-ref`) is invalid: it ran the prompt through decode
   rows too, so it differs from position 0 (my specification error). The corrected reference
   (`TF_INKLING2_STATE_SUMS_DECODE_FROM`, phases `X-decode`/`S-decode`, image `…-runner-lli2`
   `sha256:5d4475ee…`) was validated by its author on one plain reply (935 of 935 positions equal under X, end
   snapshot and drafter rings equal) and never run on the lane cases.
2. **Planted conv fault.** `X-plant` with the first patch fired and changed no sum and no token: not red. The
   reworked plant (the visit's whole attention conv state, one bit) went red in its author's test at
   (position 305, visit 21); my rerun did not happen.
3. **Switch-off parity of `-lli2`** against the shipped image, and its end-snapshot comparison under S.
4. **`S-sums` is collected and not compared** (`q1_sums.py pairs` was never run on it); `S-ref` is invalid like
   `X-ref`.
5. **The first disk-resume run under X resumed nothing** (0 cached tokens after the restart, no state files): the
   harness restarted before the idle engine had written them. The rerun above waits for them. That first run's 4
   of 4 is therefore not a disk result.
6. **Token-level planted controls are weak**: changed result token red 28/30 (X) and 29/30 (S), dropped EOS 17/30
   and 20/30, seed + 1 15/20 and 16/20; with DSpark, seed + 1 was red in 0 of 4.

## For the redo

- Prefill is not a bit-exact reference for decode rows on this engine. Either the new engine makes them equal, and
  the frozen G1 applies as written, or the reference has to be a decode-rows replay as in item 1.
- Receipts are on dgx-01 under `/home/dgx01/work/lli/q1/<phase>/` (`rows.jsonl`, `ids/`, `log`, engine logs,
  `sums/`). Everything but `sums/` and `sessions/` (9 MiB) is also copied off; `X-sums/sums/sums.jsonl` is 823 MiB
  and stays there. They are not committed.
- Harness: `boot.sh`, `q1.py`, `q1_all.sh`, `q1_sums.py`, `q1_report.py`, `engine/`.
