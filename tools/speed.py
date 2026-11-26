#!/usr/bin/env python3
"""Decode and prefill speed against the running server, measured at the client so any engine can be compared.

Usage: tools/speed.py [--streams 1,2] [--tokens 512] [--reps 3] [--sizes 8k,32k,128k] [--decode-only | --prefill-only]
       (API_URL / PORT / MODEL as in client.py)
Decode: prose and code prompts, N streams at once, in two modes: `sampled` (temperature 0.7, top_k 20, thinking on) and
  `greedy` (temperature 0, thinking off). A stream's rate is (completion tokens - 1) / (wall - time to first token);
  the line gives the sum over streams and the per-stream mean, median of --reps.
Prefill: fresh generated prose of each size (a new seed each run, so nothing past the template's opening comes from
  the prompt cache), one streamed request with max_tokens 1, thinking off: prompt tokens / time to first token, and
  the server's own prefill seconds when it reports them. Median of --reps.
"""
import argparse
import statistics
import sys
import threading
import time

sys.dont_write_bytecode = True
import client  # noqa: E402

PROSE = "Explain why the sky is blue in a few paragraphs."
CODE = "Write a Python quicksort with a docstring and unit tests."
MODES = {"sampled": dict(thinking=True, temperature=0.7, top_k=20, top_p=0.95),
         "greedy": dict(thinking=False, temperature=0.0)}


def one(prompt, tokens, mode, seed):
    kw = dict(MODES[mode])
    r = client.chat([{"role": "user", "content": prompt}], max_tokens=tokens, stream=True, seed=seed, **kw)
    n = r.usage.get("completion_tokens") or 0
    span = r.seconds - (r.first_token_s or 0)
    return {"tokens": n, "tps": (n - 1) / span if n > 1 and span > 0 else 0.0, "ttft": r.first_token_s or 0.0}


def decode(prompt, streams, tokens, mode, seed):
    out = [None] * streams

    def run(i):
        out[i] = one(prompt, tokens, mode, seed + i)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    per = [o["tps"] for o in out]
    return sum(per), statistics.mean(per), statistics.median(o["ttft"] for o in out), min(o["tokens"] for o in out)


def prefill(size, seed, cpt):
    text = " ".join(client.filler_units(client.sentence, size, seed, cpt))
    r = client.chat([{"role": "user", "content": text + "\n\nSummarize the text above in one sentence."}],
                    max_tokens=1, stream=True, thinking=False, temperature=0.0)
    p = r.prompt_tokens or 0
    cached = r.cached_tokens or 0
    return {"prompt": p, "cached": cached, "ttft": r.first_token_s or r.seconds, "server_s": r.prefill_seconds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", default="1,2")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sizes", default="8k,32k,128k")
    ap.add_argument("--modes", default="sampled,greedy")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--prefill-only", action="store_true")
    a = ap.parse_args()
    seed = int(time.time()) % 100000
    if not a.prefill_only:
        for mode in a.modes.split(","):
            for name, prompt in (("prose", PROSE), ("code", CODE)):
                decode(prompt, 1, 16, mode, seed)  # warm up
                for s in map(int, a.streams.split(",")):
                    rs = [decode(prompt, s, a.tokens, mode, seed + 100 * k) for k in range(a.reps)]
                    print(f"decode {mode:7s} {name:5s} x{s}: {statistics.median(r[0] for r in rs):6.1f} tok/s total, "
                          f"{statistics.median(r[1] for r in rs):5.1f} per stream, TTFT {statistics.median(r[2] for r in rs) * 1000:5.0f} ms, "
                          f">= {min(r[3] for r in rs)} tok (median of {a.reps})", flush=True)
    if not a.decode_only:
        cpt, _ = client.calibrate(client.sentence, 7)
        for size in map(client.parse_size, a.sizes.split(",")):
            rs = sorted((prefill(size, seed + size + 7919 * k, cpt) for k in range(a.reps)), key=lambda r: r["ttft"])
            r = rs[len(rs) // 2]
            fresh = r["prompt"] - r["cached"]
            srv = f", server {fresh / r['server_s']:6.0f} tok/s ({r['server_s']:.2f} s)" if r["server_s"] else ""
            print(f"prefill {r['prompt']:>7,} tok ({r['cached']} cached): TTFT {r['ttft']:7.2f} s, "
                  f"{fresh / r['ttft']:6.0f} tok/s{srv} (median of {a.reps})", flush=True)
    return 0


if __name__ == "__main__":
    client.run(main)
