#!/usr/bin/env python3
"""Compute-lane A/B: the same computation-heavy questions under one arm, scored against exact answers.

Usage: tools/lane_ab.py ARM [--tasks 20] [--seed 1] [--parallel 2] [--max-tokens 32768] [--deadline-s 300] [--out FILE.jsonl]
  (API_URL / PORT / MODEL as in client.py; thinking on, temperature 0.7, top_p 0.95, top_k 20; the budget is AB3's:
  32,768 tokens and a 300 s deadline a request, the same for every arm)
Arms, after the DeepSeek V4.1 kit's lane studies (bench/compute-lane):
  P  plain: no tools                                                       (any server)
  D  declared, not run: the built-in tool's exact declaration as a client function named python; each call is
     answered "code execution is disabled", up to 4 rounds                  (a server with the lane off)
  L  lane: tools [{"type": "code_interpreter"}]; the server runs the calls (COMPUTE_LANE=1)
  T  tool-full: as L on a COMPUTE_LANE=tool-full server (the in-model runner on too)
The questions are generated from --seed (five families: modular powers, long products, factorial digit sums,
Collatz step counts, prime counts), each with its exact answer computed here. A reply is correct when its last
"ANSWER: <integer>" line equals the answer. Prints one JSON line a question (to --out too) and a summary line:
correct / total, replies cut (at max_tokens or the deadline: finish "length", or no answer with a cancelled
finish), completion tokens, seconds a question, calls (client rounds for D; compute_executed for L and T).
"""
import argparse
import json
import math
import random
import re
import sys
import threading
import time

sys.dont_write_bytecode = True
import client  # noqa: E402

PYTHON_TOOL = [{"type": "function", "function": {
    "name": "python",
    "description": "Run Python 3 code in a sandbox with no network, no files and no installs, and get its stdout and "
                   "stderr. print() what you need; the value of a final bare expression is printed too.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python 3 source to run"}},
                   "required": ["code"]}}}]
ASK = "\n\nEnd your reply with the final answer as a bare integer on its own last line, written as `ANSWER: <integer>`."
DISABLED = "Error: code execution is disabled in this session. Work the answer out yourself."


def primes_below(n: int) -> int:
    sieve = bytearray([1]) * n
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n, i)))
    return sum(sieve)


def collatz(n: int) -> int:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def tasks(count: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        kind = i % 5
        if kind == 0:
            a, b, m = rng.randrange(10 ** 5, 10 ** 6), rng.randrange(10 ** 4, 10 ** 5), rng.randrange(10 ** 8, 10 ** 9)
            q, ans = f"Compute {a}^{b} mod {m}.", pow(a, b, m)
        elif kind == 1:
            a, b = rng.randrange(10 ** 11, 10 ** 12), rng.randrange(10 ** 11, 10 ** 12)
            q, ans = f"Compute the exact product {a} × {b}.", a * b
        elif kind == 2:
            n = rng.randrange(60, 120)
            q, ans = f"What is the sum of the decimal digits of {n}! ({n} factorial)?", sum(map(int, str(math.factorial(n))))
        elif kind == 3:
            n = rng.randrange(10 ** 6, 10 ** 7)
            q, ans = (f"Starting from {n}, repeatedly replace n by n/2 when n is even and by 3n+1 when n is odd. "
                      f"How many steps does it take to reach 1?"), collatz(n)
        else:
            n = rng.randrange(50_000, 300_000)
            q, ans = f"How many prime numbers are there below {n}?", primes_below(n)
        out.append({"id": f"{seed}-{i:03d}", "kind": ["modpow", "product", "digitsum", "collatz", "primes"][kind],
                    "question": q, "answer": ans})
    return out


def answer_of(text: str):
    found = re.findall(r"ANSWER:\s*`?\s*(-?[\d,]+)", text or "")
    return int(found[-1].replace(",", "")) if found else None


def executed(r: client.Reply) -> int:
    """Calls the lane ran (the reply's tensorfold.compute.calls) plus runner cells it wrote (tensorfold.runner.<form>.owned);
    compute_executed.total instead when the server reports it (the compute default)."""
    rt = r.tensorfold if isinstance(r.tensorfold, dict) else {}
    ce = rt.get("compute_executed")
    if isinstance(ce, dict) and isinstance(ce.get("total"), int):
        return ce["total"]
    calls = (rt.get("compute") or {}).get("calls") or []
    cells = 0
    for form in ("fence", "repl", "inline", "trace"):
        owned = ((rt.get("runner") or {}).get(form) or {}).get("owned")
        cells += owned if isinstance(owned, int) else 0
    return len(calls) + cells


def run_one(task: dict, arm: str, max_tokens: int, seed: int, deadline_s: int) -> dict:
    messages = [{"role": "user", "content": task["question"] + ASK}]
    extra = {"top_p": 0.95, "top_k": 20, "seed": seed, "deadline_s": deadline_s}
    if arm == "D":
        extra["tools"] = PYTHON_TOOL
    elif arm in ("L", "T"):
        extra["tools"] = [{"type": "code_interpreter"}]
    tokens, seconds, calls, text, finish, err, cut = 0, 0.0, 0, "", None, None, False
    try:
        for _ in range(5 if arm == "D" else 1):
            r = client.chat(messages, max_tokens=max_tokens, thinking=True, temperature=0.7, **extra)
            tokens += r.usage.get("completion_tokens") or 0
            seconds += r.seconds
            text, finish = r.content, r.finish_reason
            cut = cut or finish not in ("stop", "tool_calls")
            calls += executed(r)
            if arm != "D" or not r.tool_calls or calls >= 4:
                break
            messages.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                {"id": c["id"] or f"call_{n}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for n, c in enumerate(r.tool_calls)]})
            for n, c in enumerate(r.tool_calls):
                messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{n}", "name": c["name"], "content": DISABLED})
            calls += len(r.tool_calls)
    except Exception as e:  # noqa: BLE001 - recorded per question
        err = str(e)[:300]
    got = answer_of(text)
    return {"id": task["id"], "kind": task["kind"], "arm": arm, "correct": got == task["answer"], "got": got,
            "answer": task["answer"], "completion_tokens": tokens, "seconds": round(seconds, 2), "calls": calls,
            "finish": finish, "cut": cut and got is None, "error": err}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", choices=["P", "D", "L", "T"])
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--deadline-s", type=int, default=300)
    p.add_argument("--out")
    a = p.parse_args()
    todo = tasks(a.tasks, a.seed)
    results, lock = [], threading.Lock()
    out = open(a.out, "a") if a.out else None
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                task = todo.pop(0)
            row = run_one(task, a.arm, a.max_tokens, a.seed, a.deadline_s)
            with lock:
                results.append(row)
                line = json.dumps(row)
                print(line, flush=True)
                if out:
                    out.write(line + "\n")
                    out.flush()

    threads = [threading.Thread(target=worker) for _ in range(max(1, a.parallel))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = {"arm": a.arm, "seed": a.seed, "correct": sum(r["correct"] for r in results), "total": len(results),
               "cut": sum(1 for r in results if r["cut"]), "errors": sum(1 for r in results if r["error"]),
               "finishes": {f: sum(1 for r in results if str(r["finish"]) == f) for f in sorted({str(r["finish"]) for r in results})},
               "seconds_per_question": round(sum(r["seconds"] for r in results) / max(1, len(results)), 1),
               "completion_tokens": sum(r["completion_tokens"] for r in results),
               "reply_seconds": round(sum(r["seconds"] for r in results), 1), "wall_seconds": round(time.time() - t0, 1),
               "calls": sum(r["calls"] for r in results)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    if out:
        out.write(json.dumps({"summary": summary}) + "\n")
    return 0


if __name__ == "__main__":
    client.run(main)
