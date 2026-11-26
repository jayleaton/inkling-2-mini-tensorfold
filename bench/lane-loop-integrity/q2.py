"""Q2, thinking continuity: collects one arm's replies for a block of the frozen sets against the running engine.

  q2.py order                                   prints the frozen collection order: block, position, set, id
  q2.py smoke --arm P|L|T                       one toy request, not a frozen question: checks the client reads the stream
  q2.py run --arm P|L|T --block N --sets DIR[,DIR] --out FILE --spark NAME

The request is the compute-lane-inkling c32k pass: one user message, max_tokens 32768, deadline_s 300, temperature 1.0,
top_p 0.95, top_k 20, reasoning_effort high, seed 101, streamed, with return_token_ids. One request at a time.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lli  # noqa: E402

API = os.environ.get("API_URL", "http://127.0.0.1:8888")
HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["math5", "aime", "zebra", "gpqa", "arith", "computedl"]
BLOCK = 12
SEED = 101
WAIT_S = 390


def load(folders):
    rows = {}
    for name in SETS:
        for folder in folders:
            path = os.path.join(folder, f"{name}.jsonl")
            if os.path.exists(path):
                for line in open(path, encoding="utf-8"):
                    r = json.loads(line)
                    rows[(r["set"], r["id"])] = r
                break
        else:
            raise SystemExit(f"{name}.jsonl is in none of {folders}")
    return rows


def order():
    """Every frozen question in collection order, from the lock alone: ascending sha256 of "20261126:order:set:id"."""
    lock = json.load(open(os.path.join(HERE, "sets.lock.json")))
    keys = [(name, ident) for name in SETS for ident in lock["sets"][name]["ids"]]
    keys.sort(key=lambda k: hashlib.sha256(f"{lock['seed']}:order:{k[0]}:{k[1]}".encode()).hexdigest())
    return [keys[i:i + BLOCK] for i in range(0, len(keys), BLOCK)], lock


def body_of(prompt, arm):
    body = {"model": "Inkling-2-Mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 32768,
            "deadline_s": 300, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": SEED,
            "chat_template_kwargs": {"reasoning_effort": "high"}, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True}
    if arm in ("L", "T"):
        body["tools"] = [{"type": "code_interpreter"}]
    return body


def ask(body):
    request = urllib.request.Request(API + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = {"content": "", "reasoning": "", "finish_reason": None, "usage": None, "tensorfold": None, "speculative": None,
           "tool_calls": [], "compute_deltas": 0, "first_s": None}
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=WAIT_S) as reply:
        for raw in reply:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                out["usage"] = chunk["usage"]
            for key in ("tensorfold", "speculative"):
                if chunk.get(key) and (key != "tensorfold" or "token_sha256" in chunk[key]):
                    out[key] = chunk[key]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if out["first_s"] is None and (delta.get("content") or delta.get("reasoning_content")):
                    out["first_s"] = time.monotonic() - start
                out["content"] += delta.get("content") or ""
                out["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
                out["tool_calls"] += delta.get("tool_calls") or []
                out["compute_deltas"] += 1 if delta.get("compute") else 0
                if choice.get("finish_reason"):
                    out["finish_reason"] = choice["finish_reason"]
    out["seconds"] = time.monotonic() - start
    return out


def done(path):
    seen = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error"):
                seen.add((r["set"], r["id"], r["arm"]))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["order", "run", "smoke"])
    ap.add_argument("--arm", choices=["P", "L", "T"])
    ap.add_argument("--block", type=int)
    ap.add_argument("--sets", default=os.path.join(HERE, "sets"))
    ap.add_argument("--out")
    ap.add_argument("--spark", default="")
    args = ap.parse_args()
    blocks, lock = order()
    if args.mode == "order":
        for b, keys in enumerate(blocks):
            for i, (name, ident) in enumerate(keys):
                print(b, i, name, ident)
        return
    if args.mode == "smoke":
        got = ask(body_of("What is 123456789 * 987654321? End with `ANSWER: <integer>`.", args.arm))
        tf = got["tensorfold"] or {}
        print(json.dumps({"finish": got["finish_reason"], "usage": got["usage"], "content": got["content"][-120:],
                          "reasoning_chars": len(got["reasoning"]), "token_ids": len(tf.get("token_ids") or []),
                          "compute": tf.get("compute"), "tensorfold_keys": sorted(tf), "seconds": round(got["seconds"], 1)}))
        return
    rows = load(args.sets.split(","))
    for name in SETS:
        got = [hashlib.sha256(rows[(name, i)]["prompt"].encode()).hexdigest() for i in lock["sets"][name]["ids"]]
        if got != lock["sets"][name]["prompt_sha256"]:
            raise SystemExit(f"{name}: the prompts are not the frozen ones")
    marks = os.path.join(os.path.dirname(os.path.abspath(args.out)), "marks.json")
    if not os.path.exists(marks):
        with open(marks, "w") as f:
            json.dump(lli.Marks().id, f, indent=1)
    seen = done(args.out)
    failed = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for name, ident in blocks[args.block]:
            if (name, ident, args.arm) in seen:
                continue
            q = rows[(name, ident)]
            row = {"set": name, "id": ident, "arm": args.arm, "seed": SEED, "block": args.block, "spark": args.spark,
                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            try:
                row.update(ask(body_of(q["prompt"], args.arm)))
            except Exception as error:  # the row records the failure; a rerun asks again
                row["error"] = f"{type(error).__name__}: {error}"[:500]
                failed += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            usage = row.get("usage") or {}
            print(args.arm, args.block, name, ident, row.get("finish_reason"), usage.get("completion_tokens"),
                  round(row.get("seconds", 0), 1), row.get("error", ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
