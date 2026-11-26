"""Q2's frozen question sets: selects by the plan's rule, writes sets/<name>.jsonl and sets.lock.json.

  q2_sets.py --data DIR --private DIR

DIR holds the downloads at their pinned revisions: aime/aime2025-{I,II}.jsonl (opencompass/AIME2025 @ a6ad95f6),
zebra/mc.parquet (WildEval/ZebraLogic @ 0a473f5a, mc_mode; needs pyarrow), gpqa/gpqa_diamond.csv (the authors'
dataset.zip). GPQA's questions go to --private and are never committed: the lock keeps their ids and hashes.
Each row: {"id", "set", "prompt", "gold", ...}. The order in a file is the order of collection.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SEED = "20261126"
KIT = os.path.join(REPO, "bench", "compute-lane-inkling", "dsv41")
END = "\n\nEnd your reply with the final answer on its own last line, written as `ANSWER: {what}`."


def key(ident):
    return hashlib.sha256(f"{SEED}:{ident}".encode()).hexdigest()


def first(rows, n, ident):
    return sorted(rows, key=lambda r: key(ident(r)))[:n]


def math5():
    rows = [json.loads(line) for line in open(os.path.join(KIT, "kit", "sources", "math500.jsonl"))]
    rows = [r for r in rows if r["level"] == 5]
    assert len(rows) == 134, len(rows)
    return [{"id": r["unique_id"], "set": "math5", "prompt": r["problem"] + END.format(what="<answer>"), "gold": r["answer"],
             "problem": r["problem"], "subject": r["subject"]} for r in first(rows, 30, lambda r: r["unique_id"])]


def aime(data):
    rows = []
    for part in ("I", "II"):
        for i, line in enumerate(open(os.path.join(data, "aime", f"aime2025-{part}.jsonl")), 1):
            r = json.loads(line)
            rows.append({"id": f"aime2025-{part}-{i:02d}", "set": "aime", "gold": str(r["answer"]).strip(),
                         "prompt": r["question"] + END.format(what="<integer>")})
    assert len(rows) == 30, len(rows)
    return first(rows, 30, lambda r: r["id"])


def zebra(data):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(data, "zebra", "mc.parquet")).to_pylist()
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: key(r["id"])):
        m = re.fullmatch(r"lgp-test-(\d+)x(\d+)-(\d+)#mc-(\d+)", r["id"])
        puzzle = r["id"].split("#")[0]
        if not m or int(m.group(1)) < 4 or int(m.group(2)) < 4 or puzzle in seen:
            continue
        seen.add(puzzle)
        choices = list(r["choices"])
        text = (f"{r['puzzle']}\n\nQuestion: {r['question']}\nChoices: " + "; ".join(choices) +
                END.format(what="<one of the choices>"))
        out.append({"id": r["id"], "set": "zebra", "prompt": text, "gold": r["answer"], "choices": choices})
        if len(out) == 30:
            break
    assert len(out) == 30
    return out


def gpqa(data):
    rows = list(csv.DictReader(open(os.path.join(data, "gpqa", "gpqa_diamond.csv"), newline="", encoding="utf-8")))
    assert len(rows) == 198, len(rows)
    out = []
    for r in first(rows, 30, lambda r: r["Record ID"]):
        options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(key(r["Record ID"])).shuffle(order)
        letters = "ABCD"
        text = (r["Question"].strip() + "\n\n" + "\n".join(f"{letters[i]}. {options[j].strip()}" for i, j in enumerate(order)) +
                END.format(what="<A, B, C or D>"))
        out.append({"id": r["Record ID"], "set": "gpqa", "prompt": text, "gold": letters[order.index(0)]})
    return out


def arith():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    sys.dont_write_bytecode = True
    import lane_ab
    return [{"id": t["id"], "set": "arith", "prompt": t["question"] + lane_ab.ASK, "gold": str(t["answer"]), "kind": t["kind"]}
            for t in lane_ab.tasks(20, 1)]


def computedl():
    tasks = [json.loads(line) for line in open(os.path.join(KIT, "compute-dl", "dataset-v1", "tasks-public.jsonl"))]
    answers = json.load(open(os.path.join(KIT, "compute-dl", "dataset-v1", "answers-a.json")))
    return [{"id": t["id"], "set": "computedl", "prompt": t["prompt"], "gold": answers[t["id"]], "family": t["family"]}
            for t in tasks if t["slot"] < 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--private", required=True)
    args = ap.parse_args()
    sets = {"math5": math5(), "aime": aime(args.data), "zebra": zebra(args.data), "gpqa": gpqa(args.data),
            "arith": arith(), "computedl": computedl()}
    want = {"math5": 30, "aime": 30, "zebra": 30, "gpqa": 30, "arith": 20, "computedl": 24}
    lock = {"seed": SEED, "sets": {}}
    os.makedirs(os.path.join(HERE, "sets"), exist_ok=True)
    os.makedirs(args.private, exist_ok=True)
    for name, rows in sets.items():
        assert len(rows) == want[name], (name, len(rows))
        folder = args.private if name == "gpqa" else os.path.join(HERE, "sets")
        body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        with open(os.path.join(folder, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            f.write(body)
        lock["sets"][name] = {
            "n": len(rows), "sha256": hashlib.sha256(body.encode()).hexdigest(), "committed": name != "gpqa",
            "ids": [r["id"] for r in rows],
            "prompt_sha256": [hashlib.sha256(r["prompt"].encode()).hexdigest() for r in rows]}
    with open(os.path.join(HERE, "sets.lock.json"), "w") as f:
        json.dump(lock, f, indent=1, sort_keys=True)
        f.write("\n")
    print({name: len(rows) for name, rows in sets.items()})


if __name__ == "__main__":
    main()
