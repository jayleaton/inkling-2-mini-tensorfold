"""Q2's analysis: thinking measures from token ids, blind grading, the paired statistics.

  q2_report.py packet --rows FILE[,FILE] --out DIR     final answers under opaque ids, shuffled, no arm; the key apart
  q2_report.py grade --out DIR --sets DIR[,DIR]        grades the packet; MATH answers it cannot match go to review.jsonl
  q2_report.py join --rows FILE[,FILE] --out DIR --marks FILE   joins grades and arms: summary.json, summary.md

Review verdicts are DIR/review-<name>.jsonl rows {"opaque", "correct"}: two reviewers, a third where they split.
"""
import argparse
import hashlib
import json
import os
import random
import re
import secrets
import statistics
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(REPO, "tools"), os.path.join(REPO, "bench", "compute-lane-inkling", "dsv41", "kit"),
                os.path.join(REPO, "bench", "compute-lane-inkling", "dsv41", "compute-dl")]
import bench  # noqa: E402
import grade as computedl  # noqa: E402
import lane_ab  # noqa: E402
import lli  # noqa: E402
import q2  # noqa: E402

REASONING = ["math5", "aime", "zebra", "gpqa"]
CONTROLS = ["arith", "computedl"]
ARMS = ["P", "L", "T"]
PAIRS = [("L", "P"), ("T", "P"), ("T", "L")]
SEED = 20261126


def read(paths):
    rows = {}
    for path in paths.split(","):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error"):
                rows[(r["set"], r["id"], r["arm"])] = r
    return rows


def lines(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def final(content):
    found = re.findall(r"ANSWER:[ \t]*([^\n]*)", content or "")
    return found[-1].strip().strip("`*$ ").strip() if found else None


def extract(name, content):
    if name == "computedl":
        value, _, reason = computedl.extract(content)
        return {"value": value, "reason": reason}
    if name == "arith":
        return {"value": lane_ab.answer_of(content)}
    return {"value": final(content)}


def judge(name, answer, q):
    value = answer.get("value")
    gold = q["gold"]
    if value is None:
        return "incorrect", "no answer"
    if name == "computedl":
        return ("correct" if computedl.valid_value(value, q["family"]) and value == gold else "incorrect"), "exact"
    if name == "arith":
        return ("correct" if str(value) == str(gold) else "incorrect"), "exact"
    if name == "aime":
        m = re.fullmatch(r"\\?(?:boxed\{)?\s*(-?\d+)\s*\}?\.?", value)
        return ("correct" if m and int(m.group(1)) == int(gold) else "incorrect"), "integer"
    if name == "gpqa":
        m = re.match(r"\(?([A-D])\b", value)
        return ("correct" if m and m.group(1) == gold else "incorrect"), "letter"
    if name == "zebra":
        clean = lambda s: re.sub(r"[\s`'\"*.]+", " ", s).strip().lower()  # noqa: E731
        return ("correct" if clean(value) == clean(gold) else "incorrect"), "string"
    got = bench.math_grade(gold, value)
    return ("review" if got["grade"] == "manual_review" else got["grade"]), got["reason"]


def packet(args):
    rows = read(args.rows)
    os.makedirs(args.out, exist_ok=True)
    salt = secrets.token_hex(16)
    out, key = [], []
    for (name, ident, arm), r in rows.items():
        opaque = hashlib.sha256(f"{salt}:{name}:{ident}:{arm}".encode()).hexdigest()[:16]
        out.append({"opaque": opaque, "set": name, "id": ident, "answer": extract(name, r.get("content"))})
        key.append({"opaque": opaque, "set": name, "id": ident, "arm": arm})
    random.Random(salt).shuffle(out)
    write(os.path.join(args.out, "packet.jsonl"), out)
    write(os.path.join(args.out, "key.jsonl"), sorted(key, key=lambda k: k["opaque"]))
    print(len(out), "answers")


def grade(args):
    sets = q2.load(args.sets.split(","))
    grades, review = [], []
    for p in lines(os.path.join(args.out, "packet.jsonl")):
        q = sets[(p["set"], p["id"])]
        verdict, how = judge(p["set"], p["answer"], q)
        grades.append({"opaque": p["opaque"], "grade": verdict, "how": how})
        if verdict == "review":
            review.append({"opaque": p["opaque"], "problem": q["problem"], "official": q["gold"], "final": p["answer"]["value"]})
    write(os.path.join(args.out, "grades.jsonl"), grades)
    write(os.path.join(args.out, "review.jsonl"), review)
    print(len(grades), "graded,", len(review), "to review")


def reviewed(out):
    votes = {}
    for name in sorted(os.listdir(out)):
        if re.fullmatch(r"review-\w+\.jsonl", name):
            for r in lines(os.path.join(out, name)):
                votes.setdefault(r["opaque"], []).append(bool(r["correct"]))
    return {k: sum(v) * 2 > len(v) for k, v in votes.items() if len(v) >= 2 and sum(v) * 2 != len(v)}


def measure(r, mark):
    tf = r.get("tensorfold") or {}
    ids = tf.get("token_ids") or []
    msgs = lli.messages(ids, mark)
    calls = [i for i, m in enumerate(msgs) if m["kind"] == "invoke"]
    first = calls[0] if calls else len(msgs)
    after, i = [], 0
    while i < len(msgs):
        if msgs[i]["kind"] != "tool":
            i += 1
            continue
        while i < len(msgs) and msgs[i]["kind"] == "tool":
            i += 1
        j = i
        while j < len(msgs) and msgs[j]["kind"] not in ("invoke", "tool"):
            j += 1
        span = msgs[i:j]
        after.append({"thinking": sum(m["n"] for m in span if m["kind"] == "thinking"),
                      "text": sum(m["n"] for m in span if m["kind"] == "text"),
                      "reopened": bool(span) and span[0]["kind"] == "thinking"})
    compute = tf.get("compute") or {}
    return {
        "tokens": len(ids), "thinking": sum(m["n"] for m in msgs if m["kind"] == "thinking"),
        "text": sum(m["n"] for m in msgs if m["kind"] == "text"),
        "before_first_call": sum(m["n"] for m in msgs[:first] if m["kind"] == "thinking"),
        "calls": len(calls), "results": len(after), "after": after,
        "calls_inside_thinking": sum(1 for i in calls if i and msgs[i - 1]["kind"] == "thinking" and not msgs[i - 1]["closed"]),
        "forced_other": (compute.get("inserted_tokens") or 0) - sum(m["span"] for m in msgs if m["kind"] == "tool"),
        "stray": sum(m["n"] for m in msgs if m["kind"] == "stray"),
        "runner_calls": compute.get("runner_calls") or 0,
        "runner_cells": sum((form or {}).get("owned") or 0 for form in (tf.get("runner") or {}).values() if isinstance(form, dict)),
    }


def mcnemar(a, b):
    """a against b on paired booleans: the difference, the discordant counts, the exact two-sided p, a 95% interval."""
    n = len(a)
    wins = sum(1 for x, y in zip(a, b) if x and not y)
    losses = sum(1 for x, y in zip(a, b) if y and not x)
    rng = random.Random(SEED)
    boots = sorted(sum(a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(10000))
    return {"n": n, "difference": (wins - losses) / n, "wins": wins, "losses": losses,
            "p": bench.binomial_exact(wins, losses), "ci95": [boots[249], boots[9749]]}


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def join(args):
    rows = read(args.rows)
    mark = json.load(open(args.marks))
    key = {k["opaque"]: (k["set"], k["id"], k["arm"]) for k in lines(os.path.join(args.out, "key.jsonl"))}
    votes = reviewed(args.out)
    correct, open_reviews = {}, 0
    for g in lines(os.path.join(args.out, "grades.jsonl")):
        verdict = g["grade"]
        if verdict == "review":
            if g["opaque"] not in votes:
                open_reviews += 1
                continue
            verdict = "correct" if votes[g["opaque"]] else "incorrect"
        correct[key[g["opaque"]]] = verdict == "correct"
    if open_reviews:
        raise SystemExit(f"{open_reviews} answers still wait for a settled review")
    triples = sorted({(s, i) for (s, i, a) in rows if all((s, i, arm) in correct for arm in ARMS)})
    measures = {k: measure(r, mark) for k, r in rows.items()}
    write(os.path.join(args.out, "measures.jsonl"),
          [{"set": s, "id": i, "arm": a, "correct": correct.get((s, i, a)), "finish": rows[(s, i, a)].get("finish_reason"),
            **measures[(s, i, a)]} for (s, i, a) in sorted(rows)])
    summary = {"triples": len(triples), "replies": len(rows), "groups": {}}
    groups = {name: [name] for name in REASONING + CONTROLS}
    groups.update({"reasoning-heavy": REASONING, "controls": CONTROLS})
    for label, names in groups.items():
        qs = [t for t in triples if t[0] in names]
        if not qs:
            continue
        g = {"n": len(qs), "arms": {}, "pairs": {}}
        for arm in ARMS:
            rs = [rows[(s, i, arm)] for s, i in qs]
            ms = [measures[(s, i, arm)] for s, i in qs]
            after = [a for m in ms for a in m["after"]]
            g["arms"][arm] = {
                "correct": sum(correct[(s, i, arm)] for s, i in qs),
                "accuracy": sum(correct[(s, i, arm)] for s, i in qs) / len(qs),
                "cut": sum(1 for r in rs if r.get("finish_reason") != "stop"),
                "median_seconds": median([r.get("seconds") for r in rs]),
                "median_tokens": median([m["tokens"] for m in ms]),
                "median_thinking": median([m["thinking"] for m in ms]),
                "mean_thinking": statistics.fmean(m["thinking"] for m in ms),
                "replies_with_calls": sum(1 for m in ms if m["calls"]),
                "calls": sum(m["calls"] for m in ms),
                "calls_inside_thinking": sum(m["calls_inside_thinking"] for m in ms),
                "median_thinking_before_first_call": median([m["before_first_call"] for m in ms if m["calls"]]),
                "results": len(after),
                "results_followed_by_thinking": sum(1 for a in after if a["reopened"]),
                "median_thinking_after_result": median([a["thinking"] for a in after]),
                "forced_other": sum(m["forced_other"] for m in ms),
                "runner_calls": sum(m["runner_calls"] for m in ms),
                "runner_cells": sum(m["runner_cells"] for m in ms),
            }
        for a, b in PAIRS:
            g["pairs"][f"{a}-{b}"] = mcnemar([correct[(s, i, a)] for s, i in qs], [correct[(s, i, b)] for s, i in qs])
        summary["groups"][label] = g
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    md = ["| Set | n | P | L | T | L − P (95%) | p | T − P | T − L | cut P/L/T | median thinking P/L/T |", "|" + " --- |" * 11]
    for label, g in summary["groups"].items():
        arms, lp = g["arms"], g["pairs"]["L-P"]
        md.append("| {} | {} | {} | {} | {} | {:+.1f} ({:+.1f}, {:+.1f}) | {:.3f} | {:+.1f} | {:+.1f} | {} | {} |".format(
            label, g["n"], *(f"{arms[a]['correct']} ({100 * arms[a]['accuracy']:.1f}%)" for a in ARMS),
            100 * lp["difference"], 100 * lp["ci95"][0], 100 * lp["ci95"][1], lp["p"],
            100 * g["pairs"]["T-P"]["difference"], 100 * g["pairs"]["T-L"]["difference"],
            "/".join(str(arms[a]["cut"]) for a in ARMS), "/".join(f"{arms[a]['median_thinking']:.0f}" for a in ARMS)))
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["packet", "grade", "join"])
    ap.add_argument("--rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=os.path.join(HERE, "sets"))
    ap.add_argument("--marks")
    args = ap.parse_args()
    {"packet": packet, "grade": grade, "join": join}[args.mode](args)


if __name__ == "__main__":
    main()
