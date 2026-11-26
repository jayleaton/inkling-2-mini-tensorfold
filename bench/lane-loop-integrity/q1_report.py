"""Q1's tallies: every phase folder's rows.jsonl as one table (checks by arm, the planted controls).

  q1_report.py DIR [PHASE...]      DIR holds the phase folders (X, S, S0, ...); prints markdown, writes DIR/summary.json

A reply cut by max_tokens that the replay continues past is "length only": every token the two share is equal.
"""
import collections
import json
import os
import sys

ARMS = ["A.warm", "A.nodraft", "A.with-lane", "A.with-lane.peer", "A.with-plain", "A.with-plain.peer",
        "B", "C", "D", "D.second-lane-turn", "N.route", "N.cold", "N.turn2", "R"]


def arm(check):
    if check.startswith("D.second-lane-turn"):
        return "D.second-lane-turn"
    return check.split(".")[0] if check[0] in "BCD" else check


def tally(path):
    counts = collections.defaultdict(collections.Counter)
    red = collections.defaultdict(collections.Counter)
    for line in open(path):
        r = json.loads(line)
        check = r["check"]
        if check.startswith("control."):
            red[check]["red" if r.get("red") else "not red"] += 1
            continue
        if check.endswith((".render", ".cached", ".states")) or check in ("A", "N"):
            counts[check]["ok" if r["ok"] else "failed"] += 1
            continue
        if r.get("skipped"):
            continue
        length_only = not r["ok"] and r.get("first_difference") is not None and r["first_difference"] == r.get("want")
        counts[arm(check)]["equal" if r["ok"] else "length only" if length_only else "different"] += 1
    return {"checks": {k: dict(v) for k, v in counts.items()}, "controls": {k: dict(v) for k, v in red.items()}}


def main():
    root = sys.argv[1]
    phases = sys.argv[2:] or sorted(p for p in os.listdir(root) if os.path.exists(os.path.join(root, p, "rows.jsonl")))
    out = {p: tally(os.path.join(root, p, "rows.jsonl")) for p in phases}
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print("| Check | " + " | ".join(phases) + " |")
    print("|" + " --- |" * (len(phases) + 1))
    for name in ARMS:
        cells = []
        for p in phases:
            c = out[p]["checks"].get(name)
            if not c:
                cells.append("")
                continue
            n = sum(c.values())
            cell = f"{c.get('equal', 0)}/{n}"
            if c.get("length only"):
                cell += f" (+{c['length only']} length only)"
            cells.append(cell)
        if any(cells):
            print(f"| {name} equal | " + " | ".join(cells) + " |")
    for name in sorted({k for p in phases for k in out[p]["controls"]}):
        cells = []
        for p in phases:
            c = out[p]["controls"].get(name)
            cells.append(f"{c.get('red', 0)}/{sum(c.values())}" if c else "")
        print(f"| {name} red | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
