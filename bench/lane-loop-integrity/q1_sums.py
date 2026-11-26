"""Q1's state sums comparer: one request's stream against another's, by position, visit and kind.

  q1_sums.py pairs --dir PHASE [--ref PHASE] --out FILE    every case: A against its variants, its cold replays, and the
                                                         reference boot's replay of the same history (check R)
  q1_sums.py plant --dir PHASE --clean PHASE    the planted conv bit: the sums must first differ at the row the flip
                                                preceded, on the visit after the planted one (a visit writes its K and V
                                                before its attention conv reads the state), with every earlier visit equal
  q1_sums.py selftest                                    one sum changed in a copy must be named exactly

A PHASE is a q1.py --out folder booted with SUMS: rows.jsonl (each row's byte span of sums/sums.jsonl) and ids/.
Sums come from TF_INKLING2_STATE_SUMS (engine/0001-state-sums.patch): `rows` events hold, a row and a visit, the K and
the V row's hash, the taps' hashes and the drawn logits' hash, then the conv states after the forward; `keep` is a cut
window's rollback; `snap` the kept state and the drafter's rings at a prompt pass's end and at release.
"""
import argparse
import collections
import copy
import json
import os
import sys

sys.dont_write_bytecode = True
HEX = 16
ZERO = "0" * HEX
SITES = ["k", "v", "attn", "mlp"]
ORDER = {"k": 0, "v": 1, "conv.k": 2, "conv.v": 3, "conv.attn": 4, "conv.mlp": 5, "tap": 6, "logit": 7}
PASSES = [(0, 5, "before"), (6, 17, "pass 1"), (18, 29, "pass 2"), (30, 41, "pass 3"), (42, 47, "after")]
TAPS = [5, 23, 41, 48]


def split(text):
    return [text[i:i + HEX] for i in range(0, len(text), HEX)]


class Stream:
    def __init__(self):
        self.plen = None
        self.resume = None
        self.pos = {}
        self.conv = {}
        self.snaps = {}
        self.plant = None
        self.reruns = []

    def feed(self, e):
        ev = e["ev"]
        if ev == "open":
            self.plen, self.resume = e["plen"], e["resume"]
        elif ev == "rows":
            for r in range(e["n"]):
                row = {"kind": e["kind"], "k": split(e["k"][r]), "v": split(e["v"][r])}
                if e.get("tap"):
                    row["tap"] = split(e["tap"][r])
                logits = e.get("logit") or []
                if e["kind"] == "decode" and r < len(logits):
                    row["logit"] = logits[r]
                elif e["kind"] == "prompt" and logits and r == e["n"] - 1:
                    row["logit"] = logits[0]
                old = self.pos.get(e["p0"] + r)
                if old and old["kind"] == "decode" and e["kind"] == "decode":
                    self.reruns.append((e["p0"] + r, old["k"] == row["k"] and old["v"] == row["v"]))
                self.pos[e["p0"] + r] = row
            self.conv[e["p0"] + e["n"]] = split(e["conv"])
        elif ev == "keep":
            for p in [p for p in self.pos if p >= e["len"]]:
                del self.pos[p]
            for n in [n for n in self.conv if n > e["len"]]:
                del self.conv[n]
            self.conv[e["len"]] = split(e["conv"])
        elif ev == "snap":
            self.snaps[(e["when"], e["at"])] = e
        elif ev == "plant":
            self.plant = e


def streams(path, span):
    out = {}
    with open(path, "rb") as f:
        f.seek(span[0])
        data = f.read(span[1] - span[0])
    for line in data.splitlines():
        e = json.loads(line)
        out.setdefault(e["sid"], Stream()).feed(e)
    return out


def stream(path, span, plen):
    got = [s for s in streams(path, span).values() if s.plen == plen]
    if len(got) != 1:
        raise ValueError(f"{len(got)} streams with a prompt of {plen} tokens in bytes {span}")
    return got[0]


def differences(a, b):
    """Every sum at a common position that both sides kept: (position, visit, kind) where they differ, and the counts."""
    diffs, counts = [], collections.Counter()

    def one(position, visit, kind, x, y):
        if not x or not y or x == ZERO or y == ZERO:
            return
        counts[kind] += 1
        if x != y:
            diffs.append((position, visit, kind))

    for p in sorted(set(a.pos) & set(b.pos)):
        x, y = a.pos[p], b.pos[p]
        for kind in ("k", "v"):
            for visit in range(min(len(x[kind]), len(y[kind]))):
                one(p, visit, kind, x[kind][visit], y[kind][visit])
        for i, visit in enumerate(TAPS):
            if "tap" in x and "tap" in y:
                one(p, visit, "tap", x["tap"][i], y["tap"][i])
        one(p, 48, "logit", x.get("logit"), y.get("logit"))
    for n in sorted(set(a.conv) & set(b.conv)):
        for i, (x, y) in enumerate(zip(a.conv[n], b.conv[n])):
            one(n - 1, i // 4, "conv." + SITES[i % 4], x, y)
    diffs.sort(key=lambda d: (d[0], d[1], ORDER[d[2]]))
    return diffs, counts


def classes(prompt, inserts):
    def of(p):
        if p < prompt:
            return "prompt"
        return "insert" if any(s <= p < e for s, e in inserts) else "model"
    return of


def passes(diffs):
    out = collections.Counter()
    for _, visit, kind in diffs:
        if kind in ("k", "v") or kind.startswith("conv."):
            out[next(name for lo, hi, name in PASSES if lo <= visit <= hi)] += 1
    return dict(out)


def report(a, b, of=None):
    diffs, counts = differences(a, b)
    unequal = collections.Counter(d[2] for d in diffs)
    out = {"positions": len(set(a.pos) & set(b.pos)), "conv_lengths": len(set(a.conv) & set(b.conv)),
           "compared": dict(counts), "unequal": dict(unequal), "equal": not diffs, "by_pass": passes(diffs)}
    if diffs:
        p, visit, kind = diffs[0]
        out["first"] = {"position": p, "visit": visit, "kind": kind}
        if of:
            out["first"]["class"] = of(p)
    if of:
        rows = collections.defaultdict(lambda: [0, 0])
        bad = {d[0] for d in diffs if not d[2].startswith("conv.")}
        for p in set(a.pos) & set(b.pos):
            rows[of(p)][p in bad] += 1
        out["positions_by_class"] = {k: {"equal": v[0], "unequal": v[1]} for k, v in rows.items()}
    snaps = sorted(set(a.snaps) & set(b.snaps))
    out["snapshots"] = {f"{when}@{at}": a.snaps[(when, at)]["all"] == b.snaps[(when, at)]["all"] for when, at in snaps}
    ends = {s["at"]: s for (when, _), s in a.snaps.items() if when == "release"}
    for (when, at), s in b.snaps.items():
        if when == "release" and at in ends:
            x, y = ends[at].get("draft"), s.get("draft")
            out["end_snapshot"] = ends[at]["all"] == s["all"]
            if x and y and (x["from"], x["end"]) == (y["from"], y["end"]):
                out["drafter_rings"] = x["ring"] == y["ring"]
    out["reruns"] = {"rows": len(a.reruns), "changed": sum(1 for _, same in a.reruns if not same)}
    return out


def rows_of(folder):
    out = collections.defaultdict(dict)
    for line in open(os.path.join(folder, "rows.jsonl")):
        r = json.loads(line)
        if r.get("sums") and "case" in r:
            out[(r["case"], r["sampler"])][r["check"]] = r
    return out


def pairs(args):
    sums = os.path.join(args.dir, "sums", "sums.jsonl")
    ref = rows_of(args.ref) if args.ref else {}
    bad = 0
    with open(args.out, "w") as out:
        for (case, sampler), checks in sorted(rows_of(args.dir).items()):
            if "A" not in checks:
                continue
            ids = json.load(open(os.path.join(args.dir, "ids", f"{case}-{sampler}.json")))
            plen = len(ids["prompt"])
            of = classes(plen, ids["inserts"])
            a = stream(sums, checks["A"]["sums"], plen)
            others = [(name, sums, r["sums"], plen if name.startswith("A.") else r["history"], name.startswith("A."))
                      for name, r in sorted(checks.items())
                      if name != "A" and not name.endswith(".peer") and (name.startswith("A.") or name.startswith("C."))]
            if (case, sampler) in ref and "R" in ref[(case, sampler)]:
                r = ref[(case, sampler)]["R"]
                others.append(("R", os.path.join(args.ref, "sums", "sums.jsonl"), r["sums"], r["history"], True))
            for name, path, span, n, must in others:
                line = {"case": case, "sampler": sampler, "pair": f"A|{name}", "must_equal": must,
                        **report(a, stream(path, span, n), of)}
                bad += 1 if must and not line["equal"] else 0
                out.write(json.dumps(line) + "\n")
                first = line.get("first")
                print(("ok   " if line["equal"] else "DIFF " if not must else "FAIL "), case, sampler, line["pair"],
                      line["positions"], line["unequal"], first or "", flush=True)
    sys.exit(1 if bad else 0)


def plant(args):
    def lane(folder):
        checks = next(iter(rows_of(folder).values()))
        r = checks["A"]
        ids = json.load(open(os.path.join(folder, "ids", f"{r['case']}-{r['sampler']}.json")))
        return stream(os.path.join(folder, "sums", "sums.jsonl"), r["sums"], len(ids["prompt"]))
    planted, clean = lane(args.dir), lane(args.clean)
    diffs, _ = differences(planted, clean)
    first = diffs[0] if diffs else None
    at = planted.plant or {}
    red = bool(first) and first[0] == at.get("len") and first[1] == at.get("visit", -2) + 1 and first[2] in ("k", "v")
    print(json.dumps({"planted": planted.plant, "first": first, "differences": len(diffs), "red": red}))
    sys.exit(0 if red else 1)


def selftest():
    a = Stream()
    a.feed({"ev": "open", "plen": 2, "resume": 0})
    for p in range(6):
        a.feed({"ev": "rows", "kind": "decode", "p0": p, "n": 1, "len": p + 1, "k": ["".join(f"{p * 100 + v:016x}" for v in range(48))],
                "v": ["".join(f"{p * 100 + v + 50:016x}" for v in range(48))], "tap": ["".join(f"{p + t:016x}" for t in range(1, 5))],
                "logit": [f"{p + 7:016x}"], "conv": "".join(f"{p * 1000 + i + 1:016x}" for i in range(192))})
    assert differences(a, copy.deepcopy(a))[0] == []
    for position, visit, kind in [(3, 20, "k"), (4, 47, "v"), (2, 23, "tap"), (5, 48, "logit"), (3, 11, "conv.attn")]:
        b = copy.deepcopy(a)
        if kind in ("k", "v"):
            b.pos[position][kind][visit] = "f" * HEX
        elif kind == "tap":
            b.pos[position]["tap"][TAPS.index(visit)] = "f" * HEX
        elif kind == "logit":
            b.pos[position]["logit"] = "f" * HEX
        else:
            b.conv[position + 1][visit * 4 + SITES.index(kind[5:])] = "f" * HEX
        got = differences(a, b)[0]
        assert got == [(position, visit, kind)], (got, position, visit, kind)
    b = copy.deepcopy(a)
    b.feed({"ev": "keep", "len": 4, "conv": "".join(f"{3 * 1000 + i + 1:016x}" for i in range(192))})
    assert sorted(b.pos) == [0, 1, 2, 3] and differences(a, b)[0] == []
    print("selftest ok: 5 planted sums named exactly; a rollback drops its rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "plant", "selftest"])
    ap.add_argument("--dir")
    ap.add_argument("--ref")
    ap.add_argument("--clean")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "selftest":
        return selftest()
    {"pairs": pairs, "plant": plant}[args.mode](args)


if __name__ == "__main__":
    main()
