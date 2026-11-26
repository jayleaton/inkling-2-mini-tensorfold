"""Q1, state integrity: a lane reply's tokens against a client round trip and a cold prefill of the same token ids.

  q1.py lane    --out DIR [--cases 1,2] [--samplers greedy,t07] [--sums FILE] [--light]
  q1.py control --out DIR [--samplers ...] [--sums FILE]       the no-lane baseline (arm N)
  q1.py ref     --out DIR --from DIR --sums FILE               each arm A history of --from but its last token, cold, on
                                                               a boot whose prompt pass runs the decode rows (check R)
  q1.py disk    --out DIR --restart "CMD" [--cases 2,7]         arm D: a restart between rounds (states on the NVMe)

Every check is one JSON line in DIR/rows.jsonl: {"check", "ok", ...}. Runs on the Spark (the flush is loopback only).
"""
import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import time

import lli

MAX_TOKENS = 6000
WORDS = ("amber basil cedar delta ember fjord grove harbor island juniper kettle lantern meadow nectar orchard "
         "pebble quartz river saddle timber umber valley willow yarrow zephyr").split()


def seq(n, a, b):
    return f"print(*[(i*{a}+{b})%10007 for i in range({n})])"


CASES = {
    1: {"codes": [seq(20, 7919, 3), seq(20, 7907, 5)], "cls": ("before", 1024)},
    2: {"codes": [seq(150, 7919, 11), seq(10, 7907, 5)], "cls": ("straddle", 1024)},
    3: {"codes": [seq(12, 7919, 1), seq(12, 7907, 2), seq(12, 7901, 3)], "cls": ("between", 1024, 1536)},
    4: {"codes": [seq(150, 7919, 13), seq(10, 7907, 7)], "cls": ("straddle", 1536)},
    5: {"codes": [seq(150, 7919, 17), seq(10, 7907, 9)], "cls": ("straddle", 2048)},
    6: {"codes": [seq(150, 7919, 19), seq(10, 7907, 11)], "cls": ("straddle", 4096)},
    7: {"codes": [seq(170, 7919, 23), seq(170, 7907, 29), seq(170, 7901, 31)], "cls": ("forced", 400, 1024)},
    8: {"codes": [f"print({' '.join('ABCDEF'[:k])!r})" for k in range(1, 7)], "cls": ("short", 8)},
    9: {"codes": [seq(20, 7919, 37), seq(20, 7907, 41)], "cls": ("together",), "together": True},
    10: {"codes": [seq(8, 7919 - 2 * k, 43 + k) for k in range(6)], "cls": ("from", 3000)},
}
PEER_LANE = [{"role": "user", "content": "Use the python tool twice, one call at a time. First call: "
              "print(123456789*987654321). Second call: print the first result modulo 1000003. Then tell me both numbers."}]
PEER_PLAIN = [{"role": "user", "content": "Explain in about 500 words how inserting a key into a B-tree works, with "
               "the node split step by step."}]
CONTROL = {
    "essay": {"content": "Write a careful 1,200-word explanation of how a hash map with open addressing handles "
                         "insertions, deletions with tombstones, and resizing. Include a worked example.", "pad": 0},
    "proof": {"content": "Prove that the square root of 2 is irrational in three different ways, explaining each "
                         "proof step by step, then compare them in a closing paragraph.", "pad": 0},
    "padded": {"content": "Summarise in about 600 words why binary search needs a sorted array, with a worked trace.",
               "pad": 3300},
}
CUTS = [700, 1016, 1024, 1032, 1300, 1536, 2048, 2500, 4096, 4400]


def pad_text(seed, words):
    if words <= 0:
        return ""
    rng = random.Random(seed)
    return "Background notes, not needed for the task:\n" + " ".join(rng.choice(WORDS) for _ in range(words)) + "\n\n"


def task(case):
    c = CASES[case]
    n = len(c["codes"])
    steps = "\n".join(f"Call {i + 1}: {code}" for i, code in enumerate(c["codes"]))
    how = ("Make all the calls in the same turn, one right after the other, before you read any result."
           if c.get("together") else
           "Make one call at a time: wait for each result before the next call, and never put two calls in one turn.")
    return (f"Use the python tool to run these {n} snippets exactly as written, in order.\n{steps}\n{how}\n"
            "After the last result, reply with one line: DONE and the first number of the last output.")


def build(case, words):
    return [{"role": "user", "content": pad_text(case, words) + task(case)}]


def inserts(plen, segs):
    out = []
    at = plen
    for kind, ids in segs:
        if kind == "ins":
            out.append((at, at + len(ids)))
        at += len(ids)
    return out


def fits(case, pos, segs, marks):
    """Whether the realised insert positions meet the case's class; else how many prompt tokens to add."""
    c = CASES[case]
    cls = c["cls"]
    want = 1 if c.get("together") else len(c["codes"])
    if len(pos) < want:
        return False, None, f"{len(pos)} inserts, {want} wanted"
    kind = cls[0]
    if kind == "before":
        return all(e < cls[1] for _, e in pos), None, "an insert ends past the bound"
    if kind == "straddle":
        s, e = pos[0]
        if s + 8 <= cls[1] <= e - 8:
            return True, None, ""
        return False, cls[1] - (e - s) // 2 - s, f"insert 1 at [{s},{e}) misses {cls[1]}"
    if kind == "between":
        if pos[0][0] >= cls[1] and pos[-1][1] <= cls[2]:
            return True, None, ""
        return False, cls[1] + 24 - pos[0][0], f"inserts span [{pos[0][0]},{pos[-1][1]})"
    if kind == "forced":
        lens = [e - s for s, e in pos]
        return min(lens) >= cls[1] and sum(lens) > cls[2], None, f"insert lengths {lens}"
    if kind == "short":
        lens = [e - s for s, e in pos]
        return min(lens) <= cls[1] and {7, 8, 9} <= set(lens), None, f"insert lengths {lens}"
    if kind == "together":
        two = any(k == "ins" and ids.count(marks.tool) >= 2 for k, ids in segs)
        return two, None, "no insert holds two tool turns"
    if kind == "from":
        if pos[0][0] >= cls[1]:
            return True, None, ""
        return False, cls[1] + 30 - pos[0][0], f"insert 1 starts at {pos[0][0]}"
    raise ValueError(kind)


class Run:
    def __init__(self, out, sums):
        self.out = out
        os.makedirs(os.path.join(out, "ids"), exist_ok=True)
        self.rows = open(os.path.join(out, "rows.jsonl"), "a")
        self.sums = sums
        self.marks = lli.Marks()
        self.bad = 0

    def mark(self):
        return os.path.getsize(self.sums) if self.sums and os.path.exists(self.sums) else 0

    def row(self, check, ok, **more):
        self.bad += 0 if ok else 1
        line = {"check": check, "ok": bool(ok), **more}
        self.rows.write(json.dumps(line) + "\n")
        self.rows.flush()
        print(("ok   " if ok else "FAIL ") + check + " " + json.dumps({k: v for k, v in more.items() if k != "sums"}), flush=True)

    def span(self, start):
        """The engine's sums for the requests since `start`: the file's byte range, once the last stream released."""
        if not self.sums:
            return None
        time.sleep(0.5)
        return [start, self.mark()]

    def equal(self, check, got, want, **more):
        n = min(len(got), len(want))
        first = next((i for i in range(n) if got[i] != want[i]), n if len(got) != len(want) else None)
        self.row(check, first is None, got=len(got), want=len(want), first_difference=first, **more)
        return first is None


def lane_request(msgs, sampler, draft=True):
    return lli.chat(msgs, sampler, MAX_TOKENS, tools=lli.LANE_TOOLS, draft=draft)


def calibrate(run, case, sampler, saved):
    key = f"{case}-{sampler}"
    words = saved.get(key, 0)
    per_word = None
    for attempt in range(6):
        msgs = build(case, words)
        prompt = lli.prompt_ids(msgs, [lli.BUILT_IN])
        lli.flush()
        start = run.mark()
        reply = lane_request(msgs, sampler)
        span = run.span(start)
        ids = lli.ids_of(reply)
        segs = lli.segments(ids, run.marks)
        pos = inserts(len(prompt), segs)
        ok, delta, why = fits(case, pos, segs, run.marks)
        said = reply["usage"]["prompt_tokens"]
        if said != len(prompt):
            run.row("A.prompt", False, case=case, sampler=sampler, tokenized=len(prompt), served=said)
            return None
        if ok:
            saved[key] = words
            return {"msgs": msgs, "prompt": prompt, "reply": reply, "ids": ids, "segs": segs, "pos": pos, "words": words,
                    "attempts": attempt + 1, "span": span}
        if delta is None:
            break
        if per_word is None:
            a = len(lli.prompt_ids(build(case, 400), [lli.BUILT_IN]))
            b = len(lli.prompt_ids(build(case, 0), [lli.BUILT_IN]))
            per_word = (a - b) / 400
        words = max(0, words + round(delta / per_word))
    run.row("A.class", False, case=case, sampler=sampler, why=why, positions=pos)
    return None


def histories(prompt, segs):
    """Each round's (history, expected next segment): the prompt, then the prompt and everything through each insert."""
    out = []
    hist = list(prompt)
    seg = None
    for kind, ids in segs:
        if kind == "seg":
            seg = ids
            out.append((list(hist), ids))
            hist += ids
        else:
            hist += ids
    return out


NEVER = {"stop": ["<<never-7f3a91>>"]}  # a stop string turns the lane's hold off for the request; it never matches


def spans(ids, marks):
    """A run of whole messages as (role token, header text, content token, body text)."""
    out = []
    at = 0
    while at < len(ids) and ids[at] != marks.eos:
        role = ids[at]
        j = at + 1
        while ids[j] not in marks.name:
            j += 1
        end = ids.index(marks.end, j)
        out.append((role, lli.detokenize(ids[at + 1:j]) if j > at + 1 else "", ids[j], lli.detokenize(ids[j + 1:end]) if end > j + 1 else ""))
        at = end + 1
    return out


def client_turns(run, segs):
    """Each segment and insert as the messages a client would send back; None for a segment the template cannot
    write (two thinking messages, text after a call)."""
    turns = []
    for kind, ids in segs:
        if kind == "ins":
            turns.append([{"role": "tool", "name": name, "content": body} for _, name, _, body in spans(ids, run.marks)])
            continue
        if not ids or ids[-1] != run.marks.eos:
            turns.append(None)
            continue
        turn = {"role": "assistant", "content": ""}
        order = []
        for _, name, content, body in spans(ids, run.marks):
            if content == run.marks.thinking:
                order.append("t")
                turn["reasoning_content"] = body
            elif content == run.marks.text:
                order.append("x")
                turn["content"] = body
            elif content == run.marks.id["<|content_invoke_tool_json|>"]:
                order.append("c")
                call = json.loads(body)
                turn.setdefault("tool_calls", []).append({"type": "function", "function": {"name": call["name"], "arguments": call["args"]}})
            else:
                order.append("?")
        shape = "".join(order)
        turns.append([turn] if shape in ("t", "x", "tx", "c", "tc", "xc", "txc") or set(shape) <= {"c"} or (shape[:1] == "t" and set(shape[1:]) <= {"c"}) else None)
    return turns


def round_trip(run, label, case, sampler, messages, hist, want, need_cached):
    """One client request on the chat route: the template must render the history's exact ids."""
    rendered = lli.prompt_ids(messages, [lli.BUILT_IN])
    if rendered != hist:
        first = next((i for i in range(min(len(rendered), len(hist))) if rendered[i] != hist[i]), min(len(rendered), len(hist)))
        run.row(label + ".render", False, case=case, sampler=sampler, rendered=len(rendered), history=len(hist), first_difference=first)
        return False
    start = run.mark()
    reply = lli.chat(messages, sampler, len(want) + 32, tools=[lli.BUILT_IN], extra=NEVER)
    cached = lli.cached_of(reply)
    ok = run.equal(label, lli.ids_of(reply), want, case=case, sampler=sampler, history=len(hist), cached=cached,
                   finish=reply["choices"][0]["finish_reason"], sums=run.span(start))
    if need_cached and cached == 0:
        run.row(label + ".cached", False, case=case, sampler=sampler, cached=cached, why="no state was reused")
    return ok


def client_rounds(run, label, case, sampler, msgs, prompt, segs, first=0, last=None, cached_from=1):
    """Arm B's requests from round `first` on: each sends the earlier turns back as messages."""
    turns = client_turns(run, segs)
    rounds = histories(prompt, segs)
    messages = list(msgs)
    k = 0
    for i, (kind, ids) in enumerate(segs):
        if kind != "seg":
            messages = messages + turns[i]
            continue
        if k >= first and (last is None or k < last):
            hist, want = rounds[k]
            round_trip(run, f"{label}.round{k}", case, sampler, messages, hist, want, cached_from is not None and k >= cached_from)
        k += 1
        if turns[i] is None:
            if i + 1 < len(segs):
                run.row(f"{label}.round{k}.render", False, case=case, sampler=sampler, why="the template cannot write this segment")
            return
        messages = messages + turns[i]


def replay(run, label, case, sampler, hist, want, cold, need_cached=None):
    if cold:
        lli.flush()
    start = run.mark()
    reply = lli.complete(hist, sampler, len(want) + 32)
    cached = lli.cached_of(reply)
    ok = run.equal(label, lli.ids_of(reply), want, case=case, sampler=sampler, history=len(hist), cached=cached,
                   finish=reply["choices"][0]["finish_reason"], sums=run.span(start))
    if need_cached is True and cached == 0:
        run.row(label + ".cached", False, case=case, sampler=sampler, cached=cached, why="no state was reused")
    if need_cached is False and cached != 0:
        run.row(label + ".cached", False, case=case, sampler=sampler, cached=cached, why="a cold replay reused a state")
    return ok


def solo(run, name, msgs, sampler, tools):
    lli.flush()
    reply = lli.chat(msgs, sampler, MAX_TOKENS if tools else 900, tools=tools)
    return lli.ids_of(reply)


def lane(run, cases, samplers, light):
    saved_path = os.path.join(run.out, "pads.json")
    saved = json.load(open(saved_path)) if os.path.exists(saved_path) else {}
    for sampler in samplers:
        peers = None
        for case in cases:
            a = calibrate(run, case, sampler, saved)
            json.dump(saved, open(saved_path, "w"))
            if a is None:
                continue
            ids, segs, prompt = a["ids"], a["segs"], a["prompt"]
            tf = a["reply"]["tensorfold"]
            calls = [{"status": c["status"], "exit_code": c["exit_code"], "inserted": c.get("inserted")} for c in tf.get("compute", {}).get("calls", [])]
            json.dump({"case": case, "sampler": sampler, "pad_words": a["words"], "messages": a["msgs"], "prompt": prompt,
                       "continuation": ids, "inserts": a["pos"], "calls": calls, "token_sha256": tf["token_sha256"],
                       "speculative": a["reply"].get("speculative")},
                      open(os.path.join(run.out, "ids", f"{case}-{sampler}.json"), "w"))
            run.row("A", True, case=case, sampler=sampler, prompt=len(prompt), continuation=len(ids), inserts=a["pos"],
                    insert_lengths=[e - s for s, e in a["pos"]], calls=calls, attempts=a["attempts"], sums=a["span"],
                    finish=a["reply"]["choices"][0]["finish_reason"])
            # the lane reply again: from the kept state of its own prompt, and one token a round with no drafts
            start = run.mark()
            warm = lane_request(a["msgs"], sampler)
            run.equal("A.warm", lli.ids_of(warm), ids, case=case, sampler=sampler, cached=lli.cached_of(warm), sums=run.span(start))
            lli.flush()
            start = run.mark()
            run.equal("A.nodraft", lli.ids_of(lane_request(a["msgs"], sampler, draft=False)), ids, case=case, sampler=sampler, sums=run.span(start))
            if not light:
                if peers is None:
                    peers = {"lane": solo(run, "lane", PEER_LANE, sampler, lli.LANE_TOOLS), "plain": solo(run, "plain", PEER_PLAIN, sampler, None)}
                for name, msgs, tools in (("lane", PEER_LANE, lli.LANE_TOOLS), ("plain", PEER_PLAIN, None)):
                    lli.flush()
                    start = run.mark()
                    with concurrent.futures.ThreadPoolExecutor(2) as pool:
                        peer = pool.submit(lli.chat, msgs, sampler, MAX_TOKENS if tools else 900, tools)
                        time.sleep(0.2)
                        mine = pool.submit(lane_request, a["msgs"], sampler)
                        got, other = lli.ids_of(mine.result()), lli.ids_of(peer.result())
                    span = run.span(start)
                    run.equal(f"A.with-{name}", got, ids, case=case, sampler=sampler, sums=span)
                    run.equal(f"A.with-{name}.peer", other, peers[name], case=case, sampler=sampler)
            rounds = histories(prompt, segs)
            # B: the client's round trip, each request extending the last one's prompt and reply
            lli.flush()
            client_rounds(run, "B", case, sampler, a["msgs"], prompt, segs)
            # C: each round's history prefilled cold
            for k, (hist, want) in enumerate(rounds):
                replay(run, f"C.round{k}", case, sampler, hist, want, cold=True, need_cached=False)
            # the whole history but its last token, cold: the state the lane reply ended in, and its last draw
            whole = prompt + ids
            lli.flush()
            start = run.mark()
            last = lli.complete(whole[:-1], sampler, 1)
            run.equal("C.final", lli.ids_of(last), whole[-1:], case=case, sampler=sampler, history=len(whole) - 1, sums=run.span(start))
            if len(rounds) > 1:
                controls(run, case, sampler, prompt, segs)
    return run.bad


def controls(run, case, sampler, prompt, segs):
    """Planted faults in the replayed history: each should change the reply (a control that stays equal is reported)."""
    seg1, ins1, seg2 = segs[0][1], segs[1][1], segs[2][1]
    last_ins = max(i for i, (kind, _) in enumerate(segs) if kind == "ins")
    before = prompt + [t for _, ids in segs[:last_ins] for t in ids]
    ins, tail = segs[last_ins][1], segs[last_ins + 1][1]
    at = ins.index(run.marks.text) + 1
    bad = list(ins)
    bad[at] = next((t for t in ins[at + 1:] if t not in run.marks.name and t != ins[at]), ins[at] + 1)
    lli.flush()
    start = run.mark()
    got = lli.ids_of(lli.complete(before + bad, sampler, len(tail) + 32))
    run.row("control.result-token", True, red=got != tail, case=case, sampler=sampler, history=len(before) + len(bad), sums=run.span(start))
    lli.flush()
    start = run.mark()
    got = lli.ids_of(lli.complete(prompt + seg1[:-1] + ins1, sampler, len(seg2) + 32))
    run.row("control.no-eos", True, red=got != seg2, case=case, sampler=sampler, history=len(prompt) + len(seg1) - 1 + len(ins1), sums=run.span(start))
    if sampler != "greedy":
        other = dict(lli.SAMPLERS[sampler])
        other["seed"] += 1
        lli.flush()
        got = lli.ids_of(lli.complete(prompt + seg1 + ins1, other, len(seg2) + 32))
        run.row("control.seed", True, red=got != seg2, case=case, sampler=sampler)


def control(run, samplers):
    """Arm N: no lane anywhere. A plain reply, then the same ids cut at fixed positions and prefilled cold or warm."""
    for sampler in samplers:
        for name, spec in CONTROL.items():
            msgs = [{"role": "user", "content": pad_text(99, spec["pad"]) + spec["content"]}]
            prompt = lli.prompt_ids(msgs)
            lli.flush()
            start = run.mark()
            reply = lli.chat(msgs, sampler, 2600)
            ids = lli.ids_of(reply)
            json.dump({"name": name, "sampler": sampler, "prompt": prompt, "continuation": ids},
                      open(os.path.join(run.out, "ids", f"N-{name}-{sampler}.json"), "w"))
            run.row("N", reply["usage"]["prompt_tokens"] == len(prompt), name=name, sampler=sampler, prompt=len(prompt),
                    continuation=len(ids), sums=run.span(start))
            lli.flush()
            start = run.mark()
            run.equal("N.route", lli.ids_of(lli.complete(prompt, sampler, len(ids) + 8)), ids, name=name, sampler=sampler, sums=run.span(start))
            cuts = [c - len(prompt) for c in CUTS if 8 <= c - len(prompt) <= len(ids) - 8]
            for cut in cuts:
                hist = prompt + ids[:cut]
                lli.flush()
                start = run.mark()
                cold = lli.complete(hist, sampler, len(ids) - cut + 8)
                run.equal("N.cold", lli.ids_of(cold), ids[cut:], name=name, sampler=sampler, cut_at=len(hist), cached=lli.cached_of(cold), sums=run.span(start))
            # warm: a second turn resumed from the first reply's kept state, against the same ids prefilled cold
            turn = client_turns(run, [("seg", ids)])[0]
            if turn is None or reply["choices"][0]["finish_reason"] != "stop":
                run.row("N.turn2", True, name=name, sampler=sampler, skipped="the first reply does not end as one turn")
                continue
            second = msgs + turn + [{"role": "user", "content": "Now give the five most important points of that as a numbered list."}]
            prompt2 = lli.prompt_ids(second)
            if prompt2[:len(prompt) + len(ids)] != prompt + ids:
                run.row("N.turn2.render", False, name=name, sampler=sampler, why="the template does not render the first reply's ids")
                continue
            lli.flush()
            lli.chat(msgs, sampler, 2600)
            warm = lli.chat(second, sampler, 500)
            lli.flush()
            cold = lli.complete(prompt2, sampler, 500)
            run.equal("N.turn2", lli.ids_of(warm), lli.ids_of(cold), name=name, sampler=sampler, history=len(prompt2), cached=lli.cached_of(warm))
            if lli.cached_of(warm) == 0:
                run.row("N.turn2.cached", False, name=name, sampler=sampler, why="no state was reused")
    return run.bad


def restart(command, sessions):
    """Waits for the idle engine to write its kept states, restarts it, and returns the state files before and after."""
    def states():
        return sum(name.endswith(".st") for _, _, names in os.walk(sessions) for name in names) if sessions else None
    last = -1
    for _ in range(20):
        time.sleep(3)
        if states() == last:
            break
        last = states()
    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL)
    return [last, states()]


def disk(run, cases, samplers, command, sessions):
    """Arm D: the container restarts between two rounds, so the later one resumes from the session folder."""
    for sampler in samplers:
        for case in cases:
            path = os.path.join(run.out, "ids", f"{case}-{sampler}.json")
            if not os.path.exists(path):
                run.row("D.history", False, case=case, sampler=sampler, why="no arm A history for this case")
                continue
            a = json.load(open(path))
            segs = lli.segments(a["continuation"], run.marks)
            rounds = histories(a["prompt"], segs)
            middle = max(1, len(rounds) // 2)
            lli.flush()
            client_rounds(run, "D", case, sampler, a["messages"], a["prompt"], segs, last=middle, cached_from=None)
            kept = restart(command, sessions)
            run.row("D.states", True, case=case, sampler=sampler, state_files_before_and_after_restart=kept)
            client_rounds(run, "D.after-restart", case, sampler, a["messages"], a["prompt"], segs, first=middle, cached_from=middle)
            # a second lane turn on a conversation whose first lane turn is resumed from the disk after a restart
            first = a["messages"]
            answer = lli.chat(first, sampler, MAX_TOKENS, tools=lli.LANE_TOOLS)["choices"][0]["message"].get("content") or ""
            other = 1 if case != 1 else 3
            second = first + [{"role": "assistant", "content": answer}, {"role": "user", "content": task(other)}]
            lli.flush()
            fresh = lli.ids_of(lli.chat(second, sampler, MAX_TOKENS, tools=lli.LANE_TOOLS))
            lli.flush()
            lli.chat(first, sampler, MAX_TOKENS, tools=lli.LANE_TOOLS)
            kept = restart(command, sessions)
            again = lli.chat(second, sampler, MAX_TOKENS, tools=lli.LANE_TOOLS)
            cached = lli.cached_of(again)
            run.equal("D.second-lane-turn", lli.ids_of(again), fresh, case=case, sampler=sampler, cached=cached,
                      state_files_before_and_after_restart=kept)
            if cached == 0:
                run.row("D.second-lane-turn.cached", False, case=case, sampler=sampler, cached=0, why="nothing came from the disk")
    return run.bad


def ref(run, source, cases, samplers):
    for sampler in samplers:
        for case in cases:
            path = os.path.join(source, "ids", f"{case}-{sampler}.json")
            if not os.path.exists(path):
                continue
            a = json.load(open(path))
            whole = a["prompt"] + a["continuation"]
            lli.flush()
            start = run.mark()
            last = lli.complete(whole[:-1], sampler, 1)
            run.equal("R", lli.ids_of(last), whole[-1:], case=case, sampler=sampler, history=len(whole) - 1, sums=run.span(start))
    return run.bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lane", "control", "disk", "ref"])
    ap.add_argument("--from", dest="source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(str(c) for c in CASES))
    ap.add_argument("--samplers", default="greedy,t10,t07")
    ap.add_argument("--sums")
    ap.add_argument("--light", action="store_true", help="skip the concurrent variants")
    ap.add_argument("--restart")
    ap.add_argument("--sessions", help="the engine's session folder on this host (arm D waits for its state files)")
    args = ap.parse_args()
    run = Run(args.out, args.sums)
    cases = [int(c) for c in args.cases.split(",")]
    samplers = args.samplers.split(",")
    if args.mode == "lane":
        bad = lane(run, cases, samplers, args.light)
    elif args.mode == "control":
        bad = control(run, samplers)
    elif args.mode == "ref":
        bad = ref(run, args.source, cases, samplers)
    else:
        bad = disk(run, cases, samplers, args.restart, args.sessions)
    print(f"SUMMARY failed_checks={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
