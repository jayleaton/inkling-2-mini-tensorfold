#!/usr/bin/env python3
"""Structured output, tool calls and vision against the running server: one PASS/FAIL/SKIP line a check.

Usage: tools/features.py [--no-vision]      (API_URL / PORT / MODEL as in client.py; thinking off unless a check says;
       Inkling still reasons with thinking off, so replies that must reach content get a 2,048-token budget)
Checks: json_schema (greedy and sampled), json_object, guided_regex, guided_choice, tool_choice required, a named
function, auto, and vision (a generated solid-colour PNG named by colour; a two-image request; a corrupt image is a
400; drafted equals "draft": false). A server that refuses a feature with HTTP 4xx gets FAIL with its message.
Exit 0 when every check passed, 1 otherwise.
"""
import argparse
import base64
import json
import re
import struct
import sys
import zlib

sys.dont_write_bytecode = True
import client  # noqa: E402

WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
          "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age", "tags"],
          "additionalProperties": False}
results = []


def check(name, fn):
    try:
        ok, note = fn()
    except client.ApiError as e:
        ok, note = False, str(e)[:200]
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {note}", flush=True)


def ask(text, **kw):
    kw.setdefault("max_tokens", 256)
    return client.chat([{"role": "user", "content": text}], **kw)


def schema_ok(text):
    v = json.loads(text)
    return isinstance(v, dict) and set(v) == {"name", "age", "tags"} and isinstance(v["age"], int) and \
        isinstance(v["tags"], list), text[:80]


def png(rgb, w=64, h=64):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def image_msg(text, *urls):
    return [{"role": "user", "content": [*({"type": "image_url", "image_url": {"url": u}} for u in urls),
                                         {"type": "text", "text": text}]}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-vision", action="store_true")
    a = ap.parse_args()
    fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA, "strict": True}}
    person = "Describe a fictional person as JSON with name, age and tags."
    check("json_schema greedy", lambda: schema_ok(ask(person, response_format=fmt).content))
    check("json_schema sampled", lambda: schema_ok(ask(person, response_format=fmt, temperature=0.8, seed=3).content))
    check("json_schema thinking", lambda: schema_ok(ask(person, response_format=fmt, thinking=True, max_tokens=4096).content))
    check("json_object", lambda: (isinstance(json.loads(ask("Give a JSON object with keys a and b.",
                                  response_format={"type": "json_object"}).content), dict), "object"))

    def regex():
        t = ask("Give a US phone number.", guided_regex=r"\(\d{3}\) \d{3}-\d{4}").content
        return re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", t) is not None, t
    check("guided_regex", regex)

    def choice():
        t = ask("Is the sky green? Answer yes or no.", guided_choice=["yes", "no"]).content
        return t in ("yes", "no"), t
    check("guided_choice", choice)

    def call(tc):
        r = ask("What's the weather in Paris?", tools=WEATHER, tool_choice=tc)
        c = r.tool_calls[0] if r.tool_calls else None
        ok = c is not None and c["name"] == "get_weather" and isinstance(json.loads(c["arguments"]).get("city"), str)
        return ok, f"{r.finish_reason} {c['name'] + ' ' + c['arguments'] if c else r.content[:80]}"
    check("tool_choice required", lambda: call("required"))
    check("tool_choice named", lambda: call({"type": "function", "function": {"name": "get_weather"}}))
    check("tool_choice auto", lambda: call("auto"))

    def none():
        r = ask("What's the weather in Paris? Answer in one sentence.", tools=WEATHER, tool_choice="none")
        return not r.tool_calls and bool(r.content.strip()), r.content[:80]
    check("tool_choice none", none)

    if not a.no_vision:
        red, blue = png((230, 20, 20)), png((20, 40, 230))

        def colour():
            t = client.chat(image_msg("What single colour fills this image? One word.", red), max_tokens=2048).content
            return "red" in t.lower(), t
        check("vision png colour", colour)

        def two():
            t = client.chat(image_msg("Name the colour of the first image, then of the second, as two words.", red, blue),
                            max_tokens=2048).content.lower()
            return t.find("red") != -1 and t.find("blue") > t.find("red"), t
        check("vision two images in order", two)

        def same():
            q = image_msg("Describe this image in one sentence.", blue)
            x = client.chat(q, max_tokens=48, seed=1).content
            y = client.chat(q, max_tokens=48, seed=1, draft=False).content
            return x == y, "equal" if x == y else f"{x[:40]!r} != {y[:40]!r}"
        check("vision drafted == draft:false", same)

        def corrupt():
            try:
                client.chat(image_msg("What is this?", "data:image/png;base64," + base64.b64encode(b"not a png").decode()),
                            max_tokens=8)
            except client.ApiError as e:
                return "400" in str(e), str(e)[:80]
            return False, "accepted"
        check("vision corrupt image is 400", corrupt)
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    client.run(main)
