"""The lane loop integrity test's client: requests, token ids and Inkling's message structure. Standard library only."""
import json
import os
import urllib.error
import urllib.request

API = os.environ.get("API_URL", "http://127.0.0.1:8888")
MODEL = "Inkling-2-Mini"
LANE_TOOLS = [{"type": "code_interpreter"}]
# compute_turns.zig's built-in tool, as the lane adds it to a request that lists code_interpreter
BUILT_IN = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python 3 code in a sandbox with no network, no files and no installs, and get its stdout and "
                   "stderr. print() what you need; the value of a final bare expression is printed too.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python 3 source to run"}},
                   "required": ["code"]}}}
SAMPLERS = {
    "greedy": {"temperature": 0},
    "t10": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 20261126},
    "t07": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 20261126},
}


def post(path, body, timeout=1200):
    request = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode(errors='replace')[:600]}") from None


def flush():
    """Drops every kept prompt state, in RAM and on disk (loopback callers only)."""
    return post("/v1/prompt_cache/flush", {})


def prompt_ids(messages, tools=None, effort=None):
    body = {"messages": messages, "add_generation_prompt": True}
    if tools:
        body["tools"] = tools
    if effort is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    return post("/tokenize", body)["tokens"]


def detokenize(ids):
    return post("/detokenize", {"tokens": ids})["prompt"]


def chat(messages, sampler, max_tokens, tools=None, draft=True, effort=None, extra=None, timeout=1200):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "return_token_ids": True}
    body.update(SAMPLERS[sampler] if isinstance(sampler, str) else sampler)
    if tools:
        body["tools"] = tools
    if not draft:
        body["draft"] = False
    if effort is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    if extra:
        body.update(extra)
    return post("/v1/chat/completions", body, timeout)


def complete(ids, sampler, max_tokens, draft=True):
    """A raw token prompt: no template, no tokenizer, no lane."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "return_token_ids": True}
    body.update(SAMPLERS[sampler] if isinstance(sampler, str) else sampler)
    if not draft:
        body["draft"] = False
    return post("/v1/completions", body)


def ids_of(reply):
    return reply["tensorfold"]["token_ids"]


def cached_of(reply):
    return (reply.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)


class Marks:
    """Inkling's message tokens, read from the served tokenizer."""

    names = ["<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>",
             "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>",
             "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>", "<|content_tool_error|>"]

    def __init__(self):
        self.id = {}
        for name in self.names:
            got = post("/tokenize", {"prompt": name, "add_special_tokens": False})["tokens"]
            if len(got) != 1:
                raise RuntimeError(f"{name} is not one token: {got}")
            self.id[name] = got[0]
        self.name = {v: k for k, v in self.id.items()}
        self.eos = self.id["<|content_model_end_sampling|>"]
        self.tool = self.id["<|message_tool|>"]
        self.end = self.id["<|end_message|>"]
        self.model = self.id["<|message_model|>"]
        self.thinking = self.id["<|content_thinking|>"]
        self.text = self.id["<|content_text|>"]
        self.invoke = (self.id["<|content_invoke_tool_json|>"], self.id["<|content_invoke_tool_text|>"])


def segments(ids, marks):
    """A lane reply's ids as [("seg", ids), ("ins", ids), ...]: a segment runs through the model's EOS, an insert is
    the run of tool messages forced after it."""
    out = []
    at = 0
    while at < len(ids):
        try:
            end = ids.index(marks.eos, at) + 1
        except ValueError:
            end = len(ids)
        out.append(("seg", ids[at:end]))
        at = end
        start = at
        while at < len(ids) and ids[at] == marks.tool:
            at = ids.index(marks.end, at) + 1
        if at > start:
            out.append(("ins", ids[start:at]))
    return out


def messages(ids, mark):
    """A reply's ids as messages: {"kind", "at", "n", "closed"}. kind: thinking, text, invoke, tool, eos or stray; n: the
    body's tokens; closed: it ended with <|end_message|>. `mark` maps Inkling's message token names to ids."""
    role = {mark["<|message_model|>"]: "model", mark["<|message_tool|>"]: "tool", mark["<|message_user|>"]: "user",
            mark["<|message_system|>"]: "system"}
    kinds = {mark["<|content_thinking|>"]: "thinking", mark["<|content_text|>"]: "text",
             mark["<|content_invoke_tool_json|>"]: "invoke", mark["<|content_invoke_tool_text|>"]: "invoke",
             mark["<|content_tool_error|>"]: "text"}
    end, eos = mark["<|end_message|>"], mark["<|content_model_end_sampling|>"]
    stop = set(role) | {end, eos}
    out = []
    at = 0
    while at < len(ids):
        t = ids[at]
        if t == eos:
            out.append({"kind": "eos", "at": at, "n": 0, "closed": True})
            at += 1
            continue
        if t not in role:
            if out and out[-1]["kind"] == "stray":
                out[-1]["n"] += 1
            else:
                out.append({"kind": "stray", "at": at, "n": 1, "closed": False})
            at += 1
            continue
        j = at + 1
        while j < len(ids) and ids[j] not in kinds and ids[j] not in stop:
            j += 1
        kind = kinds.get(ids[j], "open") if j < len(ids) else "open"
        k = j + 1 if kind != "open" else j
        while k < len(ids) and ids[k] not in stop:
            k += 1
        closed = k < len(ids) and ids[k] == end
        out.append({"kind": "tool" if role[t] == "tool" else kind, "at": at, "n": max(0, k - j - 1), "closed": closed,
                    "span": k + closed - at})
        at = k + closed
    return out
