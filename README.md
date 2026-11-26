<h1 align="center">Inkling 2 Mini on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

<p align="center">
  <img src=".github/spark.webp" alt="Inkling 2 Mini on DGX Spark (TensorFold)" width="100%" />
</p>

Serve **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** on a single **NVIDIA DGX Spark** (GB10, `sm_121`, 128 GB
unified memory) through an OpenAI-compatible API (plus a `/v1/messages` Messages-format endpoint). It runs [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine
(`tensorfold-native`, Jürgen Schmied's Inkling 2 Mini CUDA engine at [PR #4441](https://github.com/ashhart/TensorFold/pull/4441), commit
`dc2ca8a1`) plus `patches/0001`–`0019` in NVIDIA's PyTorch container. It has native CUDA vision (`inkling2_vision`: PNG, JPEG, WebP) and
structured output and tool calling (`libtfgrammar.so`).

- **Model:** a looped mixture of experts.
  - 26.4B parameters in total, 3.4B active (~5.4B effective).
  - 24 physical layers; the middle 12 run 3×, so a token takes 48 visits.
  - Each layer routes a token to 8 of 128 experts.
  - Attention: 40 visits use a 1,024-token sliding window, 8 attend the whole context.
  - Vision: a 23-layer `inkling2_vision` ViT encoder (0.59 GiB bf16).
  - Checkpoint: NVIDIA ModelOpt, with NVFP4 experts (layers 0–19) and MXFP8 (20–23). Its bf16 MTP module (`mtp.safetensors`) is the DSpark
    drafter (`DRAFTER=dspark`).
- **Default drafter:** [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), a DFlash 2 block-diffusion
  draft model (5 layers, 0.49B parameters, Apache-2.0).
  - Pinned at revision `4f9088ea`, downloaded into `$HOME/models/hf/Inkling-2-Mini-DFlash2` by `scripts/fetch.sh drafter` (~1 GB), then
    SHA-256 verified.
  - One pass drafts a block of 7 tokens. It reads the target's visits 5, 23, 41 and the final norm, keeps each position's 16 best tokens from
    the target's head, and its candidate selector picks one path. Sampled requests pick with the target sampler's own keyed noise.
  - Patch `0018` runs it natively in the engine (`--drafter /models/draft`). The target verifies every draft, so replies are unchanged.
- **Default checkpoint:** upstream `thinkingmachines/Inkling-2-Mini`.
  - Pinned at revision `7a67ae23` and downloaded into `$HOME/models/hf/Inkling-2-Mini` by `scripts/fetch.sh`, then SHA-256 verified.
  - `RERISKED=1` serves a local [`Inkling-2-Mini-Derisked`](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked) folder instead. It is
    never downloaded automatically. It is a behaviorally modified derivative with most safety refusals removed, so treat its output as
    untrusted data.
  - Blackfrost's [deployment kit](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked/blob/main/deployment-kit/AGENT_INSTRUCTIONS.md)
    ships only for vLLM. This recipe matches it on TensorFold: vLLM's `reasoning_effort` and `reasoning` output field, the SHA-256 checks, and
    the readiness gate.
- **Default serving config:**
  - **131,072** tokens a stream (the model's limit is 1,048,576) and **2** concurrent streams.
  - `fp8` KV cache, the DFlash2 drafter (`DRAFTS=1`, `DRAFTER=dflash2`) and native CUDA vision (`VISION=1`).
  - Prompt and conversation states kept in RAM and on the NVMe (`PROMPT_CACHE_DISK=1`, `--prompt-cache-disk`, patch `0019`): a later turn,
    or the same prompt after a restart, resumes from its state instead of being prefilled again.
  - Structured output and `tool_choice`.
  - FP8 attention projections (`TF_INKLING2_ATTN_FP8=1`) and Blackwell native FP4/MXFP8 prefill tensor cores (`TF_INKLING2_PREFILL_TC=1`).
- **One command:** `./start.sh` downloads the weights, builds the `sm_121` image on the first run and starts the server; `./stop.sh` stops it.

## Performance

### DFlash2 vs DSpark decode

Same engine build (patches `0001`–`0018`) and server settings on one DGX Spark, only `--drafter` changed: `Inkling-2-Mini-Derisked`,
`CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `TF_INKLING2_ATTN_FP8=1`, `TF_INKLING2_PREFILL_TC=1`, thinking on. Requests: `temperature 0.7`,
`top_k 20`, 512 tokens, eight different prompts. Aggregate decode tok/s (sum of the streams' decode rates), median of 2 runs.

| Streams | Prose DSpark | Prose **DFlash2** | Code DSpark | Code **DFlash2** |
| --- | ---: | ---: | ---: | ---: |
| ×1 | 46.3 | **52.1** (+13%) | 54.4 | **74.3** (+37%) |
| ×2 | 68.0 | **84.6** (+24%) | 63.4 | **111.1** (+75%) |
| ×4 | 96.6 | **115.6** (+20%) | 129.2 | **159.2** (+23%) |
| ×8 | 136.5 | **168.1** (+23%) | 190.0 | **237.5** (+25%) |

- **Tokens a round** at ×1–×8 (the drafter before its NVFP4 head and chances): **2.48** for DFlash2 (59.7% of drafts accepted), 1.75 for DSpark
  (51.7%).
- **Greedy, thinking off** (`temperature 0`, 1,024 tokens), the same harness: one stream does 106.7 tok/s on code and 65.1 on prose, two
  streams 154.0 and 97.6, six streams 243.3 and 163.3.
- **The NVFP4 head and calibrated chances** (after [tonyd2wild](https://github.com/tonyd2wild)'s patch, see Credits) add +33% (code) and +18% (prose) for one greedy stream and +20%
  (code) for one stream at `temperature 0.7`. With two streams or more they change nothing; that's where the chances switch off.

### sparkDash (DSpark)

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash/) before DFlash2, on one **NVIDIA DGX Spark** (GB10, `sm_121`). Settings:
`Inkling-2-Mini`, `CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `DRAFTS=1` (DSpark), `TF_INKLING2_ATTN_FP8=1`, `TF_INKLING2_PREFILL_TC=1`.
Prefill doesn't depend on the drafter.

#### Prefill

| Context | tok/s | TTFT | Prompt tokens |
| --- | ---: | ---: | ---: |
| 4k | **3,497.1 tok/s** | 1.34 s | 4,131 |
| 8k | **3,343.6 tok/s** | 2.62 s | 8,224 |
| 16k | **3,252.5 tok/s** | 5.21 s | 16,415 |
| 32k | **3,072.2 tok/s** | 10.84 s | 32,802 |
| 64k | **2,749.6 tok/s** | 24.01 s | 65,573 |
| 128k | **2,267.3 tok/s** | 57.99 s | 131,106 |

#### Decode (`temp 0.7` · thinking on · `top_k 20` · DSpark drafts)

| Streams | Prose agg | Prose / stream | Prose TTFT | Code agg | Code / stream | Code TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ×1 | **48.3 tok/s** | 48.3 tok/s | 71 ms | **56.1 tok/s** | 56.1 tok/s | 183 ms |
| ×2 | **74.9 tok/s** | 37.6 tok/s | 312 ms | **86.5 tok/s** | 46.5 tok/s | 216 ms |
| ×4 | **106.1 tok/s** | 27.9 tok/s | 366 ms | **108.5 tok/s** | 31.6 tok/s | 407 ms |
| ×8 | **145.0 tok/s** | 20.1 tok/s | 697 ms | **149.4 tok/s** | 24.6 tok/s | 755 ms |

## Checks

**The current defaults (patches `0001`–`0019`, 2 × 131,072, fp8 KV, DFlash2, vision, states on disk)**, on one Spark on 2026-11-26
([receipts](bench/spark-run-20261126/README.md); the image also carried `0020`–`0026`, lanes off):

- **`tools/exact.py`**: **PASS** with no pad and at 3k. Drafted equals `"draft": false`, and 6 concurrent replies equal serial ones.
- **`tools/prompt_reuse.py`** (30k, 4 follow-up turns, another conversation in between): **PASS**, worst follow-up 99.4% reused.
- **`tools/features.py`**: 12/14. All structured-output and `tool_choice` checks pass, as do a corrupt image giving a 400 and a
  drafted image reply equalling `"draft": false`. The two vision colour checks returned empty content under a 16-token cap: the
  model reasons first. That cap is now 2,048, and those two checks have not been rerun.
- **Peak memory:** `MemAvailable` fell by 41.8 GiB and 49.9 GiB in two boots that ran these checks and the 128k prefill.

**DFlash2 (patches `0001`–`0018`, `DRAFTER=dflash2`)**, with the defaults (`CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `DRAFTS=1`). The image
`./start.sh` built served the live checks:

- **`tools/exact.py`**: **PASS** at pads 0, 3k and 24k (551–594 drafts accepted). Serial drafted, `"draft": false` and 6-way concurrent replies
  match on every prompt. The live image repeated it at 3k.
- **`tools/prompt_reuse.py`** (30k, 4 follow-up turns, another conversation in between): **PASS**. Each follow-up reused 99.4%–99.5% of its
  prompt, with a 0.22 s warm prefill.
- **200k-token needle** (199,951 prompt tokens, `max_tokens=512`): **PASS** at 10% and 90% depth (`warm-library-4557`). These ran with the
  prompt resumed from the cache, which drafts from a context that starts mid-prompt.
- **Structured output and `tool_choice`** (live image): **PASS**, 10/10. Covers `json_schema` (greedy, sampled, thinking off), `json_object`,
  `required`, a named function, `auto`, `guided_regex` and `guided_choice`.
- **Vision** (live image, `VISION=1`): **PASS**. Three PNGs drafted equal `"draft": false`, and 4 requests at once equal serial ones.
- Startup smoke test and the Derisked readiness check: **PASS**.

**DSpark (patches `0001`–`0017`)**, the same defaults:

- **`tools/exact.py`** (pads 0, 3k and 24k): **PASS**. Every prompt matches across serial drafted, `"draft": false` and 6-way concurrent replies
  (494–503 drafts accepted).
- **`tools/prompt_reuse.py`** (30k, 4 follow-up turns, another conversation in between): **PASS**. Each follow-up reused 99.4%–99.5% of its prompt
  from the cache, with a 0.22 s warm prefill.
- **200k-token needle in a haystack** (199,951 prompt tokens, `max_tokens=512`): **PASS** at 10%, 50% and 90% depth (`warm-library-4557`; ~106 s
  cold prefill).
- **Structured output and `tool_choice`**: **PASS**, 10/10, with valid JSON and tool calls across greedy and sampled draws, and no HTTP 500s.
  Covers `json_schema`, `json_object`, `guided_regex`, `guided_choice`, `required` and a named function.
- **Vision**: **PASS**, 17/17.
  - PNG: solid colour, RGBA alpha, two images in order, and a 3000×2000 image (downscaled).
  - JPEG: text, shapes and EXIF orientation. WebP: text.
  - Images with `json_schema` and with `tool_choice: "required"`.
  - Streamed replies equal whole ones, drafted equal `"draft": false`, 4 requests at once.
  - A corrupt image returns HTTP 400.
- **Lossy defaults, task accuracy:** a 734-problem set covers arithmetic, word problems, code tracing, logic, record lookups at 8k–120k tokens
  and two-record sums. It scored **732/734** with `TF_INKLING2_PREFILL_TC=1` and with `0`. Teacher-forced top-1 agreement of the tensor-core
  prefill with the exact path is 96.3%.

## Requirements

- An **NVIDIA DGX Spark** (or another GB10 system with 128 GB of unified memory, aarch64), with nothing else large in memory.
  - Loading takes about 25 GiB, and each stream holds its whole window from the start.
  - `start.sh` stops before launching when `MemAvailable` is below what the settings need (`MEM_NEED_GIB`; `MEM_CHECK=0` starts anyway).
- Docker with the NVIDIA Container Toolkit, and your user in the `docker` group.
- ~40 GB free under Docker's root for the image, plus ~20 GB for the checkpoint and ~1 GB for the DFlash2 draft model.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold.git
cd Inkling-2-Mini-DGX-Spark-TensorFold
./start.sh
```

On the first run, `./start.sh` does five things:
1. Downloads `thinkingmachines/Inkling-2-Mini` (~18.8 GB) and `incoai/Inkling-2-Mini-DFlash2` (~1 GB), both resumable.
2. Builds `tensorfold-inkling2:c4-dc2ca8a`, with the kernels embedded in the binary for `sm_121`.
3. Verifies the checkpoint's and the draft model's SHA-256 once.
4. Launches the server on port `8888`.
5. Runs a smoke test.

Override settings with environment variables or a `.env` file next to `start.sh`.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Inkling-2-Mini",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'
```

```bash
./start.sh restart                  # restart (validates args before stopping)
RERISKED=1 ./start.sh restart       # serve the local Inkling-2-Mini-Derisked folder
PARALLEL=4 ./start.sh restart       # more streams, less prompt cache
./stop.sh                           # stop the container
docker logs -f inkling2-mini-tf     # follow server logs
curl -s http://<spark-address>:8888/health
```

- `FOREGROUND=1 ./start.sh` stays attached.
- `DRY_RUN=1 ./start.sh` prints the `docker run` command without changing anything.
- Extra arguments pass straight to `tensorfold-native serve` (`./start.sh restart --context 131072`).

## Engine base: this recipe against TensorFold main

TensorFold main now serves Inkling 2 Mini too (#4387, a clean-room engine, CUDA at one rank and Metal). This recipe stays on Jürgen
Schmied's engine (PR #4441, `dc2ca8a1`, closed unmerged) with `0001`–`0019`. Main's CUDA path, read at `920fe22c` (2026-11-26), lacks
three of the defaults:

| Default | This recipe | main on CUDA (`zig/src/families/inkling2/cuda_native.zig`) |
| --- | --- | --- |
| Drafter | DFlash2 (`0018`) or DSpark (`0002`) | refused: `Inkling2DSparkPending`, no native draft engine (DSpark drafts on Metal only) |
| Vision | `inkling2_vision` (`0016`) | refused: `Inkling2LoopTextOnly`, text only |
| fp8 KV | `--kv-cache fp8` (`0003`) | no `kv8` option: bf16 caches only |
| States on disk | `--prompt-cache-disk` (`0019`) | no kept view: `--prompt-cache-disk` refused |
| FP8 attention, tensor-core prefill | `0012`, `0015` | absent |
| Structured output, forced tool calls | `0013` | HTTP 500 ("serves only unconstrained, non-branching text drafts") for `json_schema`, `json_object`, `guided_*` and `tool_choice` required or named |

Mia's patches do not port onto main cheaply: 17 of the 18 edit Jürgen's `mini_*.zig` files, which main replaced with `loop_*.zig`. The
compute lane (`0020`–`0024`) lives in TensorFold's shared lane host and HTTP server, which `dc2ca8a1` shares with main up to their 2026-11-21 fork
point.

Measured on one Spark on 2026-11-26 (2 × 131,072, the same checkpoint; receipts and method in
[`bench/spark-run-20261126/`](bench/spark-run-20261126/README.md)). Decode is in tok/s, summed over streams; prefill is
prompt tokens / time to first token:

| | main `920fe22c` | this engine, main's settings (no drafter, bf16 KV, text only) | this recipe's defaults (DFlash2) |
| --- | ---: | ---: | ---: |
| decode, sampled prose ×1 / ×2 | 25.7 / 41.2 | 40.5 / 65.0 | **54.9 / 89.0** |
| decode, sampled code ×1 / ×2 | 25.6 / 39.5 | 40.4 / 62.2 | **72.0 / 101.2** |
| decode, greedy code ×1 / ×2 | 25.7 / 47.2 | 43.6 / 83.2 | **105.6 / 181.3** |
| prefill 8k / 32k / 128k | 256 / 256 / 251 | 3,280 / 3,080 / 2,560 | 3,085 / 2,842 / 2,144 |

Engine against engine, this one decodes 1.6–1.8× faster and prefills 10–13× faster. With its defaults it decodes 2.1–4.1×
faster than main.

## KV cache & memory budget

On a 128 GB DGX Spark with the defaults (`CONTEXT=131072`, `PARALLEL=2`, `KV_DTYPE=fp8`, `DRAFTS=1`, `VISION=1`,
`PROMPT_CACHE_DISK=1`, `TENSORFOLD_MEMORY_RESERVE_GIB=9`). These figures are worked from the engine's own logs under the previous
8 × 262,144 defaults. Measured on 2026-11-26, the 2 × 131,072 boots took 41.8–49.9 GiB of `MemAvailable` at their peak. That is
below the ~62 GiB worked out here, because the RAM prompt cache only fills as it is used.

- **Weights and scratch:** ~23.1 GiB: `17.60 GiB` target + `1.67 GiB` DFlash2 + `0.59 GiB` vision, plus scratch. That is the logged
  `model 59.83 GiB` less its 36.7 GiB of streams.
- **One stream:** **2.53 GiB**.
  - The 8 whole-context visits' fp8 K and V: 8 × 2 × (1,024 + 8 scales × 4) = 16,896 bytes a token, so 2.06 GiB for 131,072 tokens.
  - Plus 0.47 GiB a stream that does not grow with the window: the 40 sliding visits' 1,536-row rings, the 48 visits' conv states and
    DFlash2's 40 MiB ring. That is the logged `4.59 GiB` a stream at 262,144 tokens less its 4.13 GiB of whole-context K/V.
- **Active KV slots:** 2 × 2.53 = **5.1 GiB** (262,144 active tokens), down from 36.7 GiB for 8 × 262,144.
- **Engine total before the prompt cache:** ~28.2 GiB.
- **Prompt cache in RAM:** on the Spark's unified memory the engine gives it half the room left after the reserve, the weights and the
  streams. The old defaults logged an 18.5 GiB cache, which puts that room at ~37 GiB with 36.7 GiB of streams. With 31.6 GiB fewer
  stream bytes the room is ~68.6 GiB, and the cache **~34 GiB**. Peak engine memory is then ~62 GiB, the rest of the Spark is left free,
  and `start.sh` checks for `mem_need_gib` = 46 GiB of `MemAvailable` before it starts.
- **Prompt cache on disk:** with `PROMPT_CACHE_DISK=1` the states the RAM cache evicts, and each conversation's latest state, go to
  `SESSION_DIR` (default `$HOME/.cache/tensorfold-inkling2/sessions`, mounted at `/sessions`). Keep it on the NVMe. The store's key binds
  the checkpoint, the drafter, the tokenizer and template files, the KV form (fp8 or bf16, ring span, row width, visits) and a probe
  prefill's state, so a state written under other settings is never restored. A state costs its cache's bytes: 16,896 a token of fp8 K/V
  (2.06 GiB for a full 131,072-token window), plus its sliding rings' tails and conv states.
- **`DRAFTER=dspark`:** 1.54 GiB of drafter weights and 4,096 more bytes a token a stream: DSpark's global layer keeps a bf16 K/V row
  for every position in the window.

Every stream reserves its full `CONTEXT` window at startup. More streams (`PARALLEL=4`, 10.1 GiB of slots) or longer windows
(`CONTEXT=262144`, 4.59 GiB a stream) trade prompt cache for them. For exact serving:
- `KV_DTYPE=bf16` for bf16 KV caches (32,768 bytes a token: 4.47 GiB a stream at 131,072);
- `TF_INKLING2_ATTN_FP8=0` for bf16 attention projections;
- `TF_INKLING2_PREFILL_TC=0` for bf16 prefill experts.

## Configuration

All defaults live in [`scripts/config.sh`](scripts/config.sh). Override them from the environment (`PARALLEL=4 ./start.sh restart`) or from
`.env` (the environment wins):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RERISKED` | `0` | `0`: upstream `Inkling-2-Mini` (auto-fetched); `1`: local `Inkling-2-Mini-Derisked` |
| `MODEL_DIR` | `$HOME/models/hf/Inkling-2-Mini` | checkpoint folder, mounted read-only at `/models/ink` |
| `PARALLEL` | `2` | concurrent request slots (each gets a full `CONTEXT` window) |
| `CONTEXT` | `131072` | prompt + reply window a stream (up to `1048576`) |
| `PROMPT_CACHE_DISK` | `1` | `1`: keep prompt and conversation states on disk too (`--prompt-cache-disk /sessions`); `0`: RAM only |
| `SESSION_DIR` | `$HOME/.cache/tensorfold-inkling2/sessions` | the disk states' folder (put it on the NVMe), mounted at `/sessions` |
| `COMPUTE_LANE` | `0` | `1`: the compute lane (`TF_INKLING2_COMPUTE_LANE=1`, see [Compute lane](#compute-lane)); `tool-full`: also the in-model runner (needs `RUNNER=1`) |
| `RUNNER` | `0` (`1` with `tool-full`) | build the runner image (`…-runner` tag): clang-18, lld-18, the runner circuit and graph bank (~172 MB, sha256-checked) |
| `KV_DTYPE` | `fp8` | target KV cache: `fp8` or `bf16` (exact) |
| `DRAFTS` | `1` | `1`: draft with `DRAFTER`; `0`: `--no-drafts` |
| `DRAFTER` | `dflash2` | `dflash2`: the DFlash2 draft model (`--drafter /models/draft`); `dspark`: the checkpoint's DSpark head (`--drafter /models/ink`) |
| `DRAFTER_DIR` | `$HOME/models/hf/Inkling-2-Mini-DFlash2` | DFlash2 folder (auto-fetched), mounted read-only at `/models/draft` |
| `VISION` | `1` | `1`: Inkling 2 vision tower (`--vision`, ~0.59 GiB, PNG/JPEG/WebP); `0`: text only |
| `VISION_URLS` | `0` | `1`: also fetch http(s) image URLs (`--vision-urls`); `data:` URLs always work |
| `VISION_MAX_IMAGES` / `VISION_IMAGE_TOKENS` | unset | images a request (engine default 4, up to 100) and an image's token budget (default 4,096) |
| `THINKING` / `REASONING_EFFORT` | `1` / unset | thinking toggle and an optional default effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `0`–`0.99`) |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `1.0` / `0` | default sampling (a request's own values win) |
| `MAX_TOKENS` | `32768` | reply budget when a request sets none |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | API listen address |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `9` | memory kept free when sizing the KV and prompt-cache pool |
| `TF_INKLING2_ATTN_FP8` | `1` | FP8 attention projections (`0` for exact bf16) |
| `TF_INKLING2_PREFILL_TC` | `1` | Blackwell native FP4/MXFP8 prefill tensor cores (`0` for exact bf16 MMA) |
| `NATIVE_FP_DECODE` | `1` | build with `-Dnative-fp-decode` (sm_121a's native FP4/FP8 decode in the expert kernels; exact) |

### API notes

- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Messages format), `/v1/models`, `/tokenize`,
  `/detokenize`, `/health`, `/metrics`.
- **Vision:** on by default (`VISION=1`, `--vision`). It takes PNG, JPEG and WebP `image_url` data URLs: 4 images a request by default, up to
  100 with `VISION_MAX_IMAGES`. `VISION_URLS=1` also allows http(s) image URLs. An image that can't be decoded is an HTTP 400.
- **Structured output & tool calling:** `response_format` (`json_schema`, `json_object`), `guided_regex`, `guided_choice`, and `tool_choice`
  (`auto`, `required` or a named function). `TF_TOOL_CALL_GUARD=1` hides well-formed calls to tools the request did not offer, and
  `TF_TOOL_CALL_DEBUG=1` logs rejected envelopes.
- **Dashboard:** pass `--dashboard`. With `TENSORFOLD_API_KEY` set, add `--dashboard-open` so a browser can open `/dashboard` and `/stats`
  without the key.
- **Thinking:** the model reasons before answering even at `reasoning_effort: "none"`, so give replies enough `max_tokens`. Per request,
  `"chat_template_kwargs": {"reasoning_effort": "low"}` adjusts it.
- **Exact serial reference:** `"draft": false` on any request turns drafting off. Drafted replies match `"draft": false` token for token.
- **Admission:** when all `PARALLEL` slots are full, new requests are rejected rather than queued.
- **Calls to tools not offered** (`0024`): a whole call block to a plain function name with object arguments goes out as a tool call even
  when the request did not offer that tool, as vLLM returns it, so the client can answer that it has no such tool. With
  `tool_choice: "none"` or no tools offered it stays content; `TF_TOOL_CALL_GUARD=1` hides such calls instead.

## Compute lane

Off by default. With `COMPUTE_LANE=1` (`TF_INKLING2_COMPUTE_LANE=1`), a request that offers the `code_interpreter` tool has the model's
calls to it run by the server rather than the client:

1. The engine holds the stream at the call block's end-of-sampling token instead of finishing it.
2. The server runs the call's `code` in a sandboxed `python3` (Landlock and seccomp, no network, a memory and time cap, a fresh
   directory) and renders the result as Inkling's own tool turn (a `<|message_tool|>` turn under the tool's name, as the chat template writes
   it).
3. Those tokens are forced into the same stream, and the model goes on with its answer. The client gets one reply, with the call and its
   result in it.

A request without `code_interpreter` in `tools`, or with any `stop` strings, is served exactly as with the lane off. Knobs (all
`TF_INKLING2_COMPUTE_LANE_*`, passed into the container):

| Knob | Default | Meaning |
| --- | --- | --- |
| `TF_INKLING2_COMPUTE_LANE` | unset | `1` turns the lane on (`COMPUTE_LANE=1` sets it) |
| `_DEFAULT` | `0` | `1`: every thinking chat gets `code_interpreter`, whatever it sends (not with `stop`, `logprobs`, structured output or `max_tokens` under 1,024) |
| `_WORKERS` | `2` | sandboxes running at once |
| `_MAX_MS` / `_MEM_MB` | `10000` / `1024` | a call's wall time and memory cap |
| `_MAX_CALLS` / `_MAX_RESULT_TOKENS` | `8` / `512` | calls a reply, and a result's length before it is cut |
| `_CPUS` / `_PYTHON` | unset / `/usr/bin/python3` | the sandboxes' CPU list, and the interpreter (NVIDIA's container ships one) |

**Where the lane lives.** Everything but the sandbox is backend-agnostic: the hold is a field of the engine's request
(`Request.hold_eos`), the resume is an `engine_api` entry the shared lane host implements (`resume_held`), and the parsing, the tool turn
and the forcing live in the HTTP server (`compute_turns.zig`, `compute_segments.zig`). No `mini_*.zig` file changes, so a Metal build gets the
same lane from the same patches. The sandbox (`compute_sandbox.zig`) needs Linux's Landlock and seccomp. Elsewhere it reports "Linux only",
the boot self-test (`print(1+1)` in the sandbox) fails, and the server refuses to start with the lane on. The same self-test fails the boot
when a container's seccomp profile blocks Landlock.

### Tool-full

`COMPUTE_LANE=tool-full` adds the in-model runner (`0025`–`0026`), the DeepSeek V4.1 kit's AB3 "tool-full" arm:

- The model's code cells inside its thinking (fenced code, REPL lines, inline arithmetic and traces) are run on the runner circuit with the
  graph-v7-powmod bank, and their values are written back into the thinking as it streams.
- `code_interpreter` calls run on the runner first (S3) and in the sandbox when it abstains.
- Each reply's `runtime.compute_executed` counts what ran.

It needs the runner image: `RUNNER=1` (which `tool-full` sets) builds `tensorfold-inkling2:c4-dc2ca8a-runner`. That image adds:

- the circuit frontend's boot check (`tf-runner-frontend-check`, also run as a build step against the shipped circuit);
- clang-18 and lld-18 for the RV32 programs;
- the assets [`config/runner-assets.*`](config/) pin.

`start.sh` sets the profile's `TF_INKLING2_RUNNER*` and `TF_INKLING2_COMPUTE_LANE_RUNNER*` knobs, each unless already set, so any one can
be overridden from the environment or `.env`. `TF_INKLING2_COMPUTE_LANE_DEFAULT=1` makes it hold for every thinking chat. With
`TF_INKLING2_RUNNER` unset, nothing of the runner opens.

**Portability.**
- Linux-only: the sandbox; runner thread placement and niceness (`sched_setaffinity` and `setpriority`, a no-op elsewhere); the clang
  forkserver.
- Portable: the scanner, holds and delivery, the circuit and graph backends, the frontend, receipts, guides and the attach hook. The
  attach hook is `Engine.attachRunner`, which the shared lane host implements.
- On macOS, S3 has no sandbox fallback.

**Against TensorFold main** (`920fe22c`): only `0020` (the sandbox) applies as is. `0021`–`0023` need main's later `engine_api.zig` and
`lane_host.zig`, `0024` patches `inkling_reply.zig`, which main does not have (its Inkling replies go through `loop_*` code), and
`0025`–`0026` build on all of them.

## Patches (`patches/0001`–`0026`)

Applied in order onto TensorFold `dc2ca8a1` (PR #4441) by `scripts/prepare.sh`:

| Patches | Summary |
| --- | --- |
| `0001`–`0004` | Drafted verify windows with exact conv rollback and taps; the CUDA DSpark drafter (`mtp.safetensors`); `--kv-cache fp8`; the Inkling 2 reply, reasoning and tool parser (PR #4124) |
| `0005`–`0009` | Kernel speedups, all exact: tiled loop mix, batched DSpark holds, FP8 draft head and layers, a 2-warp NVFP4 expert split, a certified lazy unembedding head, warp top-k, fused sliding prompt attention and multi-row projections |
| `0010`–`0012` | vLLM-compatible `reasoning_effort` and `reasoning` field; bf16 decode attention prefetch (`cp.async`); opt-in FP8 attention projections (`TF_INKLING2_ATTN_FP8=1`) |
| `0013` | Structured output (`response_format`, `guided_*`) and `tool_choice` (`required`, a named function) on CUDA: grammar-masked greedy and sampled draws, vLLM-parity Inkling framing and tool-choice prompts, and `libtfgrammar.so` built from the patched tree with xgrammar v0.2.8's compile cache off |
| `0014` | Exact speedups: 4,096-row prompt passes and a 64-entry CUDA graph cache (`wide_graphs`) for drafted and multi-row verify rounds (`TF_INKLING2_WIDE_GRAPHS=0` for the old path) |
| `0015` | Blackwell native block-scaled tensor-core prefill experts (`sm_121a`, `TF_INKLING2_PREFILL_TC=1`): prompt activations quantized once a layer (38 MiB of scratch), then `kind::mxf4nvf4` (layers 0–19) and `kind::mxf8f6f4` (20–23) run directly over TensorFold's packed weights |
| `0016` | Native CUDA vision encoder (`inkling2_vision`): `Inkling2ImageProcessor` (PNG/JPEG/WebP, EXIF orientation, 8×8 checkerboard alpha, `-1.0` padding to 40×40 tiles), the 23-layer ViT with 2D pairwise axial RoPE, LayerScale and bilinear position embeddings, `vision_kernel<64, 8>` tensor-core attention, the 2×2 spatial merge and GELU projector, and image embeddings spliced into the prompt (`--vision`) |
| `0017` | Server fixes: the reply grammar is freed when a stream fails to start (`lane_stream.zig`); opt-in `TF_TOOL_CALL_GUARD=1` and rejected tool-envelope logs (`TF_TOOL_CALL_DEBUG=1`); `--vision-max-images` up to 100; `--dashboard-open` |
| `0018` | DFlash2 drafter (`mini_dflash.zig`, `inkling2/dflash.cu`): `--drafter` on a DFlash2 folder loads it in place of DSpark. Committed taps go through `fc` and the hidden norm into 5 layers' 2,048-position K/V rings; one pass embeds the pending token and 7 masks and runs the two-tap dynamic convs, non-causal block attention and SwiGLU (layers as FP8 copies); an NVFP4 copy of the target head gives each row's top 16, and a GPU candidate selector walks the path (keyed draws for sampled requests). A stream alone gives the round planner each draft's chance, the selector's probability calibrated by what landed. After TensorFold's GLM-5.3-Flash CUDA DFlash2 and [tonyd2wild](https://github.com/tonyd2wild)'s [Inkling DFlash2 patch](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark) |
| `0019` | Prompt and conversation states on disk (`--prompt-cache-disk`, CUDA): `mini_disk.zig` describes `mini_snap`'s states to TensorFold's kept-state disk store (`native/cuda_kept.zig`): whole caches as rows from position 0, sliding rings as their tails, conv states, a blank state to read into, and the KV form the store's key binds |
| `0020`–`0022` | The compute lane's sandboxed executor (`core/compute_sandbox.zig`: Landlock, seccomp, rlimits, kept niceness above 10, the `__compute_sandbox` trampoline) and the engine half: `Request.hold_eos`, `resume_held` in the shared lane host, held results forced in as prompt tokens. Ported from the DeepSeek V4.1 Spark kit's patches 0197–0201 |
| `0023` | The compute lane for any lane-hosted family, read from the reply format: Inkling 2's call blocks (`chat_inkling.read`), its tool turn rendered by name from the chat template, `TF_INKLING2_COMPUTE_LANE*` |
| `0024` | Inkling 2: a call to a tool the request did not offer goes out as a tool call, as vLLM returns it (after the DeepSeek kit's 0206–0207) |
| `0025` | The in-model runner, S3 and the compute default, from the DeepSeek V4.1 kit's tool-full range (`3adb042bb..01a62132c`) without its DeepSeek family wiring: the scanner and held cells, the RV32 frontend and clang forkserver, the CPU circuit and graph backends, runner-first `code_interpreter` (S3), `_COMPUTE_LANE_DEFAULT`, and `Engine.attachRunner` in the shared lane host |
| `0026` | The runner and S3 for any lane-hosted family (`server/runner_boot.zig`): `TF_INKLING2_RUNNER*` and `TF_INKLING2_COMPUTE_LANE_RUNNER*` for Inkling 2's replies; the image's boot check fails closed when either runner knob is set |

## Tools

`tools/` talks to the running server (`API_URL`, default `http://127.0.0.1:8888`, or `PORT`):

| Script | What it checks |
| --- | --- |
| `tools/exact.py` | greedy replies: drafted, `"draft": false` and concurrent must match token for token |
| `tools/prompt_reuse.py` | multi-turn prompt cache reuse |
| `tools/needle.py` | passphrase retrieval in a ~200k-token haystack |
| `tools/speed.py` | decode (prose and code, sampled and greedy) at 1 to N streams and prefill at 8k, 32k and 128k, timed at the client |
| `tools/features.py` | structured output, `tool_choice` and vision, one PASS/FAIL line each |
| `tools/lane_ab.py` | the compute-lane A/B: generated arithmetic with exact answers under arms P, D, L and tool-full |
| `tools/bench_streams.py` | needs a `tools/bench.py` that this repository does not have; use `tools/speed.py` |
| `tools/client.py` | CLI chat client and shared helpers |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      config.sh, prepare.sh, fetch.sh, banner.sh, runner_assets.py
patches/      0001–0026 applied onto TensorFold at TF_REF
config/       the runner image's asset pins (runner-assets.release, runner-assets.sha256)
tools/        verification scripts
CREDITS.md    upstream credits
NOTICE        third-party notices
```

## License

Apache-2.0, see [`LICENSE`](LICENSE). Model weights are not part of this repository: `thinkingmachines/Inkling-2-Mini` is Apache-2.0 with Thinking
Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/), and `Inkling-2-Mini-Derisked` is subject to
the upstream terms and notices.

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
(`nvcr.io/nvidia/pytorch:26.07-py3`). The NVIDIA software in it is governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the
[Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
By running the image you accept them. The Apache-2.0 license covers this repository's scripts and patches only. The image also carries the licenses of xgrammar,
libjpeg-turbo and libwebp, which it builds; a `RUNNER=1` image also carries Qwenputer's MIT notice for the graph bank (see [`NOTICE`](NOTICE)).

## Credits

- **Recipe & patches (`0001`–`0003`, `0005`–`0018`):** created and built by **[MiaAI-Lab](https://github.com/MiaAI-Lab)** (Mia's AI Lab). That
  includes:
  - CUDA DSpark and DFlash2 drafting, `--kv-cache fp8`, and decode and prefill kernel optimizations;
  - 4,096-row passes, wide verify CUDA graphs, and Blackwell native FP4/MXFP8 prefill tensor cores;
  - structured output and `tool_choice`, and native CUDA `inkling2_vision` image support;
  - full TensorFold serving support for `Inkling-2-Mini-Derisked`, whose upstream
    [deployment kit](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked/blob/main/deployment-kit/AGENT_INSTRUCTIONS.md) only targets
    vLLM.

  Benchmarked with [sparkDash](https://github.com/MiaAI-Lab/sparkDash/).
- **Spark defaults, states on disk (`0019`), the compute lane and tool-full (`0020`–`0026`):** by [jayleaton](https://github.com/jayleaton)
  on Mia's recipe; the lane and tool-full are ported from the DeepSeek V4.1 Spark kit.
- **Engine:** [TensorFold](https://github.com/ashhart/TensorFold) and `tensorfold-native` by Ash Hart ([ashhart](https://github.com/ashhart))
  and contributors.
- **Inkling 2 Mini on CUDA:**
  - Jürgen Schmied ([jschmied](https://github.com/jschmied), [PR #4441](https://github.com/ashhart/TensorFold/pull/4441),
    [#4640](https://github.com/ashhart/TensorFold/pull/4640));
  - [ajensenwaud](https://github.com/ajensenwaud): the Inkling 2 kernels and the reply parser, patch `0004` (#4124–#4128);
  - [BlivionIaG](https://github.com/BlivionIaG), [sethforprivacy](https://github.com/sethforprivacy), [gilby](https://github.com/gilby) and
    [akol1](https://github.com/akol1).
- **Blackwell prefill & chunking ideas:** [Basalt](https://github.com/jesdga95/basalt) by [jesdga95](https://github.com/jesdga95).
- **Reply-grammar leak fix (`0017`):** SvangenStudios' PR #28.
- **DFlash2 drafter (`0018`):** [tonyd2wild](https://github.com/tonyd2wild)'s independent Inkling 2 Mini DFlash2 patch
  ([Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), Apache-2.0): the NVFP4 draft-head copy and
  the lone-stream draft chances calibrated by what landed.
- **Model & derivative:** [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines;
  `Inkling-2-Mini-Derisked` by Blackfrost.
- **Runtime dependencies:** [xgrammar](https://github.com/mlc-ai/xgrammar) (`libtfgrammar.so`, Apache-2.0),
  [libjpeg-turbo](https://libjpeg-turbo.org) and [libwebp](https://chromium.googlesource.com/webm/libwebp) (BSD-3-Clause),
  [Zig](https://ziglang.org), and NVIDIA CUDA.

Full list in [`CREDITS.md`](CREDITS.md).
