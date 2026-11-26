# Inkling 2 Mini on Mac Studio (Metal / TensorFold)

Native Apple Silicon (Metal) recipe for serving [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`rev 7a67ae2314b4`) with [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, speculative drafting (`incoai/Inkling-2-Mini-DFlash2`), vision (`--vision`), and structured output / tool calling (`libtfgrammar.dylib` via xgrammar `v0.2.8`).

The engine is TensorFold's native Metal Inkling 2 Mini — upstream `main` at `920fe22c9` — plus two upstream PRs carried as `mac/patches/`: the Metal drafter window-fit fix ([#4622](https://github.com/ashhart/TensorFold/pull/4622), sethforprivacy) and the native Metal DFlash2 drafter ([#4917](https://github.com/ashhart/TensorFold/pull/4917), cwschroeder). None of the root recipe's CUDA engine or patches is built here; from Mia's AI Lab's recipe the Mac build takes the repository layout and the shared test tools (`tools/exact.py`, `tools/prompt_reuse.py` and the rest). Full credits are in [`../CREDITS.md`](../CREDITS.md); see [`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE) for licensing.

## Quick Start

Requirements: Apple Silicon Mac, Xcode Command Line Tools (`xcrun`), `zig` `0.17.0`, `git`, `curl`, `python3`.

```bash
# Fetch + SHA-256 verify weights, build tensorfold-native + libtfgrammar.dylib, serve on :8888, run smoke test
./mac/start.sh

# Stop the server
./mac/stop.sh
```

Defaults:
- **Context**: `131072` (128k tokens)
- **Streams (`--parallel`)**: `2`
- **Drafter**: `dflash2` (`incoai/Inkling-2-Mini-DFlash2`, rev `4f9088ea66c2`); set `DRAFTER=dspark` for the checkpoint's built-in DSpark head or `DRAFTS=0` to disable
- **Vision**: enabled (`VISION=1`; set `VISION=0` for text-only)
- **Structured output / tools**: enabled via `libtfgrammar.dylib` (built automatically from xgrammar `v0.2.8`)
- **NVMe / disk offload**: **not supported** on TensorFold's Metal `inkling2_loop` path (`--learn-conversations` is explicitly refused by `zig/src/native/inkling2.zig`). In-memory exact prompt reuse (`--prompt-cache-gib`, default automatic budget) is enabled and verified below.

## Pinned Engine & Patch Selection

- **Base commit**: `ashhart/TensorFold` `main` at `920fe22c97995d3dae9bed3e320f08aaa64c25ae`
- **Minimal patch set (`mac/patches/`)**:
  1. [`0001-metal-window-fit.patch`](patches/0001-metal-window-fit.patch) — upstream PR `#4622`: fixes Metal drafter window-fit accounting (previously priced each drafted token at 128x its actual bytes, refusing `--context 131072 --parallel 2` on a 96 GiB Mac).
  2. [`0002-metal-dflash2-drafter.patch`](patches/0002-metal-dflash2-drafter.patch) — upstream PR `#4917` (rebased cleanly onto `920fe22c9` + `#4622`): adds native Metal `DFlash2DraftModel` support alongside DSpark.

### Main-Only vs. Main + PRs (Apple M5 Ultra, 96 GiB RAM)

Measured on Studio-01 (Apple M5 Ultra, 96 GiB unified memory, `--parallel 2 --vision`, greedy `temperature=0`, 192 tokens):

| Build | Drafter | Max Context (2 streams) | Load Time | Prose 1s | Prose 2s total (per stream) | Code 1s | Code 2s total (per stream) | RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main` (`920fe22c9`) | DSpark | 48,345 (refuses 128k) | 13.9 s | 36.5 tok/s | 41.5 (20.7) tok/s | 46.9 tok/s | 67.4 (33.7) tok/s | 19.80 GiB |
| `main` + `#4622` | DSpark | 131,072 | 14.1 s | 36.3 tok/s | 41.4 (20.7) tok/s | 46.7 tok/s | 66.1 (33.0) tok/s | 19.80 GiB |
| `main` + `#4622` + `#4917` | **DFlash2** | **131,072** | **5.8 s** | **42.2 tok/s (+16%)** | **61.8 (30.9) tok/s (+49%)** | **56.7 tok/s (+21%)** | **90.3 (45.2) tok/s (+34%)** | **20.02 GiB** |

Why this set: `#4622` is required to serve 2 × 128k streams with a drafter on 96 GiB, and `#4917` delivers +16% to +49% decode throughput and >2x faster startup while maintaining 100% exact greedy parity.

## Verification Receipt (Studio-01, Apple M5 Ultra, 96 GiB)

All checks run against `./mac/start.sh` (`920fe22c9` + `0001` + `0002`, DFlash2, 2 streams, 128k context, vision on):

| Check | Result | Details |
| --- | --- | --- |
| `tools/exact.py` | **PASS** | 6/6 prompts, 709 drafts accepted; identical text & token SHAs across serial, `draft: false`, and concurrent (`max_batch_size 2`) |
| `tools/prompt_reuse.py 8000 3` | **PASS** | Cold 9,608 tok (26.75 s) → warm follow-ups 96.8% / 97.7% / 95.9% cached (0.74–1.34 s) |
| Structured output (`json_schema`, strict) | **PASS** | Valid JSON matching schema (`city`, `country`, `population_millions`) via `libtfgrammar.dylib` |
| Tool calling (`tool_choice: required`) | **PASS** | Emitted `get_weather({"city": "Sydney"})` |
| Vision (`data:image/png;base64,...`) | **PASS** | Identified 32×32 solid red PNG (`'Red'`) |
| Prefill 8k (8,182 tok, 0 cached) | **367 tok/s** | 22.27 s prefill (22.79 s wall) |
| Prefill 32k (32,797 tok, 0 cached) | **330 tok/s** | 99.48 s prefill (100.17 s wall) |
| Prefill 128k (126,000 tok, 0 cached) | **287 tok/s** | 439.38 s prefill (440.10 s wall) |
| Peak memory (RSS) | **20.26 GiB** | 29.46 GiB budgeted for 2 × 128k streams (49.0 GiB Metal state budget) |
