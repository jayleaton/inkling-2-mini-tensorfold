# Credits

This repository is a thin layer of scripts. Almost everything that makes it work was built by others. Its own work is licensed under the Apache
License, Version 2.0 ([`LICENSE`](LICENSE)). The image `scripts/prepare.sh` builds also carries TensorFold's license files from the pinned checkout.

## Model

- **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** by Thinking Machines: the looped mixture-of-experts model, its
  ModelOpt NVFP4/MXFP8 checkpoint, its chat template and its MTP (DSpark) module. Apache-2.0, with Thinking Machines'
  [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
- **[Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)** by Inco AI: the DFlash 2 block-diffusion draft model
  (two-tap dynamic convs, candidate selector) this recipe drafts with by default. Apache-2.0. Read unmodified from its pinned revision.
- **[DFlash](https://github.com/z-lab/dflash)** by Jian Chen, Yesheng Liang and Zhijian Liu (z-lab, MIT): block diffusion for speculative
  decoding, the method the draft model and its serving follow; SGLang's DFlash2 implementation is the reference for the drafter's math.
- **Blackfrost**: `Inkling-2-Mini-Derisked`, the behaviorally modified derivative this recipe serves: an edited target shard with refusals
  reduced, the MTP module unchanged. Released subject to the upstream terms and notices (its README). Not part of this repository.

## Engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the TensorFold contributors:
  the Zig engine (`tensorfold-native`), its lane rounds and drafter interface, and the Inkling 2 family code. Apache-2.0 (code before 0.6.0 also MIT).
- **Jürgen Schmied** ([jschmied](https://github.com/jschmied)): the Inkling 2 Mini CUDA engine this recipe builds on, "C4"
  ([#4441](https://github.com/ashhart/TensorFold/pull/4441)), with Mini's shapes on the attention kernels (C1, #4364) and its stored NVFP4 and
  MXFP8 experts in core (C3, #4365); and the load, attention, cross-pass, slot and snapshot fixes to the lead branch
  ([#4640](https://github.com/ashhart/TensorFold/pull/4640)).
- **ajensenwaud** ([ajensenwaud](https://github.com/ajensenwaud)): Inkling 2's reply split into reasoning, text and tool calls (#4124, patch `0004`), its CUDA attention kernels (#4125), its embedding, residual mix, conv,
  router, SwiGLU and pick kernels (#4126), and Inkling 2 Small's DSpark drafter kernels ([#4128](https://github.com/ashhart/TensorFold/pull/4128)):
  the block attention, Markov context, recurrence and vocabulary kernels this recipe's drafter adapts.
- **BlivionIaG** ([BlivionIaG](https://github.com/BlivionIaG)): Inkling 2 Mini on Metal with its bundled DSpark drafter
  ([#4337](https://github.com/ashhart/TensorFold/pull/4337)), the reference for the drafter's math, taps and folded Markov head.
- **sethforprivacy** ([sethforprivacy](https://github.com/sethforprivacy)): the DSpark fixes on Metal (FP32 kappa, the 1/sqrt(128) attention scale,
  [#4447](https://github.com/ashhart/TensorFold/pull/4447)) and Mini's serving fixes.
- **TensorFold's GLM-5.3-Flash CUDA DFlash2 drafter** (TensorFold contributors): the context rings, two-tap convs, chunked block attention,
  top-16 and selector this recipe's Inkling DFlash2 drafter (patch `0018`) ports to Mini's widths.
- **tonyd2wild** ([tonyd2wild](https://github.com/tonyd2wild)): an independent Inkling 2 Mini DFlash2 patch for TensorFold
  ([Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), Apache-2.0). Patch `0018` takes its NVFP4
  draft-head copy (`fp4Of`) and its lone-stream draft chances calibrated by the lanes' Landing table.
- **gilby** ([gilby](https://github.com/gilby)): Mini's Metal kernels, DSpark prompt cache and snapshot fixes.
- **akol1** ([akol1](https://github.com/akol1)): the plan for Inkling 2 Mini on CUDA and Metal in claimable slices
  ([#4274](https://github.com/ashhart/TensorFold/pull/4274)).
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** by the MLC-AI team (Apache-2.0): `libtfgrammar.so`, built from the patched tree over xgrammar
  v0.2.8 at its pinned commit (its license, NOTICE and dlpack's license ship in the image).
- **[libjpeg-turbo](https://libjpeg-turbo.org)** 3.1.4.1 (IJG, BSD-3-Clause and zlib licenses) and **[libwebp](https://chromium.googlesource.com/webm/libwebp)**
  1.6.0 (BSD-3-Clause) by their authors: the JPEG and WebP decoders for image inputs, built in the image (licenses under `LICENSES-codecs`).
- **SvangenStudios**: the reply-grammar leak fix in patch `0017` (their PR #28, by way of the GLM-5.3-Flash DGX Sparks recipe's patch 0007).
- **[Basalt](https://github.com/jesdga95/basalt)** by [jesdga95](https://github.com/jesdga95): the Blackwell prefill and chunking ideas behind patches
  `0014` and `0015`.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and contributors.
- **NVIDIA**: the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) the image is based on, and Model Optimizer
  (the checkpoint's format).

## Mac (Metal)

The `mac/` recipe serves the same checkpoint through TensorFold's native Metal `inkling2_loop` path: upstream TensorFold `main` pinned at
`920fe22c9` plus two upstream PRs carried as `mac/patches/` ([#4622](https://github.com/ashhart/TensorFold/pull/4622),
[#4917](https://github.com/ashhart/TensorFold/pull/4917)). Nothing of the CUDA engine or `patches/` above is built there; from this repository's
recipe the Mac build takes the layout and the shared test tools (`tools/exact.py`, `tools/prompt_reuse.py` and the rest). Every PR below is
merged into the pinned `920fe22c9` unless marked open.

- **Ash Hart** ([ashhart](https://github.com/ashhart)): Inkling 2 Mini on Metal itself, the clean-room Mini kernels with their contracts and
  native serving with admission, keepalive and prompt logprobs ([#4387](https://github.com/ashhart/TensorFold/pull/4387), "Inkling 2 Mini on
  CUDA and Metal"); the routed-experts-and-combine and router-and-loop-gate kernels, a thread an output
  ([#4894](https://github.com/ashhart/TensorFold/pull/4894), [#4912](https://github.com/ashhart/TensorFold/pull/4912)); and the serving fixes
  whose thinking-off requests reach the template as effort 0 ([#4876](https://github.com/ashhart/TensorFold/pull/4876)) and whose required tool
  calls use the template's own message format ([#4898](https://github.com/ashhart/TensorFold/pull/4898)).
- **sethforprivacy** ([sethforprivacy](https://github.com/sethforprivacy)): the Metal loader's per-read cap that lets a checkpoint span of
  16 GiB or more load ([#4435](https://github.com/ashhart/TensorFold/pull/4435); the Mac build's 18 GiB shard goes through it), the DSpark
  fixes on Metal, the 1/sqrt(128) attention scale and the recurrent step's kappa read as FP32
  ([#4447](https://github.com/ashhart/TensorFold/pull/4447)), and the DSpark overhaul the engine holds in one command buffer with drafts
  priced up to the whole block ([#4465](https://github.com/ashhart/TensorFold/pull/4465), open); the window-fit fix that prices a drafted
  token at a token's bytes, carried as patch `0001` ([#4622](https://github.com/ashhart/TensorFold/pull/4622), open).
- **cwschroeder** ([cwschroeder](https://github.com/cwschroeder)): the native Metal DFlash 2 drafter for `DFlash2DraftModel` folders such as
  Inco AI's, with its weights and config validation, its BF16 kernels and the adaptive depth timing that unlocks full draft blocks
  ([#4917](https://github.com/ashhart/TensorFold/pull/4917), open; patch `0002`).
- **BlivionIaG** ([BlivionIaG](https://github.com/BlivionIaG)): images under `--vision` on Metal, the tower, the processor and their contract
  ([#4704](https://github.com/ashhart/TensorFold/pull/4704)); the startup line's report of the asked streams' demand
  ([#4705](https://github.com/ashhart/TensorFold/pull/4705)); prompt experts, projections and loop gates on simdgroup matrices without tensor
  units, with 256-row prompt chunks, one attention-residual launch a chunk and a combine that sorts a token's routes once
  ([#4756](https://github.com/ashhart/TensorFold/pull/4756)); and decode without tensor units, 9 to about 21 tok/s, every output bit equal to
  main's ([#4919](https://github.com/ashhart/TensorFold/pull/4919)).
- **gilby** ([gilby](https://github.com/gilby)): the DSpark Metal kernel overhaul on main, batched multi-stream speculative drafting, softmax
  chances and two-pass GPU top-k, sethforprivacy's #4465 rebased
  ([#4699](https://github.com/ashhart/TensorFold/pull/4699)); the DSpark lane-state snapshots the Metal prompt cache restores
  ([#4542](https://github.com/ashhart/TensorFold/pull/4542)); and the Metal kernels in the pin: threadgroup reservations matched and load
  guards enforced ([#4981](https://github.com/ashhart/TensorFold/pull/4981)), exact-serial grouped routed experts for two rows or more
  ([#4995](https://github.com/ashhart/TensorFold/pull/4995)), ceiling-pow2 dense dispatch and guarded `bf16_matvec`
  ([#4998](https://github.com/ashhart/TensorFold/pull/4998)), the relative-bias projection fused into decode and prompt attention
  ([#5000](https://github.com/ashhart/TensorFold/pull/5000)), depth-block `inv_rms` cached with partial updates fused into AR reductions
  ([#5001](https://github.com/ashhart/TensorFold/pull/5001)), and the prompt-attention thread qualifiers
  ([#4980](https://github.com/ashhart/TensorFold/pull/4980)).
- **plotarmordev** ([plotarmordev](https://github.com/plotarmordev)): the Inkling 2 Mini load and serve refusals, mismatched head widths and
  block splits at load, out-of-vocabulary ids and expired token handles at serve
  ([#4990](https://github.com/ashhart/TensorFold/pull/4990)).
- **Nipale-ai** ([Nipale-ai](https://github.com/Nipale-ai)): the Inkling 2 family's host package, the config parsing that refuses layouts the
  engine does not run, the weight map checked against the checkpoint's headers, the layer plan, and the chat template and tokenizer checks
  ([#3578](https://github.com/ashhart/TensorFold/pull/3578)), on the `float` filter the template core needed for it
  ([#3577](https://github.com/ashhart/TensorFold/pull/3577)).
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** by the MLC-AI team: on macOS this recipe compiles upstream xgrammar `v0.2.8` itself into
  `libtfgrammar.dylib` for structured output and tool calling (Engine above; Apache-2.0, with DLPack as xgrammar carries it).

The model, its DFlash 2 draft model and the method they follow are credited under Model above: Thinking Machines'
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Inco AI's
[Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) and z-lab's DFlash.

## Recipe

- **MiaAI-Lab** (Mia's AI Lab): this repository's scripts, after its
  [Qwen3.8 Flash Next recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), whose `start.sh`, `prepare.sh` and
  `tools/` it adapts.
