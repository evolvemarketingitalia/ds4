# DeepSeek V4.1 Flash on one Strix Halo: SSD streaming + DSpark on ROCm

This fork branch (`strix-halo-v41-dspark`) is the engine we run in production on a single
AMD Strix Halo (Ryzen AI Max+ 395, `gfx1151`, 128 GB unified memory). It starts from the ROCm
port of DwarfStar and adds four things:
- faster SSD-streamed decode;
- DSpark speculative decoding ported from Metal to ROCm;
- a rewind-safe draft state;
- server-side fixes so that long conversations survive model unloads without being re-read.

Headline numbers, measured on this machine with a 12k-token prompt and greedy decoding:

| Workload | Where we started | This branch |
|---|---|---|
| Decode, no speculation (official `ds4-bench` sweep, 2k-32k context) | 10.3-10.5 tok/s | 13.8-14.0 tok/s, frontier logits byte-identical |
| Rewriting a 143-line C function (agent-style code edit), DSpark | 13.8 tok/s | **31.7 tok/s**, output identical |
| Writing new code (follow-up turn), DSpark | ~13.8 tok/s | 20.0 tok/s |
| Italian prose analysis, 256 tokens, DSpark | 13.4 tok/s | 16.6 tok/s |
| Short answer about an image, DSpark | 11.5 tok/s | 17.3 tok/s |

On the code edit the draft is accepted 97-98% of the time, 3.6-3.7 tokens per cycle. On prose it guesses
far less, so the gain is smaller. Adaptive drafting stops proposing when drafting does not pay.

## Where we started, and credits

Nothing here would exist without the work it is built on:

- **[DwarfStar / ds4](https://github.com/antirez/ds4)** by Salvatore Sanfilippo (antirez) and the ds4
  contributors: the engine, the DeepSeek V4.1 Flash support, the Q2 GGUF, SSD expert streaming and the server.
- **Donato Capitella ([kyuz0](https://github.com/kyuz0))**: the ROCm port for Strix Halo
  ([antirez/ds4#1036](https://github.com/antirez/ds4/pull/1036)), the gfx1151 kernels, and the Strix Halo
  toolbox container we build in. Our base is his `feat/rocm-deepseek41-halo` branch as of 20 Sep 2026
  (`8164e28`), merged with antirez `main` at `0aaea5a`.
- **kernelpool**: DSpark for DeepSeek V4.1 on Metal ([antirez/ds4#1073](https://github.com/antirez/ds4/pull/1073)).
  We ported its draft state, draft stages, Markov chain, confidence head, verify and rollback to ROCm with
  SSD-streamed experts. Its gguf-tools produce the DSpark support file.
- **gaineyllc / RTSAI Coder**: V4.1 live-session rewind ([antirez/ds4#1089](https://github.com/antirez/ds4/pull/1089)),
  merged here unchanged. Our multimodal rewind and persistence fixes build on it.
- **TripleSparkleAI** ([antirez/ds4#1083](https://github.com/antirez/ds4/pull/1083)): the hits-first expert launch on
  CUDA. We applied the same resident/missing split to the DSpark verify path.
- **lukeckprobierts** ([#848](https://github.com/antirez/ds4/pull/848), [#849](https://github.com/antirez/ds4/pull/849)):
  packed experts and router-lookahead prefetch. We reproduced the lookahead idea here and report below
  why it does not pay on this setup.
- **DeepSeek**: the V4.1 Flash weights and the multi-token-prediction (DSpark) weights.
- The ggml authors (MMQ kernels vendored by ds4) and AMD ROCm.

Starting point on this box, before any change, on 22 Sep 2026 in the morning:
- 7.7 tok/s decode with kyuz0's integration and a 64 GB expert cache;
- the model sat on a Crucial P310.

Part of that gap was hardware and configuration: the cache is now 72 GB and the model moved to an SN850X.
The fair baseline is kyuz0's current `main-gfx1151` on today's hardware. It measures 10.3-10.5 tok/s here, in
line with the 10.2-10.5 tok/s he documents on a Framework Desktop with a 76 GB cache. The table above uses that
baseline.

**How the work was done.** Alessandro Puricelli did this work with **Claude Opus 5.5** (Anthropic),
working as the engineering agent in Claude Code on this machine. Opus 5.5 did the DSpark port, the
verify-path kernels, the rewind-safe draft rings, the persistence fixes and the measurements behind this
page. Each commit's `Co-Authored-By` trailer records which Claude model co-authored it; three earlier kernel
commits carry Claude Fable 5.1.

## Hardware and software

- ACEMAGIC M1A Pro: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB LPDDR5X. GPU
  `power_dpm_force_performance_level=high`.
- Model on a WD Black SN850X 2 TB (PCIe 4.0 x4). An optional sparse mirror of the routed experts sits on a
  Crucial P310 2 TB (see [Expert mirror](#expert-mirror-on-a-second-nvme)).
- Ubuntu 24.04.5, Linux 7.0. ROCm SDK 10.0 in kyuz0's toolbox container (HIP 7.15.26333), built with
  `make strix-halo`.
- `DeepSeek-V4.1-Flash-Q2.gguf`: 365,713,686,528 bytes, sha256 `1ce6a8f8…6f42`.
- `DeepSeek-V4.1-Flash-Vision.gguf`.
- DSpark support file: 4,568,907,776 bytes, sha256 `1a130f0f…98fb`, generated as described below.

## What is in the branch

Our commits sit on top of kyuz0's integration, `main` and #1089. In order:

| Commit | What it does |
|---|---|
| host: `1.0f/sqrtf` instead of device-only `rsqrtf` | ROCm 7.2.4+ host build fix |
| V4.1: rewind with images, multimodal disk checkpoints, disk-aware rewind | A live session that holds images can rewind instead of rebuilding. Disk KV checkpoints carry image identities (DSVI trailer). A rewind first persists a long cut. |
| server: `DS4_SERVER_PREFILL_QUANTUM` | Prefill slice size. At 8192 the V4.1 sweep stages the experts once per slice, which made prefill +70% faster on this SSD. |
| rocm: opt-in resident/missing MoE split | Wires the existing one-token split: cached experts run while misses are read (`DS4_ROCM_SELECTED_SPLIT=1`). |
| rocm: deterministic pointer path | Fixes the split's nondeterminism. After the compaction, generic kernels read the compact table while D2D copies on the upload stream were still in flight. One upload wait after the compaction fixes it. Adds a staging ring. The converter derives the FP8 block size from the scale shape. |
| rocm: fuse bf16 rounding into epilogues (×2) | −765 kernel launches per token; byte-identical output. |
| ds4-server: keep model-generated conversations across unload and sampled-token splits | GLM tool-call spans are persisted in the disk tool map. The tool-memory restore steps over the vision trailer. A detokenized-prefix reuse tier covers sampled-token splits. Images per request go from 16 to 256. |
| rocm V4.1: same-session row batches for DSpark verify | Verify rows run as one batch. Q8, attention-output and MoE kernels read each weight once for 2-8 rows, with per-row arithmetic identical to decode. |
| V4.1 DSpark on ROCm with SSD expert streaming | The port of #1073. The draft stages stay resident in the support file (128 experts, top-3); the trunk streams its experts. |
| rocm V4.1 DSpark: rewind-safe draft rings, faster verify, expert mirror reads | See [the draft-ring bug](#the-draft-ring-bug). Adds the resident/missing split in the verify MoE, multi-row F16 (Engram) and HC kernels and a one-wave-per-row Markov kernel. Adds the optional expert mirror. |
| ds4-server: speculative decoding on the single resident session in batched mode | With `--batched-session 1` there is nothing to batch with, so speculation runs directly under the inference lock instead of being turned off. |
| rocm V4.1 verify: read each shared expert once across rows | Gate/up runs one block per unique expert, and down runs in one launch for all rows. The F32 router uses one launch for all rows. |

### The draft-ring bug

DSpark's draft attends over a ring of the last 128 trunk positions. With DSpark (#1073) combined with
live-session rewind (#1089), a rewound session kept the discarded generation in the ring. That happens after
resending a prompt or when a client edits history. On a repeated code edit it halved acceptance, from 4.8 to
2.4 tokens per cycle. The rings now keep 4096 positions with a valid range, and rewind, reset and snapshot load
update that range. Anyone merging #1073 with #1089 needs the same clamp in `ds41_graph_rewind`, on any
backend.

## Build and run

```bash
make strix-halo ROCM_ARCH=gfx1151
MODEL=/path/DeepSeek-V4.1-Flash-Q2.gguf
VISION=/path/DeepSeek-V4.1-Flash-Vision.gguf
DSPARK=/path/DeepSeek-V4.1-Flash-DSpark-Q2-1073.gguf

export DS4_ROCM_SELECTED_SPLIT=1           # resident/missing split (deterministic now)
export DS4_SERVER_PREFILL_QUANTUM=8192     # prefill slices
# export DS4_ROCM_STREAM_MIRROR=/second-nvme/DeepSeek-V4.1-Flash-Q2.experts-mirror.gguf   # optional

./ds4-server --rocm -m "$MODEL" --vision "$VISION" \
  --ssd-streaming --ssd-streaming-cache-experts 72GB --ctx 300000 \
  --batched-session 1 --host 127.0.0.1 --port 8080 \
  --kv-disk-dir /fast/ssd/kv --kv-disk-space-mb 32768 --kv-cache-min-tokens 256 \
  --kv-cache-cold-max-tokens 30000 --kv-cache-continued-interval-tokens 2048 \
  --kv-cache-boundary-align-tokens 512 --kv-cache-reject-different-quant \
  --dspark --mtp-model "$DSPARK"
```

- **Memory.** 72 GB is the largest expert cache this 128 GB box admits with a desktop running. The plan is
  86.9 GiB plus the 4.6 GB DSpark file; about 15 GB stay free while generating. Above 72 GB misses no longer
  matter: the hit rate is 98.5%.
- **Drafting.** It is adaptive by default. `--dspark-confidence 0.6` helps prose by about 3% but costs code
  about 3%, so we keep the default 0.7.

## Building the DSpark support file

The DSpark weights ship in the official DeepSeek V4.1 Flash checkpoint: the `mtp.*` tensors in shards 44-46 of
48, about 8 GB. The conversion uses #1073's gguf-tools. `tools/strix-halo/dspark41_support.py` only skips the
tokenizer metadata, so the other 45 shards are not needed.

```bash
# 1. shards 44, 45, 46 + model.safetensors.index.json of deepseek-ai/DeepSeek-V4.1-Flash (revision dba1be0a),
#    plus config.json: tools/strix-halo/dspark41_config.json is the one we used (official values under
#    text_config, quantization_config at the top; compress_ratios are placeholders, informative only here)
# 2. gguf-tools from PR #1073
git fetch https://github.com/antirez/ds4 pull/1073/head:pr-1073 && git worktree add ../ds4-pr1073 pr-1073
make -C ../ds4-pr1073/gguf-tools
# 3. convert the three DSpark stages only
python3 tools/strix-halo/dspark41_support.py --gguf-tools ../ds4-pr1073/gguf-tools \
  --hf /path/to/DeepSeek-V4.1-Flash --dspark-out DeepSeek-V4.1-Flash-DSpark-Q2-1073.gguf
```

The result has 78 tensors: 3 stages, block size 5, Markov rank 256, target layers 37-39, and 128 experts
top-3 per stage.

## Expert mirror on a second NVMe

`tools/strix-halo/mirror_experts.py` copies only the byte ranges of the routed experts (153 GB of the 366 GB
file) into a sparse file with the same size and offsets on a second drive. Reads use O_DIRECT on both sides,
and a sampled comparison runs at the end. It writes a `.ok` marker with the model's size and mtime.

- **When the engine uses it.** With `DS4_ROCM_STREAM_MIRROR` set, the engine alternates streamed expert reads
  between the two drives and splits large reads into halves. It uses the mirror only when the marker matches.
  The first 24 mirror reads are compared with the model.
- **Measured.** Cold 12k prefill improves from 23.7 to 20.5 s. Decode does not change: one miss costs about
  2 ms of latency, not bandwidth.

## How we measured

- **Decode without speculation.** kyuz0's documented `ds4-bench` sweep, run on his current `main-gfx1151` and
  on the same commits replayed on it:
  - 2048 → 32768 by ×2, 128 greedy tokens per frontier, `--ctx-alloc 34816`, 72 GB cache, 2 GiB snapshot cap;
  - frontier logits dumped and compared.

  Prefill is unchanged at 91 → 351 tok/s. Decode goes from 10.28-10.48 to 13.82-14.01 tok/s. All five
  frontier logit files are byte-identical.
- **DSpark.** `ds4-bench` has no speculative mode, so `tools/strix-halo/bench_server.sh` starts the server with
  the production flags, sends the same chat request twice, and reports client-side decode tokens/s. The
  second request has warm caches and a rewound live session.
- **Correctness.**
  - `make test-rocm` passes on this branch and on the same commits replayed on kyuz0's current branch.
  - Greedy outputs are byte-identical to the non-speculative path on code.
  - On prose, a near-tie word can differ, around token 90 of 128. The verify rows use different attention
    and norm kernels than single-token decode, so the text is valid but not bit-identical.
  - Every decode-path change was checked for determinism: three consecutive greedy runs must match byte for
    byte.

Verify costs after this work, for one trunk step over the draft rows:

| Rows | 1 (plain decode) | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| ms | 72-75 | 107 | 136 | 160-169 | 193 |

A draft proposal costs 11.8 ms:
- 5.9 ms dense Q8 draft projections;
- 2.5 ms draft MoE;
- 1.6 ms Markov chain.

## What did not work (so you don't have to try)

- **Next-layer expert prefetch.** It runs router L+1 on layer L's FFN input (`DS4_V41_PREFETCH`, off by
  default). Measured recall is 71% at top-6 and 86% at top-12.
  - Top-6 with one expert per layer removes 38% of decode misses but leaves decode flat, because the extra
    router pass and sync cost what the reads save.
  - Top-12 reads too many cold experts: 9.6 tok/s.
  - Restricted to verify batches it is slower too: 24 vs 31.7 tok/s.

  This matches #849, which only wins together with packed experts (#848).
- **Splitting each expert read across two drives.** No decode gain: a single miss is latency-bound.
- **The token-tiled F16 prefill kernel for 2-7 verify rows.** It keeps only 3 blocks busy; a per-row-group
  kernel replaced it.

## Tools

`tools/strix-halo/`:
- `mirror_experts.py`: the sparse expert mirror;
- `dspark41_support.py` and `dspark41_config.json`: DSpark support conversion without the tokenizer;
- `bench_server.sh` and `client.py`: the DSpark/server benchmark.

License: MIT, like the rest of ds4.
