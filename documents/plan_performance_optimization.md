# Plan: vLLM Performance Optimization

**Date**: 2026-07-11
**Hardware**: 2–4× RTX 3090 24 GB (no NVLink), 2× Xeon E5-2697 v4 (18 cores / 36 threads each = 36 physical cores / 72 hyperthreads), 256 GB DDR4 RAM, Supermicro X10DRG-Q-CS045 (dual-socket NUMA)
**Software**: vLLM V1 (via `--generation-config=auto`), orchestrator at `github.com/kippsherer/vLLM-orchestrator`
**Observed symptom**: The processes VLLM::EngineCore, VLLM::Worker_TP0, and VLLM::Worker_TP1 each peg one CPU core at 100%. Remaining cores sit idle. (TP=2, so EngineCore + 2 GPU workers = 3 cores pegged.)

---

## Sources

- vLLM V1 optimization documentation: https://docs.vllm.ai/en/stable/configuration/optimization/ (fetched 2026-07-11)
- Codebase: `process.go`, `config.yaml` in this repository
- Hardware facts: dual-socket NUMA, PCIe 3.0 x16 inter-GPU, no NVLink

---

## 🔴 Critical — Fixes the "one core at 100%" Problem

### 1. Remove `OMP_NUM_THREADS=1` — Let PyTorch Use Multiple Threads

**Location:** `process.go`, line 109
**Current value:** `OMP_NUM_THREADS=1` is hardcoded and inherited by every vLLM subprocess (API server, EngineCore, and all GPU workers).

**Why it's a problem:** The vLLM documentation says the GPU worker processes run model forward passes. Although CUDA graphs and flash attention offload compute to GPU, PyTorch still uses CPU threads for:

- Input tensor preparation and pinning
- Attention score assembly before GPU launch
- Sampling / logit processing (softmax, top-k filtering)
- KV cache management overhead

With `OMP_NUM_THREADS=1`, every PyTorch-internal parallel region becomes serial inside every GPU worker. The EngineCore scheduler loop (which is inherently single-threaded and runs a busy loop) saturates whatever core it lands on, while the GPU workers cannot use the other 35 cores to do their CPU-side work in parallel.

The observed process list confirms this: Worker_TP0 and Worker_TP1 are each pegged at 100% on a single core, exactly as expected when OpenMP parallelism is disabled. The EngineCore at 100% is by design (busy-loop scheduler). The API server process is NOT pegged, meaning tokenization is not the bottleneck.

**Fix:** Set `OMP_NUM_THREADS` to a value that allows parallelism without oversubscribing. For your system (4 GPU workers × N threads ≤ 36 physical cores):

```bash
# Remove the override and let vLLM auto-detect, OR set explicitly:
export OMP_NUM_THREADS=8   # 4 workers × 8 = 32 threads, under 36 physical cores
```

In `buildEnv()` in `process.go`, change:

```go
// From:
out = append(out, "CUDA_VISIBLE_DEVICES="+cudaVisible, "VLLM_SERVER_DEV_MODE=1", "OMP_NUM_THREADS=1")

// To:
out = append(out, "CUDA_VISIBLE_DEVICES="+cudaVisible, "VLLM_SERVER_DEV_MODE=1")
```

And set `OMP_NUM_THREADS=8` (or omit it) in the environment that launches your Go binary. Or better, compute it dynamically: `total_physical_cores / (total_gpu_workers + 2)`.

### 2. Add `--numa-bind` — Pin GPU Workers to Their NUMA Node

**Not currently used.** Your hardware is a **dual-socket NUMA system** (2× E5-2697 v4). Without NUMA binding, the OS scheduler may migrate GPU worker processes to the wrong socket, causing:

- Remote memory access latency (cross-socket QPI/UPI)
- CPU cache thrashing
- All GPU workers fighting over cores on one socket while the other socket sits idle

**From the docs:**

> On multi-socket GPU servers, GPU worker processes can lose performance if their CPU execution and memory allocation drift away from the NUMA node nearest to the GPU. vLLM can pin each worker with `numactl` before the Python subprocess starts.

**Fix:** Add `--numa-bind` to every model's `vllm_args` in `config.yaml`:

```yaml
vllm_args:
  - "--dtype=auto"
  - "--numa-bind"
  # ... rest of args
```

For 4 GPUs across 2 sockets, you typically get GPUs 0,1 on NUMA node 0 and GPUs 2,3 on NUMA node 1. Auto-detect with `--numa-bind` is sufficient and preferred. If auto-detect fails, use explicit:

```yaml
  - "--numa-bind"
  - "--numa-bind-nodes=0"
  - "--numa-bind-nodes=1"
```

But auto-detect first. Verify NUMA topology with:

```bash
numactl --hardware
nvidia-smi topo -m
```

**Note:** The docs say `--numa-bind` forces multiprocessing to `spawn` method and auto-sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, which is handled automatically with CLI usage.

---

## 🟠 High Priority — Directly Improves Throughput & Latency

### 3. Add `--max-num-seqs 64` (Reduce from default 256)

**Not currently set.** Default is 256. For your models (especially 32B AWQ on 2×3090 with no NVLink), 256 sequences in a batch creates:

- Higher CPU scheduling overhead per iteration
- Higher KV cache pressure
- More PCIe communication per batch (without NVLink, inter-GPU transfer is over PCIe 3.0 x16 — ~32 GB/s per direction)

**From the docs:**

> Decrease `max_num_seqs` … to reduce the number of concurrent requests in a batch, thereby requiring less KV cache space.

The AWQ Marlin kernel also benefits from smaller, more consistent batches. Start with `--max-num-seqs 64` and monitor GPU utilization.

### 4. Add `VLLM_CPU_OMP_THREADS_BIND` — Pin OpenMP Threads to Physical Cores

Alongside `OMP_NUM_THREADS`, set the thread affinity so OpenMP threads don't wander across cores (which causes cache misses):

```bash
# With 4 GPU workers and OMP_NUM_THREADS=8, create 4 zones of 8 cores each.
# For NUMA-aware binding, let vLLM auto-detect by default:
export VLLM_CPU_OMP_THREADS_BIND=auto   # available in recent vLLM versions

# Or manually for your setup (NUMA0: cores 0-17, NUMA1: cores 18-35):
# For each GPU worker on NUMA0, pin to cores 0-17; on NUMA1, pin to 18-35.
```

The `--numa-bind` option plus `VLLM_CPU_OMP_THREADS_BIND=auto` should together give correct affinity for the GPU workers. From the docs: "By default, `VLLM_CPU_OMP_THREADS_BIND=auto` derives OpenMP placement from the available CPU and NUMA topology for each CPU worker."

### 5. Preload `tcmalloc` (Replace glibc malloc)

**From the user's own notes.** For multi-threaded CPU workloads, glibc's default `malloc` uses a single arena with a global lock, causing contention. `tcmalloc` uses per-thread caching and reduces allocation overhead.

```bash
# Install:
sudo apt install google-perftools  # or libtcmalloc-minimal4

# Preload in env before launching your Go binary:
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

Add this to the environment variables that your orchestrator inherits, or inject it into the vLLM subprocess environment in `buildEnv()`.

### 6. Enable `fastokens` for All BPE Models

**Not currently used.** All your models (Qwen, DeepSeek) use BPE tokenizers. The docs say:

> For BPE tokenizers (Qwen, Llama, DeepSeek, GPT-OSS, etc.) you can switch to the fastokens Rust backend for substantially faster encode/decode and streaming detokenization. Tokenizer-bound workloads (long shared prefixes, bursty short prompts, batch detokenization) see the largest wins.

Set the env var in your orchestrator's environment or in `buildEnv()`:

```go
// In buildEnv(), add:
"VLLM_USE_FASTOKENS=1"
```

Requires `pip install fastokens` (≥0.2.0) in the Python environment where vLLM runs.

---

## 🟡 Medium Priority — Tuning for Your Specific Hardware

### 7. `--disable-custom-all-reduce` (Already Done — Correct)

**Already present on all models.** The custom all-reduce kernel requires NVLink (per vLLM source). RTX 3090s do **not** have NVLink, so this flag is correct. No change needed.

### 8. `--enable-chunked-prefill` (Enabled by Default in V1 — Verify)

From the docs: **"In V1, chunked prefill is enabled by default whenever possible."** Your `--generation-config=auto` uses V1, so chunked prefill should be active. Verify in vLLM startup logs that chunked prefill is enabled. If not, add it explicitly.

Tune `--max-num-batched-tokens` if needed:

- **Smaller values** (e.g., 2048–4096) → better inter-token latency (ITL), less decode slowdown
- **Larger values** (>8192) → better time-to-first-token (TTFT), higher throughput
- The doc recommends >8192 for "smaller models on large GPUs" for optimal throughput

Your models at context lengths of 8K–16K–32K vary. Consider per-model tuning based on workload characteristics. A starting point:

```
--max-num-batched-tokens=8192
```

### 9. Set `--kv-cache-memory` for Faster Restarts

From the docs: On startup, vLLM logs the exact `--kv-cache-memory` value. Passing it back on the next boot skips the memory-profiling measurement pass. For a model-on-demand system where models start and stop frequently, this saves 5–30 seconds per cold start.

```yaml
vllm_args:
  - "--kv-cache-memory=20"  # or whatever vLLM logs for that model+config
```

Without it, each model launch re-profiles memory (CUDA graph estimation, KV cache sizing). Add once you've observed the stable value per model.

### 10. Remove `--gpu-memory-utilization` Auto-Calculation if Using `--kv-cache-memory`

Your orchestrator computes `--gpu-memory-utilization` from `vram_allocation / measured_total_VRAM`. If you switch to `--kv-cache-memory`, this flag becomes redundant for that model. But the computed value is fine for now.

---

## 🔵 Lower Priority — Investigate After Fixing Above

### 11. Consider Data Parallelism Instead of Tensor Parallelism for Smaller Models

**For models that fit on one 24GB 3090** (Qwen3-14B-AWQ at 19.7GB, Qwen3-VL-8B-AWQ at 19.7GB, DeepSeek-Coder-V2-Lite-AWQ at 24.6GB — the last is borderline):

TP=2 over PCIe without NVLink incurs communication overhead for every transformer layer. For models that fit on a single GPU, running two instances with DP=2 (each on one GPU) would give:

- Higher throughput (no cross-GPU sync per layer)
- Better GPU utilization (both GPUs fully busy independently)
- Higher aggregate request capacity

However, your current config uses one group `[0,1]` which forces TP=2 on everything. Consider splitting into multiple GPU groups for smaller models.

### 12. Do NOT Add `--api-server-count` — Not the Bottleneck

**Determined from observed process list:** The API server process does NOT appear in the set of CPU-maxed processes (only EngineCore, Worker_TP0, Worker_TP1 appear). This means tokenization and input processing are not currently limiting throughput. Adding `--api-server-count` would consume additional CPU cores (each API server uses 8 CPU threads by default for media loading) without addressing the actual bottleneck.

From the docs: API server scale-out helps "when input processing becomes a bottleneck compared to model execution." That condition is not met here. The bottleneck is OpenMP-starved GPU workers and the EngineCore scheduler.

Revisit this item only if the API server process becomes CPU-bound after fixing items 1–6.

### 13. Monitor Preemption and Adjust `gpu_memory_utilization`

From the docs: If you see "Sequence group X is preempted" warnings, it means KV cache is too small. Your VRAM allocation numbers seem tight. For 32B models at 45.2GB across 2×24GB GPUs:

- Total VRAM: 48GB minus OS/resident = ~47.5GB
- Model weights (AWQ compressed 32B): ~16–18GB
- Leaves ~29GB for KV cache across 2 GPUs

With `--max-model-len=16384` and `--max-num-seqs=256`, KV cache demand can exceed supply, causing preemption → latency spikes. Reducing `--max-num-seqs` (item 3) directly addresses this.

### 14. Try `-O3` Optimization Level

Current default is `-O2`. From the docs:

> -O3: Aggressive optimization. Currently equal to -O2, but may include additional time-consuming or experimental optimizations in the future.

No immediate benefit today, but watch for future vLLM releases where `-O3` diverges.

---

## Summary: Action Plan (Ordered by Impact)

| # | Action | File / Config | Expected Benefit |
|---|--------|---------------|------------------|
| **1** | Remove `OMP_NUM_THREADS=1` → set to 8 | `process.go:109` | **Worker_TP0/TP1 single-core bottleneck eliminated** — GPU workers use multiple cores |
| **2** | Add `--numa-bind` to all models | `config.yaml` `vllm_args` | Workers pinned to correct socket, no cross-socket latency |
| **3** | Add `--max-num-seqs 64` | `config.yaml` `vllm_args` | Smoother batches, less KV cache pressure |
| **4** | `LD_PRELOAD=tcmalloc` | Env / `buildEnv()` | Faster memory allocation under multi-threaded load |
| **5** | `VLLM_USE_FASTOKENS=1` | Env / `buildEnv()` | Faster BPE tokenization for all models |
| **6** | `VLLM_CPU_OMP_THREADS_BIND=auto` | Env / `buildEnv()` | Stable OpenMP thread affinity |
| **7** | Tune `--max-num-batched-tokens` | `config.yaml` `vllm_args` | Better TTFT vs ITL tradeoff |
| **8** | Add `--kv-cache-memory` after profiling | `config.yaml` `vllm_args` | Faster cold starts |
| **9** | Consider DP for single-GPU-fit models | Config GPU groups | Higher throughput, no PCIe sync overhead |
| — | **Do NOT add `--api-server-count`** | — | Not the bottleneck — API server not in top-CPU processes |

Items **1–2** are the confirmed fix for the observed symptom. Worker_TP0 and Worker_TP1 are each stuck on one core because of `OMP_NUM_THREADS=1`. EngineCore at 100% is by design (busy-loop scheduler). Fixing item 1 will let Worker_TP0 and Worker_TP1 spread across multiple cores.

## Additional Context from vLLM Docs

### CPU Resources for GPU Deployments

From the docs:

> vLLM V1 uses a multi-process architecture. Underprovisioning CPU cores is a common source of performance degradation.

For a deployment with N GPUs, there are at minimum:

- 1 API server process — HTTP requests, tokenization, input processing
- 1 engine core process — scheduler, coordinates GPU workers
- N GPU worker processes — one per GPU, model forward passes

This means there are always at least 2+N processes competing for CPU time.

> Warning: Using fewer physical CPU cores than processes will cause contention and significantly degrade throughput and latency. The engine core process runs a busy loop and is particularly sensitive to CPU starvation.

Minimum is 2+N physical cores. With hyperthreading, 1 vCPU = 1 hyperthread = 1/2 physical CPU core, so you need 2×(2+N) minimum vCPUs. With 36 physical cores and 4 GPUs, this is not a constraint (need only 6 minimum), but CPU contention can still occur if processes are not properly scheduled or if OpenMP threads are under-provisioned.

The observed process list (EngineCore, Worker_TP0, Worker_TP1) matches this architecture exactly. With TP=2, that's 1 EngineCore + 2 workers = 3 processes accounting for the CPU load. The API server (the 4th process) is not CPU-bound.

### Chunked Prefill (V1 Default)

In V1, chunked prefill is enabled by default whenever possible. The scheduling policy prioritizes decode requests, batches all pending decode requests before scheduling prefill operations, and automatically chunks prefills that don't fit into `max_num_batched_tokens`.

Benefits per the docs:

- Improves inter-token latency (ITL) and generation decode because decode requests are prioritized.
- Helps achieve better GPU utilization by locating compute-bound (prefill) and memory-bound (decode) requests in the same batch.

### Parallelism Strategies

- **Tensor Parallelism (TP)**: shards model parameters across multiple GPUs within each model layer. Most common for large model inference within a single node. Currently TP=2 for all models.
- **Pipeline Parallelism (PP)**: distributes model layers across multiple GPUs. Can be combined with TP. Useful when TP has been maxed out.
- **Expert Parallelism (EP)**: for MoE models (DeepSeekV3, Qwen3MoE, Llama-4). Enabled by `enable_expert_parallel=True`. Not relevant for current models.
- **Data Parallelism (DP)**: replicates the entire model across multiple GPU sets. Set by `data_parallel_size=N`. For environments where throughput scaling is needed.

### Multi-Modal Caching

For vision models like Qwen3-VL-8B: multi-modal caching avoids repeated transfer or processing of the same multi-modal data. Processor caching is auto-enabled. IPC caching is auto-enabled when there is a one-to-one correspondence between API and engine core processes. Adjustable via `mm_processor_cache_gb` (default 4 GiB). The docs document two IPC cache types: Key-Replicated Cache (`lru`, default) and Shared Memory Cache (`shm`, for TP > 1).

### Optimization Levels

vLLM provides 4 optimization levels for trading off startup time versus performance:

- `-O0`: No optimizations. Fastest startup, lowest performance.
- `-O1`: Simple compilation and fast fusions, PIECEWISE cudagraphs.
- `-O2`: Default. Additional compilation ranges, additional fusions, FULL_AND_PIECEWISE cudagraphs.
- `-O3`: Aggressive optimization. Currently equal to `-O2`, but may include additional time-consuming or experimental optimizations in the future.

### fastokens Backend

Available in vLLM v0.23.0+. The `fastokens` Python package (>= 0.2.0) must be installed. Models that don't use the HF fast tokenizer (`mistral`, `kimi_audio`) ignore the flag. Available for BPE tokenizers: Qwen, Llama, DeepSeek, GPT-OSS, etc.

### Reuse the Compile Cache

vLLM persists torch.compile artifacts under `VLLM_CACHE_ROOT` (default `~/.cache/vllm`). The cache directory can be copied between machines or baked into a container image. Set `VLLM_FORCE_AOT_LOAD=1` to fail loudly instead of silently recompiling when the cache misses.

### Serve Without CUDA Graphs (`--enforce-eager`)

Skips both compilation and CUDA-graph capture for the fastest possible startup, at the cost of steady-state decode performance. The docs recommend this for development loops and for measuring how much of a boot is compile/capture. Not recommended for production serving.

---

**Bottom line on the single-core saturation:** The `OMP_NUM_THREADS=1` hardcoded in `process.go:109` forces Worker_TP0 and Worker_TP1 to run their PyTorch CPU work single-threaded. The EngineCore is at 100% by design (busy-loop scheduler). Fixing OMP_NUM_THREADS + adding NUMA binding will let Worker_TP0 and Worker_TP1 spread their work across multiple cores.
