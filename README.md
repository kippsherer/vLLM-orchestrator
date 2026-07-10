# vLLM Orchestrator

Model-on-demand lifecycle manager for [vLLM](https://github.com/vllm-project/vllm). Load models automatically on first request, sleep idle models to free GPU VRAM, and wake them again when needed — Ollama-style UX with vLLM performance and full OpenAI API compatibility.

---

## How it works

The orchestrator sits in front of one or more vLLM processes and acts as a transparent reverse proxy. It:

- **Starts vLLM on demand** when a request arrives for an unloaded model
- **Routes requests** to the correct vLLM instance over Unix domain sockets (no TCP overhead)
- **Sleeps idle models** using vLLM's native `/sleep` endpoint to release GPU VRAM without killing the process
- **Wakes sleeping models** automatically when a new request arrives
- **Terminates models** that have been idle long enough to free resources entirely
- **Schedules GPU groups** — fits each model into the smallest GPU group it requires, preserving large groups for large models

Every byte of every request and response is forwarded unmodified. The orchestrator never inspects, validates, or transforms request bodies beyond extracting the `"model"` field for routing.

### Model state machine

```
UNLOADED ──request──► LOADING ──ready──► ACTIVE ──ttl_active──► SLEEP1 ──ttl_inactive──► SLEEP2 ──ttl_unused──► UNLOADED
                                            ▲          │              ▲            │
                                            └──request─┘              └──request───┘
```

| State | GPU VRAM | CPU RAM | vLLM process |
|---|---|---|---|
| `UNLOADED` | 0 | 0 | not running |
| `LOADING` | allocating | 0 | starting |
| `ACTIVE` | weights + KV cache | 0 | running |
| `SLEEP1` | KV cache only | weights | running |
| `SLEEP2` | CUDA context only (~450 MB/GPU) | small model buffers | running, suspended |

**SLEEP1** offloads weights to CPU RAM via `POST /sleep?level=1`. KV cache remains on GPU. Resume is fast — weights are mapped back from CPU without reloading from disk.  
**SLEEP2** releases weights and KV cache from GPU via `POST /sleep?level=2`. Only the CUDA context (~450 MB per GPU) remains on GPU. A small set of model buffers (auxiliary tensors such as rotary embeddings) are retained in CPU RAM. Resume reloads weights from disk.

---

## Quick start

### Prerequisites

- Go 1.23+
- `vllm` on `$PATH` (installed in a virtualenv or system-wide)
- NVIDIA GPU(s) with drivers and `nvidia-smi` available
- Models accessible via HuggingFace (set `HF_TOKEN` if using gated models)

### Build and run

```sh
git clone https://github.com/kippsherer/vLLM-orchestrator
cd vLLM-orchestrator
go run . --config config.yaml
```

Or build a binary:

```sh
go build -o vllm-orchestrator .
./vllm-orchestrator --config config.yaml
```

The config path can also be set via environment variable:

```sh
export VLLM_ORCH_CONFIG=/etc/vllm-orchestrator/config.yaml
./vllm-orchestrator
```

### Send a request

The orchestrator listens on the port defined in `listen` (default `:8000`, matching vLLM's default). Use it exactly like you would use vLLM directly:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

If the model is not loaded, the request queues while vLLM starts. When vLLM is ready, the response streams back transparently. The caller sees no difference from talking to vLLM directly.

---

## Configuration

Config is a single YAML file. All fields are required unless marked optional.

```yaml
# Address the orchestrator listens on.
# Mirrors vLLM's default port so existing tooling works without changes.
listen: ":8000"

# Directory where per-model Unix socket files are created.
# Created at startup if it does not exist. Must be writable.
vllm_socket_dir: "/run/vllm"

# Maximum queued requests per model before 503 is returned.
queue_depth: 100

# Idle TTL: ACTIVE → SLEEP1 (or SLEEP2 if CPU RAM is insufficient).
# Resets on every completed request.
ttl_active: 10m

# Idle TTL: SLEEP1 → SLEEP2.
ttl_inactive: 60m

# Idle TTL: SLEEP2 → UNLOADED (process terminated).
ttl_unused: 120m

# GPU groups. Each group is a scheduling unit — one model per group at a time.
# The orchestrator measures total VRAM per group via nvidia-smi at startup.
# Do not declare VRAM here; it is measured automatically.
gpu_groups:
  - id: "group0"
    gpus: [0, 1, 2, 3]   # CUDA device IDs — used for CUDA_VISIBLE_DEVICES
                           # and --tensor-parallel-size

  - id: "group1"
    gpus: [4]

models:
  - name: "meta-llama/Meta-Llama-3-8B-Instruct"
    aliases: ["llama3-8b", "llama3"]   # optional; any alias routes to this model
    load_at_startup: true              # optional; load immediately on orchestrator start
    vram_allocation: 75162             # MB this model is allowed to consume on the group;
                                       # --gpu-memory-utilization is derived from this automatically
    vllm_args:                         # passed verbatim to `vllm serve`
      - "--dtype=float16"
      - "--max-model-len=8192"
      - "--enable-prefix-caching"
      - "--tool-call-parser=hermes"

  - name: "mistralai/Mistral-7B-Instruct-v0.3"
    aliases: ["mistral-7b"]
    vram_allocation: 75162
    vllm_args:
      - "--dtype=float16"
      - "--max-model-len=32768"
```

### Configuration rules

**GPU groups**
- Each CUDA device ID must appear in exactly one group. Duplicates are rejected at startup.
- Group VRAM is measured from `nvidia-smi` at startup. If measured values diverge from accounting during operation, a warning is logged every 60 seconds.
- Models are assigned to the **smallest group they fit into**, preserving larger groups for larger models.

**Models**
- `vram_allocation` is the number of MB this model is allowed to consume across all GPUs in its group. The orchestrator derives `--gpu-memory-utilization` from this value automatically at launch time. Set it to `gpu_memory_utilization × total_group_vram_mb` for your hardware.
- `aliases` are additional names clients may use in the `"model"` field. `/v1/models` always returns the canonical name.
- `load_at_startup` is optional (default `false`). When `true`, the model begins loading when the orchestrator starts.
- `vllm_args` are appended to `vllm serve <model_name>` verbatim. **Do not include** `--uds`, `--tensor-parallel-size`, `--gpu-memory-utilization`, or `CUDA_VISIBLE_DEVICES` — these are injected automatically.

**HuggingFace gated models**
- Set `HF_TOKEN` in the orchestrator's environment. It is forwarded into each vLLM subprocess automatically. It is never written to config or logs.

**TTL ordering requirement**
- `ttl_active < ttl_inactive < ttl_unused` — startup aborts if violated.

---

## API

The orchestrator is a transparent pass-through for all vLLM routes. The following are handled directly:

| Method | Path | Response |
|---|---|---|
| `GET` | `/health` | `{"status":"ok"}` |
| `GET` | `/ping` | `{"status":"ok"}` |
| `GET` | `/version` | `{"version":"<build version>"}` |
| `GET` | `/v1/models` | Aggregated list from all running instances + stubs for unloaded models |
| `GET` | `/metrics?model=<name>` | Forwarded to the named model's vLLM instance |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | Forwarded to any active vLLM instance; 503 if none |
| `POST` | `/sleep`, `/wake_up`, `/is_sleeping` | **403 Forbidden** — these are internal orchestrator operations |

All other routes (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/tokenize`, `/v1/detokenize`, `/v1/score`, LoRA management, `/v1/realtime` WebSocket, etc.) are forwarded transparently to the appropriate vLLM instance, identified by the `"model"` field in the request body.

### `/v1/models` response

Each entry includes a non-standard `orchestrator_state` field (`"active"`, `"sleep1"`, `"sleep2"`, `"loading"`, `"unloaded"`) and an `allocated_vram_mb` field showing the model's configured VRAM allocation. For active models, the full vLLM response (including `max_model_len`, `per_model_config`, etc.) is returned unmodified with `orchestrator_state` appended.

---

## Memory accounting

The orchestrator maintains VRAM and CPU RAM accounting:

- **VRAM** per GPU group: measured from `nvidia-smi` (`memory.free`) at startup and re-queried before every model launch and every 60 seconds.
- **Model VRAM allocation**: set by `vram_allocation` in config. Used to derive `--gpu-memory-utilization` and to determine whether freeing rules must run before launching a new model.
- **CPU RAM**: read from `/proc/meminfo` (`MemAvailable`) at startup and re-read every 60 seconds. Used to decide whether a sleeping model's weights can be offloaded to CPU (`SLEEP1`) or must be discarded (`SLEEP2`).

When memory is needed to load a new model, the orchestrator first runs freeing rules proactively on any group that has other models assigned to it, then verifies free VRAM is sufficient. Freeing rules run in order:
1. Move idle ACTIVE models to SLEEP1 (smallest first; offloads weights to CPU, frees that VRAM)
2. Terminate SLEEP2 processes (frees all remaining GPU VRAM)
3. Move SLEEP1 models to SLEEP2 (discards weights from CPU RAM)
4. Terminate SLEEP2 processes again

If memory is still insufficient after all four rules, all queued requests for the model receive 503.

---

## Startup validation

The orchestrator aborts at startup if:

- `nvidia-smi` is unavailable or fails
- Any configured GPU device ID is not found in `nvidia-smi` output
- GPU device IDs are duplicated across groups
- Model names or aliases are duplicated
- `ttl_active >= ttl_inactive` or `ttl_inactive >= ttl_unused`
- `vllm_socket_dir` is not writable

---

## Development

```sh
go test ./...          # run all tests
go test -run TestFoo ./...   # run a specific test
gofmt -w .             # format
goimports -w .         # fix imports
go vet ./...           # vet
```
