# Plan: vLLM Orchestrator in Go

**Date**: 2026-07-09 (revised 4)
**Module**: `github.com/kippsherer/vLLM-orchestrator`
**Goal**: Model-on-demand vLLM process lifecycle manager with a transparent pass-through reverse proxy. Not an API gateway — every byte of every request is forwarded to vLLM unmodified.

**Code discipline**: flat, dense, no unnecessary helpers, no speculative fallbacks, no abstraction layers that serve a single call site.

---

## 1. What "transparent pass-through" means here

The router **must not** inspect, validate, or transform request or response bodies. The only body read is extraction of the top-level `"model"` string for routing. After routing, the raw body bytes are streamed to vLLM and the raw response bytes are streamed back.

This preserves:
- vLLM-specific sampling params (`best_of`, `use_beam_search`, `top_k`, `min_p`, `repetition_penalty`, `length_penalty`, `stop_token_ids`, `include_stop_str_in_output`, `ignore_eos`, `min_tokens`, `truncate_prompt_tokens`, `enable_prefix_caching`, `prompt_logprobs`, `skip_special_tokens`, `spaces_between_special_tokens`)
- Guided decoding (`guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `guided_decoding_backend`, `guided_whitespace_pattern`)
- Multimodal bodies (`multi_modal_data`, content arrays with `image_url` entries)
- Non-OpenAI endpoints (`/v1/tokenize`, `/v1/detokenize`, `/v1/score`, `/tokenize`, `/detokenize`)
- LoRA management routes (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/v1/list_lora_adapters`)
- Prometheus metrics at `/metrics`
- WebSocket at `/v1/realtime`
- vLLM dev/debug endpoints (`/version`, `/ping`, `/docs`, `/redoc`, `/openapi.json`, `/sleep`, `/wake_up`, `/is_sleeping`)
- vLLM's extended `/v1/models` fields (`root`, `parent`, `max_model_len`, `per_model_config`)
- Streaming SSE chunks including `usage.prompt_tokens_details.cached_tokens`

---

## 2. Configuration

Config is a single YAML file (path provided via `--config` flag or `VLLM_ORCH_CONFIG` env var).

```yaml
listen: ":8000"               # orchestrator bind address; mirrors vLLM's default port
vllm_socket_dir: "/run/vllm"  # directory where per-model Unix socket files are created
queue_depth: 100              # max queued requests per model before 503 is returned

ttl_active: 10m               # idle TTL: ACTIVE → SLEEP1 (resets on each completed request)
ttl_inactive: 60m             # idle TTL: SLEEP1 → SLEEP2
ttl_unused: 120m              # idle TTL: SLEEP2 → UNLOADED (process terminated)

gpu_groups:
  - id: "group0"
    gpus: [0, 1, 2, 3]        # CUDA device IDs in this group

  - id: "group1"
    gpus: [4]                  # single-GPU group

models:
  - name: "meta-llama/Meta-Llama-3-8B-Instruct"
    aliases: ["llama3-8b", "llama3"]
    load_at_startup: true      # optional: load this model immediately on orchestrator start
    vllm_args:                 # passed verbatim to `vllm serve`
      - "--dtype=float16"
      - "--max-model-len=8192"
      - "--enable-prefix-caching"
      - "--tool-call-parser=hermes"

  - name: "mistralai/Mistral-7B-Instruct-v0.3"
    aliases: ["mistral-7b"]
    vllm_args:
      - "--dtype=float16"
      - "--max-model-len=32768"
```

**Key config rules:**
- GPU groups declare only CUDA device IDs. **Total VRAM per group is measured from `nvidia-smi` at orchestrator startup** — not declared. The orchestrator queries `nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits`, maps each device ID to its measured total, and sums per group. This is the authoritative value used for all scheduling.
- `weights_vram_mb` and `full_kv_vram_mb` are **not** in config — auto-measured at first launch of each model by parsing vLLM's startup logs (see §10).
- `vllm_args` are appended to `vllm serve <model_name>` exactly as written. The orchestrator auto-injects `--uds`, `--tensor-parallel-size`, and `CUDA_VISIBLE_DEVICES`; the user must not include these.
- `HF_TOKEN`: if set in the orchestrator's environment, it is forwarded verbatim into each vLLM subprocess environment. This is the only mechanism needed for gated HuggingFace models — vLLM's bundled `huggingface_hub` library picks it up automatically when downloading weights. The token is never written to config or logs.

---

## 3. Model State Machine

Each model has exactly one of these states at any moment:

```
UNLOADED ──request──► LOADING ──ready──► ACTIVE ──ttl_active──► SLEEP1 ──ttl_inactive──► SLEEP2 ──ttl_unused──► UNLOADED
                                           ▲          │              ▲            │
                                           └──request─┘              └──request───┘
                                           └──request (from SLEEP2)──────────────────────────────────────────────────┘
```

| State | Description | VRAM held | CPU RAM held | vLLM process |
|---|---|---|---|---|
| `UNLOADED` | No resources | 0 | 0 | Not running |
| `LOADING` | Process starting, measuring memory, not yet ready | reserved (see §10) | 0 | Starting |
| `ACTIVE` | Serving requests | `full_kv_vram_mb` | 0 | Running, GPU active |
| `SLEEP1` | Weights offloaded to CPU via `POST /sleep?level=1` | 0 | `weights_vram_mb` | Running, GPU freed |
| `SLEEP2` | All GPU memory discarded via `POST /sleep?level=2` | 0 | 0 | Running, fully suspended |

**State descriptions:**
- **SLEEP1** uses vLLM's native `POST /sleep?level=1` — offloads model weights from GPU VRAM to CPU host RAM. GPU VRAM fully released. Warm resume via `POST /wake_up`.
- **SLEEP2** uses vLLM's native `POST /sleep?level=2` — discards all GPU memory. No CPU RAM held. The vLLM process remains alive but fully suspended; weights must reload from disk on `wake_up`. Resume is slower than SLEEP1 but faster than a full cold start.
- **UNLOADED** — process terminated. `ttl_unused` elapsed in SLEEP2, or killed by freeing memory rules. Full cold start on next request.

**Transition triggers:**
- `UNLOADED → LOADING`: incoming request for this model
- `LOADING → ACTIVE`: `GET /health` returns 200 AND memory values measured from startup logs
- `ACTIVE → SLEEP1`: `active_request_count == 0` AND `ttl_active` elapses AND `free_cpu_ram_mb >= weights_vram_mb`
- `ACTIVE → SLEEP2`: `active_request_count == 0` AND `ttl_active` elapses AND insufficient CPU RAM for SLEEP1
- `SLEEP1 → ACTIVE`: incoming request → `POST /wake_up` → poll `GET /is_sleeping` until false
- `SLEEP1 → SLEEP2`: `active_request_count == 0` AND `ttl_inactive` elapses
- `SLEEP2 → ACTIVE`: incoming request → `POST /wake_up` → poll `GET /is_sleeping` until false
- `SLEEP2 → UNLOADED`: `active_request_count == 0` AND `ttl_unused` elapses → terminate process
- Any state → `UNLOADED`: process exits unexpectedly (crash)

**The TTL clock resets** on every successfully forwarded response for that model. TTL timers are per-model and fire independently based solely on that model's own idle time. Displacement of other models when memory is needed is handled by the freeing memory rules (§6).

**Sleep endpoint availability:** vLLM's `/sleep`, `/wake_up`, and `/is_sleeping` endpoints require `VLLM_SERVER_DEV_MODE=1`. The orchestrator always sets this env var when launching vLLM subprocesses.

---

## 4. Orchestrator–vLLM Transport

**The orchestrator connects to vLLM instances over Unix domain sockets.**

vLLM's `vllm serve` CLI natively supports `--uds <path>` (defined in `vllm/entrypoints/openai/cli_args.py`). When `--uds` is set, `--host` and `--port` are ignored and vLLM binds an `AF_UNIX / SOCK_STREAM` socket at the given path.

Each vLLM process gets a socket path derived from its model name — one process per model, so the model name is a unique identifier:

```
<vllm_socket_dir>/<sanitized-model-name>.sock
```

`sanitized-model-name` replaces every character outside `[a-zA-Z0-9_.-]` with `_`. Examples:

```
meta-llama/Meta-Llama-3-8B-Instruct  →  /run/vllm/meta-llama_Meta-Llama-3-8B-Instruct.sock
mistralai/Mistral-7B-Instruct-v0.3   →  /run/vllm/mistralai_Mistral-7B-Instruct-v0.3.sock
```

The orchestrator creates `vllm_socket_dir` at startup if it does not exist (`0700` permissions). Before launching any process, if the target socket file already exists and no live process owns it, it is removed. If a live process owns it, abort with an error.

**Go proxy connection to a Unix socket upstream:**

```go
socketPath := "/run/vllm/meta-llama_Meta-Llama-3-8B-Instruct.sock"
transport := &http.Transport{
    DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
        return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
    },
}
proxy := &httputil.ReverseProxy{
    Director: func(req *http.Request) {
        req.URL.Scheme = "http"
        req.URL.Host = "vllm" // arbitrary; ignored by the transport
    },
    Transport:     transport,
    FlushInterval: -1,
}
```

WebSocket connections for `/v1/realtime` use `net.Dial("unix", socketPath)` directly.

All orchestrator→vLLM HTTP calls (health checks, `/sleep`, `/wake_up`, `/is_sleeping`) use the same Unix socket transport via a shared `*http.Client` per model instance.

---

## 5. GPU Group Assignment

When a model transitions from `UNLOADED` to `LOADING`:

1. Find all GPU groups where `free_vram_mb >= model.full_kv_vram_mb`. If `full_kv_vram_mb` is not yet measured (first launch), use `group.measured_total_vram_mb * 0.85` as a placeholder.
2. Among qualifying groups, select the one with the **smallest** `measured_total_vram_mb`. (Fit the model into the smallest group it fits, preserving large groups for large models.)
3. If no group qualifies, execute the **freeing memory rules** (§6) before retrying.
4. Launch the vLLM process with `CUDA_VISIBLE_DEVICES=<group.gpus joined by comma>` and `--tensor-parallel-size=<len(group.gpus)>`.
5. Immediately reserve `full_kv_vram_mb` (or placeholder) from that group's `free_vram_mb` to prevent double-booking during startup.

`free_vram_mb` per group:

```
group.free_vram_mb = group.measured_total_vram_mb
                   − sum(full_kv_vram_mb for all LOADING or ACTIVE models on this group)
```

---

## 6. Freeing Memory Rules

Applied **per GPU group** individually, only when a group has insufficient `free_vram_mb` to load a requested model. Within each rule, models are processed **smallest first** (by `full_kv_vram_mb`). Models with in-flight requests are never touched.

**Rule 1 — Move ACTIVE → SLEEP1 (if CPU RAM allows; if not, run Rule 2 first to free CPU RAM)**
- For each idle ACTIVE model on the group (smallest first):
  - If `free_cpu_ram_mb >= model.weights_vram_mb`: call `POST /sleep?level=1`, poll `GET /is_sleeping` until true, update VRAM and CPU RAM accounting.
  - If CPU RAM is insufficient: run Rule 2 (terminate SLEEP2 processes) until enough CPU RAM is freed, then retry SLEEP1 for this model.
- Stop as soon as the group has enough free VRAM for the target model.

**Rule 2 — Terminate SLEEP2 processes**
- For each SLEEP2 model on the group (smallest first): send `SIGTERM` (SIGKILL after 30s), mark `UNLOADED`, remove socket file.
- Stop as soon as enough VRAM or CPU RAM is freed.

**Rule 3 — Move SLEEP1 → SLEEP2 (frees CPU RAM held by SLEEP1)**
- For each SLEEP1 model on the group (smallest first): call `POST /sleep?level=2`, poll `GET /is_sleeping`, release `weights_vram_mb` from CPU RAM accounting, mark `SLEEP2`.
- Stop as soon as enough CPU RAM is freed.

**Rule 4 — Repeat Rule 2** to terminate SLEEP2 processes and free remaining VRAM after Rule 3.

**Cycle**: `1 → 2 → 3 → 4`. If memory is still insufficient after all four rules, return 503 to all queued waiters for this model.

---

## 7. Request Handling Flow

```
HTTP/WS request arrives
         │
         ▼
Extract "model" field from body (minimal JSON scan; buffer and replay — see §8C)
         │
         ├── No model field, or non-JSON body → 400 Bad Request
         │
         ▼
Resolve model: exact name match OR alias match → not found → 404
         │
         ▼
Model ACTIVE?
  YES → forward pass-through immediately (§8A or §8B)
  NO  →
         │
         ▼
Enqueue request in per-model queue (buffered channel, depth = queue_depth)
         │
         ├── Queue full → 503 immediately
         │
         ▼
Model SLEEP1 or SLEEP2?
  YES → POST /wake_up → poll GET /is_sleeping until false → mark ACTIVE
      → drain queue: forward all waiting requests pass-through
  NO  →
         │
         ▼
Model LOADING (already in progress from another request)?
  YES → wait for ACTIVE, then drain queue
  NO (UNLOADED) →
    → Run freeing memory rules (§6) if insufficient VRAM
    ├── Still insufficient after all four rules → drain queue with 503
    └── Memory freed → assign GPU group (§5) → launch vLLM process (§9)
          → poll GET /health until 200
          ├── Timeout → drain queue with 503
          └── Ready → mark ACTIVE → drain queue: forward all waiting requests
```

**503 is only returned** when: (a) all four freeing memory rules exhausted and VRAM still insufficient, (b) health poll times out, or (c) per-model queue is full.

**Concurrent arrivals** for the same UNLOADED model: only one triggers the load sequence; all others enqueue and drain when ACTIVE.

---

## 8. Transparent Proxy Implementation

### 8A. HTTP (all routes except `/v1/realtime`)

`net/http/httputil.ReverseProxy` with a custom `Transport` (Unix socket dial) and a `Director` that:
1. Sets `req.URL.Scheme = "http"` and `req.URL.Host = "vllm"` (ignored by transport).
2. Does **not** modify headers, body, query params, or path.

`FlushInterval: -1` for immediate SSE flushing. `ErrorHandler` returns 502 on unreachable upstream.

### 8B. WebSocket (`/v1/realtime`)

`httputil.ReverseProxy` does not handle WebSocket upgrades:

1. Detect `Upgrade: websocket` header.
2. `net.Dial("unix", socketPath)` to the target vLLM process.
3. Write the original HTTP `Upgrade` request (headers intact) to the upstream connection.
4. Two goroutines: bidirectional `io.Copy` between client and upstream.
5. Both sides closed via `sync.Once` + `defer` when either side closes.

### 8C. Body peeking for model extraction

1. Read entire body into a `[]byte` buffer.
2. Minimal JSON scan: walk bytes to find the `"model"` key and extract its string value. Do **not** use `encoding/json` — re-serialization drops unknown fields.
3. `req.Body = io.NopCloser(bytes.NewReader(buf))` and `req.ContentLength = int64(len(buf))` on the upstream request.

Routes needing no model field (`GET /health`, `GET /v1/models`) bypass body peek entirely.

### 8D. Routes owned by the orchestrator

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | 200 `{"status":"ok"}` |
| `GET` | `/v1/models` | Aggregate from all running vLLM instances + stubs for UNLOADED models |

For `/v1/models`: query each running vLLM's `/v1/models`, merge the `data` arrays. For UNLOADED models, synthesize a minimal stub. Each entry includes `"orchestrator_state"` (`"active"`, `"sleep1"`, `"sleep2"`, `"unloaded"`).

All other routes — including `/metrics`, `/version`, `/ping`, `/docs`, `/sleep`, `/wake_up`, `/is_sleeping`, LoRA routes, tokenization routes — are forwarded transparently.

---

## 9. vLLM Process Management

### Launching a vLLM process

```sh
VLLM_SERVER_DEV_MODE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve <model_name> \
  --uds /run/vllm/<sanitized-model-name>.sock \
  --tensor-parallel-size 4 \
  [user-provided vllm_args...]
```

- `VLLM_SERVER_DEV_MODE=1` always injected — required for `/sleep`, `/wake_up`, `/is_sleeping`.
- `HF_TOKEN` forwarded from orchestrator environment if present.
- `--uds <socket_path>` always injected. Stale socket file removed before launch if unowned.
- Stdout/stderr piped to orchestrator's structured logger tagged with model name.
- **Memory measurement**: parse stdout for:
  - `"Loading model weights took X.XX GB"` → `weights_vram_mb = X.XX * 1024`
  - `"GPU KV cache size: X.XX GB"` → `kv_vram_mb = X.XX * 1024`; `full_kv_vram_mb = weights_vram_mb + kv_vram_mb`
  - Values cached in memory. Placeholder reservation replaced with actual on measurement.
  - If lines not found within 60s, warn and retain placeholder.
- Readiness: poll `GET /health` (via Unix socket) every 2s until 200 or 300s timeout. On timeout: SIGKILL, remove socket file, mark `UNLOADED`, drain queue with 503.

### Stopping a vLLM process

1. Confirm `active_request_count == 0`.
2. `SIGTERM`. Wait 30s. `SIGKILL` if still running.
3. Remove socket file. Mark `UNLOADED`.

### SLEEP1 transition

1. Confirm `active_request_count == 0`.
2. `POST /sleep?level=1` via Unix socket.
3. Poll `GET /is_sleeping` until `{"is_sleeping":true}` or 30s timeout.
4. Release `full_kv_vram_mb` from VRAM; charge `weights_vram_mb` to CPU RAM. Mark `SLEEP1`.

### SLEEP2 transition

1. Confirm `active_request_count == 0`.
2. `POST /sleep?level=2` via Unix socket.
3. Poll `GET /is_sleeping` until `{"is_sleeping":true}` or 30s timeout.
4. Release `weights_vram_mb` from CPU RAM. Mark `SLEEP2`.

### Wake from SLEEP1 or SLEEP2

1. `POST /wake_up` via Unix socket.
2. Poll `GET /is_sleeping` until `{"is_sleeping":false}` or 60s timeout.
3. Restore `full_kv_vram_mb` to VRAM; release CPU RAM if coming from SLEEP1. Mark `ACTIVE`.

---

## 10. Memory Accounting

### VRAM (per GPU group)

- `group.measured_total_vram_mb`: queried at startup via `nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits`. Summed per group. Only source of truth.
- `group.free_vram_mb = group.measured_total_vram_mb − sum(full_kv_vram_mb for LOADING or ACTIVE models on this group)`.
- Decremented on `LOADING`/`ACTIVE`; restored on `SLEEP1`, `SLEEP2`, or `UNLOADED`.
- Re-queried every 60s. Log warning if measured free VRAM is less than accounted.

### CPU RAM

- `free_cpu_ram_mb`: read at startup from `/proc/meminfo` (`MemAvailable`).
- Decremented by `weights_vram_mb` on `SLEEP1`; restored when exiting `SLEEP1`.
- `SLEEP2` holds no CPU RAM.
- Re-read every 60s as a sanity check.

### Per-model memory values

Not in config. Measured at first launch from vLLM stdout. Cached for the orchestrator's lifetime. Placeholder `group.measured_total_vram_mb * 0.85` used before first measurement.

```go
type modelMemory struct {
    weightsVRAM_MB int64
    fullKVVRAM_MB  int64
    measured       bool
}
```

---

## 11. Package Structure

```
vLLM-orchestrator/
├── main.go        # flag parsing, config load, signal handling, HTTP server
├── config.go      # Config struct, YAML parsing, startup validation
├── proxy.go       # ReverseProxy, WebSocket proxy, body peek, route dispatch, /v1/models aggregation
├── state.go       # model state machine, TTL timers, per-model request queue
├── scheduler.go   # freeing memory rules, GPU group assignment
├── process.go     # subprocess launch, log parsing, health poll, teardown, sleep/wake calls
├── memory.go      # VRAM + CPU RAM accounting, nvidia-smi queries
├── go.mod
└── documents/
    └── plan_vllm_orchestrator.md
```

All files `package main`. No sub-packages.

---

## 12. Concurrency Model

- One goroutine per running vLLM process: stdout log drain (memory measurement) + stderr drain.
- One goroutine per model: TTL timer loop + state transition controller + queue drain on ACTIVE.
- One goroutine per in-flight proxied request: standard Go HTTP handler goroutine.
- Two goroutines per active WebSocket connection: bidirectional `io.Copy` pair.
- All shared state (`groupState`, model states, `modelMemory`) under a single `sync.RWMutex`. Transitions write-locked; reads read-locked.
- `/sleep` and `/wake_up` HTTP calls made outside the lock; state updated under write lock after the call returns.
- Per-model queue is a `chan requestPair` (buffered to `queue_depth`) where `requestPair` holds `(http.ResponseWriter, *http.Request)`.

---

## 13. Configuration Validation at Startup

Abort if:
- `nvidia-smi` unavailable or fails.
- Any GPU device ID in config not found in `nvidia-smi` output.
- Duplicate GPU device IDs across groups.
- Duplicate model names or aliases.
- `ttl_active >= ttl_inactive` or `ttl_inactive >= ttl_unused`.
- `vllm_socket_dir` not writable.

Warn (not abort) if:
- A `load_at_startup` model's placeholder (`measured_total * 0.85`) exceeds all group totals.

---

## 14. What Is Explicitly Out of Scope

- **Request authentication**: all headers forwarded verbatim.
- **Multiple instances of the same model**: one vLLM process per model.
- **OpenAI schema validation or transformation**: never.
- **Metrics aggregation**: `/metrics` forwarded to the matching instance.
- **Dynamic config reload**.

---

## 15. Implementation Order

1. `config.go` — structs, YAML parse, validation.
2. `memory.go` — nvidia-smi startup query, groupState, `/proc/meminfo`, periodic re-query.
3. `process.go` — launch with `--uds`, log parse, health poll, SIGTERM/SIGKILL, socket cleanup, sleep/wake calls.
4. `state.go` — state machine, TTL timers, per-model queue.
5. `scheduler.go` — freeing memory rules, GPU group assignment.
6. `proxy.go` — body peek, ReverseProxy via Unix socket, WebSocket proxy, route dispatch, `/v1/models` aggregation.
7. `main.go` — wire everything, HTTP server on `:8000`, SIGTERM graceful shutdown.

Tests follow the same order, table-driven and parallel.

---

## 16. Key Implementation Risks

| Risk | Mitigation |
|---|---|
| vLLM startup log format changes | Named-group regex; fall back to placeholder on parse failure |
| TTL fires while request in-flight | Check `active_request_count == 0` before any sleep/terminate; re-arm if > 0 |
| vLLM process hangs on startup | 300s health poll timeout; SIGKILL; 503 to queue |
| VRAM accounting diverges | nvidia-smi re-query every 60s; log warning on divergence |
| Stale socket file after crash | On startup enumerate `*.sock` in `vllm_socket_dir`; remove if unowned |
| WebSocket connection leak | Both sides closed via `sync.Once` + `defer` |
| `POST /sleep` races trailing SSE | Check `active_request_count == 0` before calling `/sleep` |
| `VLLM_SERVER_DEV_MODE=1` exposure | vLLM on Unix socket only; no network port exposed |
| Rule 1 / Rule 2 CPU RAM deadlock | Rule 2 called inline from Rule 1 only when CPU is the bottleneck; Rule 2 never touches models with in-flight requests |
