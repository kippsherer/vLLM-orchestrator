# Plan: vLLM Orchestrator in Go

**Date**: 2026-07-09 (revised)  
**Module**: `github.com/kippsherer/vLLM-orchestrator`  
**Goal**: Model-on-demand vLLM process lifecycle manager with a transparent pass-through reverse proxy. Not an API gateway — every byte of every request is forwarded to vLLM unmodified.

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
listen: ":8080"               # orchestrator bind address
vllm_port_start: 9000         # first port in the pool for vLLM processes

ttl_active: 10m               # idle TTL while ACTIVE → transition to SLEEP1
ttl_inactive: 60m             # idle TTL while SLEEP1 → transition to UNLOADED
ttl_unused: 120m              # unused TTL while UNLOADED (informational; no transition)

gpu_groups:
  - id: "group0"
    gpus: [0, 1, 2, 3]        # CUDA device IDs
    total_vram_mb: 96000       # sum of group VRAM; used for scheduling calculations

  - id: "group1"
    gpus: [4]
    total_vram_mb: 24000

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
- `total_vram_mb` is a user-declared value used for scheduling. It is cross-checked against live `nvidia-smi` output at startup (warning logged if discrepancy > 5%, not fatal).
- `weights_vram_mb` and `full_kv_vram_mb` are **not** in config — they are measured automatically at first launch of each model by parsing vLLM's startup logs, then cached in memory for the lifetime of the orchestrator process (see §9).
- `vllm_args` are appended to `vllm serve <model_name>` exactly as written. The orchestrator auto-injects `--port`, `--host`, `--tensor-parallel-size`, and `CUDA_VISIBLE_DEVICES`; the user must not include these.
- TTLs are global; per-model overrides can be added in a future revision.

---

## 3. Model State Machine

Each model has exactly one of these states at any moment:

```
UNLOADED ──request──► LOADING ──ready──► ACTIVE ──ttl_active──► SLEEP1 ──ttl_inactive──► UNLOADED
                                           ▲          │
                                           └──request─┘
                                           └──request (from SLEEP1)─────────────────────┘
```

| State | Description | VRAM held | CPU RAM held | vLLM process |
|---|---|---|---|---|
| `UNLOADED` | No resources | 0 | 0 | Not running |
| `LOADING` | Process starting, measuring memory, not yet ready | reserved (see §9) | 0 | Starting |
| `ACTIVE` | Serving requests | `full_kv_vram_mb` | 0 | Running, GPU active |
| `SLEEP1` | Weights offloaded to CPU via `POST /sleep?level=1` | 0 | `weights_vram_mb` | Running, GPU freed |

**Why only three operational states:**
- **SLEEP1** uses vLLM's native `POST /sleep?level=1` — offloads model weights from GPU VRAM to CPU host RAM. The vLLM process stays alive. GPU VRAM is fully released. Resuming calls `POST /wake_up` on the same process — no restart needed, warm resume.
- **SLEEP2 and UNLOADED are merged**: `sleep(level=2)` discards all GPU memory but keeps the process alive with no CPU RAM benefit worth preserving (weights must be reloaded from disk on `wake_up` regardless). For simplicity, SLEEP2 is not used — instead the process is terminated (UNLOADED). Resuming always re-launches vLLM from disk.
- This means only two resume paths: warm (SLEEP1 → ACTIVE via `POST /wake_up`) and cold (UNLOADED → LOADING → ACTIVE via new process).

**Transition triggers:**
- `UNLOADED → LOADING`: incoming request for this model
- `LOADING → ACTIVE`: `GET /health` on vLLM returns 200 + memory values measured from logs
- `ACTIVE → SLEEP1`: active request count reaches 0 AND `ttl_active` elapses AND enough free CPU RAM exists for `weights_vram_mb`
- `ACTIVE → UNLOADED`: `ttl_active` elapses AND insufficient CPU RAM for SLEEP1 (skip directly to terminate)
- `SLEEP1 → ACTIVE`: incoming request → `POST /wake_up` → poll `GET /is_sleeping` until false
- `SLEEP1 → UNLOADED`: active request count is 0 AND `ttl_inactive` elapses → terminate process
- Any state → `UNLOADED`: process exits unexpectedly (crash)

**The TTL clock resets** on every successfully forwarded response for that model.

**Sleep endpoint availability:** vLLM's `/sleep`, `/wake_up`, and `/is_sleeping` endpoints require `VLLM_SERVER_DEV_MODE=1`. The orchestrator always sets this env var when launching vLLM subprocesses.

---

## 4. GPU Group Assignment

When a model transitions from `UNLOADED` to `LOADING`:

1. Find all GPU groups where `free_vram_mb >= model.full_kv_vram_mb`. (If `full_kv_vram_mb` is not yet known for a new model, use an estimate based on `total_vram_mb * 0.85` for the first launch; actual value is measured during LOADING and stored.)
2. Among qualifying groups, select the one with the **smallest** `total_vram_mb`. (Bin-packing heuristic: fit the model into the smallest box it fits, preserving large groups for large models.)
3. If no group qualifies, execute the **freeing memory rules** (§5) before retrying the selection.
4. Inject `--tensor-parallel-size=<len(group.gpus)>` and set `CUDA_VISIBLE_DEVICES=<group.gpus joined by comma>` in the vLLM subprocess environment.
5. Reserve `full_kv_vram_mb` of that group's `free_vram_mb` immediately (before the process is ready) to prevent double-booking.

`free_vram_mb` per group is maintained in memory:

```
group.free_vram_mb = group.total_vram_mb
                   - sum(full_kv_vram_mb for all ACTIVE or LOADING models on this group)
```

---

## 5. Freeing Memory Rules

Applied **per GPU group** individually, only when a group has insufficient `free_vram_mb` to load a requested model. Applied in order, **smallest model first** (by `full_kv_vram_mb`).

**Precondition**: Models with in-flight requests are never touched.

**Rule 1 — Move ACTIVE → SLEEP1 (if CPU RAM allows)**  
- Check `free_cpu_ram_mb >= model.weights_vram_mb`.
- Call `POST /sleep?level=1` on the vLLM process. Poll `GET /is_sleeping` until true.
- Release the model's `full_kv_vram_mb` from the group's VRAM accounting.
- Charge `model.weights_vram_mb` to CPU RAM accounting.
- Repeat smallest-first until the group has enough free VRAM. Stop when target is met.

**Rule 2 — Terminate SLEEP1 processes**  
- Send `SIGTERM` to the vLLM process of the smallest SLEEP1 model on the group.
- If it does not exit within 30s, send `SIGKILL`.
- Release `model.weights_vram_mb` from CPU RAM accounting. Mark model `UNLOADED`.
- Repeat smallest-first until enough CPU RAM is freed (for pending Rule 1 slots) or enough VRAM is freed. Stop when target is met.

**Rule 3 — If VRAM still insufficient after Rules 1 and 2 — and no SLEEP1 models remain — terminate ACTIVE models without in-flight requests** (last resort, same smallest-first order).
- Drain check: only proceed if `active_request_count == 0` for the candidate model.
- Send `SIGTERM`. Release `full_kv_vram_mb`. Mark `UNLOADED`.
- Stop when target is met.

**Cycle**: `1 → 2 → 3`. If after all three rules there is still insufficient memory, return a 503 immediately (see §6).

---

## 6. Request Handling Flow

```
HTTP/WS request arrives
         │
         ▼
Extract "model" field from body (minimal JSON scan; buffer and replay bytes — see §7C)
         │
         ├── No model field, or non-JSON body → 400 Bad Request
         │
         ▼
Resolve model: exact name match OR alias match → not found → 404
         │
         ▼
Model ACTIVE?
  YES → forward pass-through (§7A or §7B)
  NO  →
         │
         ▼
Model SLEEP1?
  YES → POST /wake_up on vLLM process → poll GET /is_sleeping until false
      → mark ACTIVE → forward pass-through
  NO  →
         │
         ▼
Model UNLOADED or LOADING?
  → Run freeing memory rules (§5) if needed
  → Assign GPU group (§4)
  → Launch vLLM process (§8)
  → Poll GET /health until 200 (or timeout → 503)
  → Mark ACTIVE → forward pass-through
         │
         ├── Insufficient memory after rules → 503 {"error": {"message": "Insufficient VRAM",
         │       "type": "ServiceUnavailableError", "code": 503}}
         │
         └── Health poll timeout → 503
```

**No request queuing.** The 503 is immediate if resources cannot be freed or the model fails to start. Clients are expected to retry.

---

## 7. Transparent Proxy Implementation

### 7A. HTTP (all routes except `/v1/realtime`)

Use `net/http/httputil.ReverseProxy` with a custom `Director` that:
1. Sets `req.URL.Scheme` and `req.URL.Host` to the target vLLM process address.
2. Does **not** modify headers, body, query params, or path.
3. Sets `X-Forwarded-For` (standard reverse proxy behavior).

For SSE (streaming) responses, `httputil.ReverseProxy` handles chunked transfer encoding natively — no special code needed. Disable response buffering via `ReverseProxy.FlushInterval = -1` to ensure SSE tokens are flushed immediately.

The `ModifyResponse` hook is a no-op. The `ErrorHandler` returns a 502 if the upstream vLLM process is unreachable.

### 7B. WebSocket (`/v1/realtime`)

`httputil.ReverseProxy` does not handle WebSocket upgrades. Use a custom handler:

1. Detect `Upgrade: websocket` header.
2. Dial a raw TCP connection to the target vLLM process using the same resolved host:port.
3. Write the original HTTP `Upgrade` request (headers intact) to the upstream connection.
4. Launch two goroutines for bidirectional `io.Copy` between client and upstream connections.
5. Close both connections when either side closes (use `sync.Once` on close).

### 7C. Body peeking for model extraction

The request body must be read once to extract the `"model"` field, then replayed verbatim. Steps:

1. Read entire body into a `[]byte` buffer. Apply a configurable size limit (`max_request_body_mb`, default 256 MB, to handle multimodal payloads).
2. Use a minimal JSON scan — walk bytes looking for the key `"model"` and extract its string value. Do **not** use `encoding/json` to unmarshal the full body (re-serialization can silently drop unknown fields; ordering is not preserved).
3. Reassign `req.Body = io.NopCloser(bytes.NewReader(buf))` and set `req.ContentLength = int64(len(buf))` for the upstream request.

Routes where no model field is needed (e.g. `GET /health`, `GET /metrics`, `GET /v1/models`) bypass the body peek entirely.

### 7D. Routes owned by the orchestrator

Handled directly by the orchestrator; not forwarded to any vLLM instance:

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | 200 `{"status":"ok"}` if orchestrator is up |
| `GET` | `/ping` | 200 `{"status":"ok"}` always |
| `GET` | `/v1/models` | Aggregate from all running vLLM instances + stubs for UNLOADED models |

For `/v1/models`: query each running vLLM's `/v1/models`, merge the `data` arrays. For UNLOADED models, synthesize a minimal stub entry using config data (name, aliases). The orchestrator adds an `"orchestrator_state"` field to each entry (`"active"`, `"sleep1"`, `"unloaded"`) so clients can observe lifecycle state without querying the orchestrator separately.

All other routes — including `/metrics`, `/version`, `/docs`, `/sleep`, `/wake_up`, `/is_sleeping`, LoRA routes, tokenization routes — are forwarded to the appropriate vLLM instance transparently.

---

## 8. vLLM Process Management

### Launching a vLLM process

```sh
VLLM_SERVER_DEV_MODE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve <model_name> \
  --host <bind_host> \
  --port <assigned_port> \
  --tensor-parallel-size 4 \
  [user-provided vllm_args...]
```

- `VLLM_SERVER_DEV_MODE=1` is always injected — required for `/sleep`, `/wake_up`, `/is_sleeping` endpoints.
- Port assignment: orchestrator maintains a pool starting at `vllm_port_start`. Each model gets a unique port; released on process exit.
- Process started with `os/exec.Cmd`. Stdout/stderr piped to orchestrator logger with model name as a structured field.
- **Memory measurement**: parse stdout during startup for lines matching:
  - `"Loading model weights took X.XX GB"` → `weights_vram_mb`
  - `"GPU KV cache size: X.XX GB"` (or equivalent log line) → `kv_cache_vram_mb`; `full_kv_vram_mb = weights_vram_mb + kv_cache_vram_mb`
  - Values stored in memory; used for all subsequent scheduling decisions for this model.
- Readiness: poll `GET http://<host>:<port>/health` with 2s intervals until 200 or configurable timeout (default 300s).
- If readiness timeout is reached: `SIGKILL` the process, release port, mark `UNLOADED`, return 503 to the waiting request.

### Stopping a vLLM process (UNLOADED transition)

1. Check `active_request_count == 0`. If not, wait.
2. Send `SIGTERM`. Wait up to 30s for clean exit.
3. If still running: `SIGKILL`.
4. Release port. Mark `UNLOADED`.

### SLEEP1 transition

1. Check `active_request_count == 0`.
2. `POST http://<host>:<port>/sleep?level=1` (mode defaults to `abort`).
3. Poll `GET /is_sleeping` until `{"is_sleeping": true}` or timeout (30s).
4. Release `full_kv_vram_mb` from VRAM accounting; charge `weights_vram_mb` to CPU RAM accounting.
5. Mark `SLEEP1`.

### Wake from SLEEP1

1. `POST http://<host>:<port>/wake_up` (no tags = wake all).
2. Poll `GET /is_sleeping` until `{"is_sleeping": false}` or timeout (60s).
3. Restore VRAM accounting; release CPU RAM accounting.
4. Mark `ACTIVE`.

---

## 9. Memory Accounting

### VRAM

- `group.free_vram_mb` starts at `group.total_vram_mb` (from config).
- Cross-checked at startup using `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` — warning logged if declared value differs from measured by more than 5%.
- Decremented by `full_kv_vram_mb` when a model enters `LOADING` or `ACTIVE`.
- Restored when a model enters `SLEEP1` or `UNLOADED`.

### CPU RAM

- `free_cpu_ram_mb` is read at orchestrator startup from `/proc/meminfo` (`MemAvailable` line).
- Decremented by `weights_vram_mb` when a model enters `SLEEP1`.
- Restored when a model exits `SLEEP1` (either to `ACTIVE` or `UNLOADED`).
- Not re-read from `/proc/meminfo` after startup — tracked entirely by the orchestrator's own accounting. This is intentional: it avoids surprises from other processes consuming RAM, and keeps the logic deterministic.

### `weights_vram_mb` and `full_kv_vram_mb`

- Not in config. Auto-measured at first launch of each model by parsing vLLM stdout.
- Cached in memory for the orchestrator process lifetime.
- Before a model's first measurement (during first LOADING), the orchestrator reserves a placeholder using `group.total_vram_mb * 0.85` to avoid double-booking. The actual value replaces this once measured.
- If log parsing fails (no matching lines found within 60s), log a warning and use the placeholder value.

```go
type modelMemory struct {
    weightsVRAM_MB int64 // measured from vLLM startup log
    fullKVVRAM_MB  int64 // measured from vLLM startup log
    measured       bool
}
```

---

## 10. Package Structure

Flat layout — one package until the codebase clearly needs splitting:

```
vLLM-orchestrator/
├── main.go          # flag parsing, config load, signal handling, start server
├── config.go        # Config struct, YAML parsing, startup validation
├── proxy.go         # HTTP reverse proxy, WebSocket proxy, body peek, route dispatch
├── state.go         # model state machine, TTL timers, transition functions
├── scheduler.go     # freeing memory rules, GPU group assignment
├── process.go       # vLLM subprocess launch, log parsing, health poll, teardown, sleep/wake
├── memory.go        # VRAM + CPU RAM accounting; groupState
├── models.go        # /v1/models aggregation + UNLOADED model stub generation
├── go.mod
└── documents/
    └── plan_vllm_orchestrator.md
```

All files are in `package main`. No internal sub-packages until a second binary is justified.

---

## 11. Concurrency Model

- One goroutine per running vLLM process: stdout log drain (for memory measurement) + stderr drain.
- One goroutine per model: TTL timer loop + state transition controller. Sleeps until timer fires or a request arrives that needs a state check.
- One goroutine per in-flight proxied request: standard Go HTTP handler goroutine.
- Two goroutines per active WebSocket connection: bidirectional copy pair.
- Shared state (`groupState`, model states, `modelMemory`) protected by a single `sync.RWMutex`. Transitions are write-locked; reads are read-locked.
- State transitions that involve HTTP calls to vLLM (`/sleep`, `/wake_up`) are performed under the write lock only for the state update; the HTTP calls themselves are made outside the lock to avoid holding it during network I/O.

---

## 12. Configuration Validation at Startup

Validate and abort with a clear error if:
- Duplicate model names or aliases across any model entries
- `ttl_active >= ttl_inactive`
- A model's `load_at_startup: true` but no GPU group has enough estimated VRAM (using `total_vram_mb * 0.85` as a proxy before measurement)
- Duplicate GPU IDs across groups
- Port range `[vllm_port_start, vllm_port_start + len(models))` overlaps with `listen` port

Warn (not abort) if:
- `nvidia-smi` is not available (memory cross-check skipped)
- A GPU group's declared `total_vram_mb` differs from `nvidia-smi` by more than 5%

---

## 13. What Is Explicitly Out of Scope

- **Request authentication**: All headers (including `Authorization`) forwarded verbatim to vLLM. Auth is vLLM's responsibility.
- **Load balancing across multiple instances of the same model**: One vLLM process per model.
- **Auto-scaling**: Out of scope.
- **OpenAI schema validation or transformation**: Never. Full pass-through.
- **Metrics aggregation**: `/metrics` forwarded to the matching vLLM instance.
- **LoRA route awareness**: LoRA routes forwarded blindly.
- **Dynamic config reload**: Config is read once at startup.
- **vLLM sleep level 2**: Merged with UNLOADED (process terminated).

---

## 14. Implementation Order

1. `config.go` — Config struct, YAML parse, startup validation.
2. `memory.go` — `groupState`, VRAM/CPU RAM accounting, `nvidia-smi` cross-check.
3. `process.go` — subprocess launch, stdout log parsing for memory values, health poll, SIGTERM/SIGKILL teardown, `POST /sleep`, `POST /wake_up`, `GET /is_sleeping` helpers.
4. `state.go` — model state machine, TTL timers, transition logic, request counter tracking.
5. `scheduler.go` — freeing memory rules (§5), GPU group assignment (§4).
6. `proxy.go` — body peek, HTTP `ReverseProxy`, WebSocket proxy, route dispatch.
7. `models.go` — `/v1/models` aggregation + stub generation.
8. `main.go` — wire all components, HTTP server, OS signal handling (SIGTERM → graceful shutdown).

Tests for each file in the same order. Table-driven, parallel, per the global `test-generator` skill.

---

## 15. Key Implementation Risks

| Risk | Mitigation |
|---|---|
| vLLM startup log format changes between versions | Parse log lines defensively; use regex with named groups; fall back to placeholder on parse failure |
| Body buffering breaks large multimodal uploads | `max_request_body_mb` default 256 MB; document; return 413 on exceed |
| TTL timer fires while a request is in-flight | Check `active_request_count == 0` before any sleep/terminate transition; re-arm timer if > 0 |
| vLLM process hangs on startup | Health poll timeout (default 300s); `SIGKILL` fallback; 503 to caller |
| VRAM accounting diverges from reality (e.g. OOM killed, other GPU users) | Log warning if `nvidia-smi` shows more used VRAM than orchestrator accounts for; do not auto-correct (deterministic accounting is the design) |
| Port not released if orchestrator crashes | On startup, scan the port pool and kill any stray vLLM processes by checking `/proc/<pid>/cmdline` |
| WebSocket proxy leaks connections | Both sides closed in `sync.Once` + `defer`; context with configurable timeout |
| `POST /sleep` call races with in-flight streaming response | Drain check: only sleep after `active_request_count` transitions to 0 with a short quiesce wait (e.g. 500ms) |
| `VLLM_SERVER_DEV_MODE=1` security exposure | Document that vLLM processes should not be exposed directly; only the orchestrator port is publicly accessible |
