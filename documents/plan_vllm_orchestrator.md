# Plan: vLLM Orchestrator in Go

**Date**: 2026-07-08  
**Module**: `github.com/kippsherer/vLLM-orchestrator`  
**Goal**: Model-on-demand vLLM process lifecycle manager with a transparent pass-through reverse proxy. Not an API gateway — every byte of every request is forwarded to vLLM unmodified.

---

## 1. What "transparent pass-through" means here

The router **must not** inspect, validate, or transform request or response bodies. The only body read is extraction of the top-level `"model"` string for routing. After routing, the raw body bytes are streamed to vLLM and the raw response bytes are streamed back.

This preserves:
- vLLM-specific sampling params (`best_of`, `use_beam_search`, `top_k`, `min_p`, `repetition_penalty`, `length_penalty`, `stop_token_ids`, `include_stop_str_in_output`, `ignore_eos`, `min_tokens`, `truncate_prompt_tokens`, `enable_prefix_caching`, `prompt_logprobs`, `skip_special_tokens`, `spaces_between_special_tokens`, `spaces_between_special_tokens`)
- Guided decoding (`guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `guided_decoding_backend`, `guided_whitespace_pattern`)
- Multimodal bodies (`multi_modal_data`, content arrays with `image_url` entries)
- Non-OpenAI endpoints (`/v1/tokenize`, `/v1/detokenize`, `/v1/score`, `/tokenize`, `/detokenize`)
- LoRA management routes (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/v1/list_lora_adapters`)
- Prometheus metrics at `/metrics`
- WebSocket at `/v1/realtime`
- vLLM dev/debug endpoints (`/version`, `/ping`, `/docs`, `/redoc`, `/openapi.json`)
- vLLM's extended `/v1/models` fields (`root`, `parent`, `max_model_len`, `per_model_config`)
- Streaming SSE chunks including `usage.prompt_tokens_details.cached_tokens`

---

## 2. Configuration

Config is a single YAML file (path provided via `--config` flag or `VLLM_ORCH_CONFIG` env var).

```yaml
listen: ":8080"               # orchestrator bind address

ttl_active: 10m               # idle TTL → sleep1 (CPU offload)
ttl_inactive: 60m             # sleep1 TTL → sleep2 (swap to disk)
ttl_unused: 120m              # sleep2 TTL → process exit

gpu_groups:
  - id: "group0"
    gpus: [0, 1, 2, 3]        # CUDA device IDs
    total_vram_mb: 96000       # sum of group VRAM (authoritative, not queried live)

  - id: "group1"
    gpus: [4]
    total_vram_mb: 24000

models:
  - name: "meta-llama/Meta-Llama-3-8B-Instruct"
    aliases: ["llama3-8b", "llama3"]
    weights_vram_mb: 16000     # VRAM for weights alone (pre-measured)
    full_kv_vram_mb: 22000     # VRAM with full KV cache allocated
    vllm_args:                 # passed verbatim to `vllm serve`
      - "--dtype=float16"
      - "--max-model-len=8192"
      - "--enable-prefix-caching"
      - "--tool-call-parser=hermes"

  - name: "mistralai/Mistral-7B-Instruct-v0.3"
    aliases: ["mistral-7b"]
    weights_vram_mb: 14000
    full_kv_vram_mb: 20000
    vllm_args:
      - "--dtype=float16"
      - "--max-model-len=32768"
```

**Key config rules:**
- `total_vram_mb` is the source of truth — the orchestrator never queries `nvidia-smi` at runtime.
- `weights_vram_mb` and `full_kv_vram_mb` are declared, not probed (pre-measure with a dry run or from vLLM startup logs).
- `vllm_args` are appended to `vllm serve <model_name>` exactly as written. The orchestrator auto-injects `--port`, `--host`, and `--tensor-parallel-size` (from the gpu_group assignment); everything else is the user's responsibility.
- TTLs apply globally; per-model TTL overrides can be added later.

---

## 3. Model State Machine

Each model has exactly one of these states at any moment:

```
UNLOADED ──load──► LOADING ──ready──► ACTIVE ──ttl_active──► SLEEP1 ──ttl_inactive──► SLEEP2 ──ttl_unused──► UNLOADED
                                        ▲                       │                         │
                                        └──request──────────────┘                         │
                                        └──request────────────────────────────────────────┘
```

| State | Description | VRAM held | CPU RAM held | vLLM process |
|---|---|---|---|---|
| `UNLOADED` | No resources | 0 | 0 | Not running |
| `LOADING` | Process starting, not ready | `full_kv_vram_mb` (reserved) | 0 | Starting |
| `ACTIVE` | Serving requests | `full_kv_vram_mb` | 0 | Running |
| `SLEEP1` | CPU offloaded (vLLM `--cpu-offload-gb`) | 0 | `full_kv_vram_mb` | Running (paused) |
| `SLEEP2` | Weights swapped to disk | 0 | `weights_vram_mb` | Running (suspended) |

**Transition triggers:**
- `ACTIVE → SLEEP1`: last request finished + no pending requests + `ttl_active` elapsed + enough free CPU RAM exists
- `ACTIVE → SLEEP2`: `ttl_active` elapsed but insufficient CPU RAM (skip sleep1)
- `SLEEP1 → SLEEP2`: `ttl_inactive` elapsed since entering SLEEP1
- `SLEEP2 → UNLOADED`: `ttl_unused` elapsed since entering SLEEP2
- `SLEEP1/SLEEP2 → ACTIVE`: incoming request (triggers memory freeing if needed)
- `UNLOADED → LOADING`: incoming request (triggers memory freeing if needed)

**The TTL clock resets** on every successfully forwarded response for that model.

---

## 4. GPU Group Assignment

When a model transitions from `UNLOADED` to `LOADING`:

1. Find all GPU groups where `free_vram_mb >= model.full_kv_vram_mb`.
2. Among qualifying groups, select the one with the **smallest** `total_vram_mb`. (Bin-packing heuristic: fit the model into the smallest box it fits, preserving large groups for large models.)
3. If no group qualifies, execute the **freeing memory rules** (§5) on each candidate group individually before selecting.
4. Inject `--tensor-parallel-size=<len(group.gpus)>` and `CUDA_VISIBLE_DEVICES=<group.gpus joined by comma>` into the vLLM subprocess environment.

`free_vram_mb` per group is tracked in memory:

```
group.free_vram_mb = group.total_vram_mb - sum(full_kv_vram_mb for all ACTIVE/LOADING models on this group)
```

---

## 5. Freeing Memory Rules

Applied **per GPU group** individually, only when a group has insufficient `free_vram_mb` to load a requested model. Applied in order, smallest model first (by `full_kv_vram_mb`).

**Precondition**: Models with in-flight requests are never touched.

**Rule 1 — Move ACTIVE models to SLEEP1 (if enough CPU RAM available)**  
- If `free_cpu_ram_mb >= model.full_kv_vram_mb`, signal the vLLM process to offload weights to CPU (via `--cpu-offload-gb` or a restart with flag), mark model as `SLEEP1`.  
- Free the VRAM reservation.  
- Stop as soon as the target group has enough free VRAM.

**Rule 2 — Terminate SLEEP2 processes**  
- Call `SIGTERM` on the vLLM process for the smallest SLEEP2 model on the group.  
- Release its `weights_vram_mb` from CPU RAM tracking.  
- Mark model `UNLOADED`.  
- Stop as soon as enough VRAM is freed.

**Rule 3 — Move SLEEP1 models to SLEEP2**  
- Signal vLLM to swap weights from CPU to disk (or restart with `--swap-space` configured), mark model `SLEEP2`.  
- Free the CPU RAM reservation.  
- Stop as soon as enough CPU RAM is freed for the pending Rule 1.

**Rule 4 — Repeat Rule 2** to cover any remaining VRAM shortfall after Rule 3.

**Cycle**: `1 → 2 → 3 → 4`. If after all four rules there is still insufficient memory, return a 503 (see §6).

---

## 6. Request Handling Flow

```
HTTP/WS request arrives
         │
         ▼
Extract "model" field from body (JSON peek: read until "model" key found, then put bytes back)
         │
         ├── No model field? → 400 Bad Request
         │
         ▼
Resolve model: exact name match → alias list match → not found → 404
         │
         ▼
Is model ACTIVE?
  YES → acquire semaphore slot (limit concurrent requests per model if configured) → forward pass-through
  NO  →
         │
         ▼
Attempt state transition to ACTIVE (run freeing memory rules if needed)
         │
         ├── Memory freed, now loading → wait for LOADING→ACTIVE transition (poll health endpoint)
         │       → forward pass-through once ready
         │
         └── Cannot free enough memory → 503 Service Unavailable
                 body: {"error": {"message": "Insufficient VRAM to load model", "type": "ServiceUnavailableError", "code": 503}}
```

**No request queuing.** The 503 is immediate if resources cannot be freed. Clients retry.

---

## 7. Transparent Proxy Implementation

### 7A. HTTP (all routes except `/v1/realtime`)

Use `net/http/httputil.ReverseProxy` with a custom `Director` that:
1. Sets `req.URL.Scheme`, `req.URL.Host` to the target vLLM process address.
2. Does **not** modify headers, body, query params, or path.
3. Sets `X-Forwarded-For` (standard reverse proxy behavior).

For SSE (streaming) responses, `httputil.ReverseProxy` handles chunked transfer encoding natively — no special code needed.

The `ModifyResponse` hook is left as a no-op. The `ErrorHandler` returns a 502 if the upstream vLLM process is unreachable.

### 7B. WebSocket (`/v1/realtime`)

`httputil.ReverseProxy` does not handle WebSocket upgrades. Use a custom handler:

1. Detect `Upgrade: websocket` header.
2. Dial a raw TCP connection to the target vLLM process.
3. Perform the WebSocket handshake forward (pass the `Upgrade` request as-is).
4. Bidirectional `io.Copy` goroutines between client and upstream connections.
5. Close both connections when either side closes.

### 7C. Body peeking for model extraction

The request body must be read once to extract the `"model"` field, then replayed into the upstream request. Approach:

1. Read entire body into a `[]byte` buffer (size-limited by a configurable `max_request_body_mb`, default 100 MB for multimodal).
2. Use `bytes.Index` or a minimal JSON token scan to extract the `"model"` string value. Do not use `encoding/json` to unmarshal the full body — that would re-serialize and potentially drop unknown fields.
3. Construct `io.NopCloser(bytes.NewReader(buf))` as the new body for the upstream request.

> **Critical**: After buffering, set `Content-Length` on the upstream request to `len(buf)` to avoid chunked encoding issues with vLLM's FastAPI server.

### 7D. Routes owned by the orchestrator itself

These are handled by the orchestrator directly and **not** forwarded to any vLLM instance:

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | 200 if orchestrator is up; checks at least one model is in non-UNLOADED state optionally |
| `GET` | `/ping` | Always 200 `{"status":"ok"}` |
| `GET` | `/v1/models` | Aggregate from all ACTIVE/SLEEP1/SLEEP2 vLLM `/v1/models` responses + UNLOADED model stubs |
| All others | `/*` | Route to the appropriate vLLM instance (see §7A/7B) |

For `/v1/models`, the orchestrator queries each running vLLM's `/v1/models` and merges the arrays. For UNLOADED models, it synthesizes a minimal entry using config data.

---

## 8. vLLM Process Management

### Startup

At startup, the orchestrator:
1. Reads config.
2. For models listed under `available_at_startup`, immediately begins the LOADING sequence.
3. For all other models, initializes them in `UNLOADED` state.
4. Does **not** launch vLLM processes for non-startup models until a request arrives.

### Launching a vLLM process

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <model_name> \
  --host <bind_host> \
  --port <assigned_port> \
  --tensor-parallel-size 4 \
  [user-provided vllm_args...]
```

- Port assignment: orchestrator maintains a pool of ports (`vllm_port_start` in config, default 9000+). Each model gets a unique port from this pool.
- The process is started with `os/exec.Cmd`. Stdout/stderr are written to a per-model log file or to the orchestrator's structured logger.
- Readiness: poll `GET http://<host>:<port>/health` until 200 or timeout (configurable, default 300s).

### Stopping a vLLM process

Send `SIGTERM`. If the process does not exit within 30s, send `SIGKILL`. Clean up port from pool.

### Sleep1 implementation note

vLLM does not natively support a "pause" command. Sleep1 requires either:
- Option A: Restart vLLM with `--cpu-offload-gb=<N>` so it offloads weights to CPU RAM. This moves VRAM to CPU but keeps the process alive and warm.
- Option B: Send `SIGSTOP` to suspend the process (stops all threads) — this freezes it in place but doesn't actually move VRAM.

**Recommend Option A** for Sleep1 (CPU offload restart). This is slower to resume but actually frees GPU VRAM. Document in config that `sleep1` requires vLLM to support `--cpu-offload-gb`.

Sleep2: Terminate the process. Model weights remain on disk (the original model path). Resuming re-launches vLLM from disk.

> **Implication**: Transitions to SLEEP1/SLEEP2 involve a process restart, not a signal. The orchestrator must drain in-flight requests before initiating the transition (wait for active request count to reach 0, then start the TTL countdown).

---

## 9. Memory Accounting

The orchestrator tracks memory **entirely in config-declared values** — no live nvidia-smi calls.

```go
// Per GPU group, maintained in memory:
type groupState struct {
    freeVRAM_MB   int64   // = total_vram_mb - sum(full_kv_vram_mb for ACTIVE/LOADING models)
    usedCPURAM_MB int64   // = sum(full_kv_vram_mb for SLEEP1 models) + sum(weights_vram_mb for SLEEP2 models)
}
```

Free CPU RAM is checked at startup by reading `/proc/meminfo` (Linux) and then tracked in-memory by the same accounting pattern.

---

## 10. Package Structure

Flat layout — one package until the codebase clearly needs splitting:

```
vLLM-orchestrator/
├── main.go              # flag parsing, config load, start server
├── config.go            # config struct, YAML parsing, validation
├── proxy.go             # HTTP reverse proxy, WebSocket proxy, body peek
├── state.go             # model state machine, transitions, TTL timers
├── scheduler.go         # freeing memory rules, GPU group assignment
├── process.go           # vLLM subprocess launch, health poll, teardown
├── memory.go            # memory accounting (VRAM + CPU RAM tracking)
├── models.go            # /v1/models aggregation + stub generation
├── go.mod
└── documents/
    └── plan_vllm_orchestrator.md
```

All files are in `package main`. No internal sub-packages until a second binary is justified.

---

## 11. Concurrency Model

- One goroutine per running vLLM process: health-poll loop + log drain.
- One goroutine per model: TTL timer + state transition controller.
- One goroutine per in-flight proxied request: net/http handler goroutine (standard Go HTTP server).
- One goroutine for WebSocket connections: pair of bidirectional copy goroutines per connection.
- Shared state (`groupState`, model states) protected by a single `sync.RWMutex`. Transitions are write-locked, reads are read-locked.

No channels for state transitions — a mutex is sufficient here and avoids deadlock complexity.

---

## 12. Configuration Validation at Startup

Validate and abort with a clear error if:
- A model's `full_kv_vram_mb > group.total_vram_mb` for all groups (model can never fit anywhere)
- A model lists a GPU group that doesn't exist in `gpu_groups`
- Duplicate model names or aliases
- `ttl_active >= ttl_inactive` or `ttl_inactive >= ttl_unused`
- Port pool exhausted (num models > available ports)

---

## 13. What Is Explicitly Out of Scope

- **Request authentication / API key forwarding logic**: The orchestrator passes all headers including `Authorization` verbatim to vLLM. Auth is vLLM's responsibility.
- **Load balancing across multiple vLLM instances of the same model**: Each model maps to exactly one vLLM process at a time.
- **Auto-scaling** (spawning multiple copies of the same model): Out of scope.
- **OpenAI schema validation or transformation**: Never. Full pass-through.
- **Metrics aggregation**: `/metrics` is forwarded to the matching vLLM instance.
- **LoRA route awareness**: LoRA management routes are forwarded blindly to the model's vLLM process.
- **Dynamic config reload**: Config is read once at startup.

---

## 14. Implementation Order

1. `config.go` — parse and validate YAML config; define all structs.
2. `memory.go` — VRAM/CPU RAM accounting; `groupState` tracker.
3. `process.go` — launch, health-poll, teardown vLLM subprocesses.
4. `state.go` — model state machine with TTL timers; transition functions.
5. `scheduler.go` — freeing memory rules; GPU group assignment.
6. `proxy.go` — transparent HTTP + WebSocket reverse proxy; body peek.
7. `models.go` — `/v1/models` aggregation.
8. `main.go` — wire everything together; HTTP server; signal handling.

Tests for each file follow the same order. Use table-driven tests per the global `test-generator` skill conventions.

---

## 15. Key Implementation Risks

| Risk | Mitigation |
|---|---|
| Body buffering breaks large multimodal uploads | Set `max_request_body_mb` high (default 100 MB); document the limit |
| Sleep1 via CPU-offload restart drops in-flight requests | Drain active requests before triggering any sleep transition |
| TTL timer fires while a request is in-flight | Check `active_request_count == 0` before any state transition; re-arm timer if count > 0 |
| vLLM process hangs during startup | Configurable readiness timeout; SIGKILL fallback |
| GPU group free VRAM drifts from reality | VRAM accounting is config-driven; document that users must provide accurate `full_kv_vram_mb` values |
| Port collision if vLLM crashes and port is not released | Track port → PID mapping; release port on any process exit signal |
| WebSocket proxy leaks connections | Close both sides in a `defer`; use a context with timeout |
