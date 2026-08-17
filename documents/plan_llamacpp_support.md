# Plan: llama.cpp Backend Support

**Date**: 2026-07-20
**Module**: `github.com/kippsherer/vLLM-orchestrator`
**Goal**: Add `llama.cpp` (`llama-server`) as a second selectable engine, alongside vLLM, configured per-model via a new `engine` field. From a client's point of view the OpenAI-compatible API surface, routing, aliasing, and `/v1/models` behavior must be identical regardless of which engine backs a given model. GPU-group scheduling, TTL-based lifecycle, and the transparent pass-through proxy design stay unified across both engines in one Go binary — no separate router process, no per-engine gateway.

**Code discipline**: same as the base plan — flat, dense, no speculative abstractions. Engine differences are handled with plain `if cfg.Engine == engineLlamaCpp { ... } else { ... }` branches at the small number of call sites that are genuinely engine-specific (launch, env vars, socket dir, TTL/eviction logic). No interface/strategy-pattern abstraction is introduced — two engines does not justify one, and a third engine is not anticipated.

---

## 0. Research basis (verified, sourced — no guessed facts)

All llama.cpp-specific technical claims below were fetched live from the official `ggml-org/llama.cpp` GitHub repository on 2026-07-20. Sources are cited per claim. Anything not sourced this way is marked **UNVERIFIED** and called out explicitly as a pre-implementation verification step, never assumed.

| # | Fact | Source |
|---|---|---|
| 1 | `--host HOST` — binds a Unix domain socket instead of TCP if the address **ends with `.sock`**. Otherwise binds `host:port` TCP (default `127.0.0.1:8080`). | `tools/server/README.md` (master), Debian manpages |
| 2 | `-m, --model FNAME` — path to a GGUF model file. | `tools/server/README.md` |
| 3 | `-a, --alias STRING` — comma-separated model name aliases, used by the API (i.e. the `id` returned in `/v1/models`). | `tools/server/README.md` |
| 4 | `-ngl, --gpu-layers, --n-gpu-layers N` — max layers to store in VRAM; accepts an exact integer, `auto`, or `all`. Default `auto`. | `tools/server/README.md`, `docs/multi-gpu.md` |
| 5 | `-sm, --split-mode {none,layer,row,tensor}` (default `layer`), `-ts, --tensor-split N0,N1,...`, `-mg, --main-gpu INDEX` (default 0), `-dev, --device <dev1,dev2,..>` — multi-GPU layer/tensor split controls. | `docs/multi-gpu.md` |
| 6 | `-t, --threads N` (generation), `-tb, --threads-batch N` (batch/prompt processing, defaults to `--threads`). | `tools/server/README.md` |
| 7 | `CUDA_VISIBLE_DEVICES` is respected exactly as with any CUDA program: "if you set it, llama.cpp only sees the specified GPUs." `--device` selects among visible devices and works for any backend. | `docs/multi-gpu.md` (explicit statement) |
| 8 | `GET /health` (also `/v1/health`): **503** `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}` while loading; **200** `{"status":"ok"}` once ready. Public, no API-key check. | `tools/server/README.md` |
| 9 | Native idle-sleep exists: `--sleep-idle-seconds SECONDS` (default `-1` = disabled). "When the server enters sleep mode, the model and its associated memory (including the KV cache) are unloaded... Any new incoming task will automatically trigger the model to reload." Sleep status queryable via `GET /props`. `GET /health`, `GET /props`, `GET /models` do **not** count as activity and do **not** trigger reload. | `tools/server/README.md` §"Sleeping on Idle", PR #18228 |
| 10 | `--sleep-idle-seconds` does **not** terminate the subprocess and does **not** fully release the CUDA context: open issue reports ~600 MiB/GPU residual after sleep triggers, and a separate feature request for subprocess-level idle termination (`--stop-idle-seconds`) remains open/unimplemented. | GitHub issues #19379, #18189 |
| 11 | `llama-server` builds via CMake (`cmake --build build --config Release -t llama-server`), binary at `./build/bin/llama-server`. This is the current name (not `server`). | `tools/server/README.md` "Build" section |
| 12 | `server.cpp` installs a `signal_handler` / `shutdown_handler` (source confirmed in `tools/server/server.cpp`). Exact SIGTERM-vs-SIGINT registration not itself fetched; one third-party anecdote claims SIGTERM causes "unclean shutdown" on their build/version vs. clean SIGINT. **Not fully verified either way.** | `tools/server/server.cpp` (partial); anecdote unverified |
| 13 | OpenAI-compatible routes: chat completions, "responses", embeddings; also Anthropic Messages API compatibility, reranking, `/completion` (native, non-OAI), `/slots` (default enabled, `--no-slots` to disable), `/metrics` (Prometheus, **disabled by default**, needs `--metrics`), `/props` (GET always available for status; POST gated by `--props`). | `tools/server/README.md` |
| 14 | A separate built-in **"router mode"** exists (launch `llama-server` with no `-m`, manage multiple models via `--models-dir`/`--models-preset`/`--models-max`/`--models-autoload` and a `/models` HTTP management API). This is a llama.cpp-native alternative multi-model manager. | `tools/server/README.md` §"Using multiple models" |

**Design consequence of #14**: we do **not** use llama.cpp's router mode. Every model — vLLM or llama.cpp — gets exactly one `llama-server`/`vllm serve` subprocess launched and lifecycle-managed by our own orchestrator, exactly as today. This is required to keep GPU-group scheduling, TTL semantics, and alias resolution unified and identical from a client's point of view, per the stated goal. Router mode is mentioned here only so it is not "rediscovered" and mistakenly adopted later.

**Design consequence of #9/#10**: llama.cpp's own `--sleep-idle-seconds` is a fully opaque, self-managed, orchestrator-invisible optimization — it requires zero orchestrator code awareness. If a user opts into it via `llama_cpp_args`, requests naturally wake the model on the next proxied call (no `/wake_up`-equivalent call is needed, since llama.cpp handles this internally). It does **not** replace the orchestrator's own state machine — our TTL loop still separately governs ACTIVE→UNLOADED (process kill) for llama.cpp models, since llama.cpp cannot free the full GPU footprint or the process's residency on its own. This is documented as optional, with the known VRAM-residual caveat from issue #19379, not recommended or defaulted.

**Not independently re-verified, kept for context only (from earlier general knowledge, not sourced this session)**: none — every technical claim used in the design below is either sourced above or explicitly marked as a decision (not a fact).

---

## 1. User decisions locked in (do not re-litigate at implementation time)

| Decision point | Choice |
|---|---|
| GPU/CPU layer-split config shape | **Raw passthrough** via `llama_cpp_args: []string`, exactly like existing `vllm_args`. No structured `n_gpu_layers`/`tensor_split` fields, no orchestrator-side derivation or validation of these flags. |
| VRAM accounting for llama.cpp models | **`vram_allocation` is authoritative from launch**, `mem.measured = true` set synchronously right after process start. No stdout log parsing is attempted for llama.cpp (log format differs from vLLM and is version-dependent; avoids guessing). |
| Idle/lifecycle model for llama.cpp | **Two states only: ACTIVE ↔ UNLOADED.** No SLEEP1/SLEEP2 for llama.cpp models. Idle timeout uses **`ttl_unused`** (the longest of the three TTLs) as the single threshold before the process is killed. `ttl_active`/`ttl_inactive` are accepted in config (for ordering-validation consistency) but semantically unused for llama.cpp models. |
| Model file path | New field, GGUF file resolved as `filepath.Join(llama_cpp_model_dir, model.gguf_path)`. Base directory is configurable (`llama_cpp_model_dir`), not hardcoded — user's real value is `/data/llama/models`, but this must be a config field, not a Go source constant, per the no-hardcoded-paths constraint. |
| Binary location | Hardcoded `"llama-server"`, resolved via `$PATH` — no config field, mirrors existing hardcoded `"vllm"` in `process.go`. |
| Unix socket directory | **Separate field**, `llama_cpp_socket_dir`, parallel to and independent from `vllm_socket_dir`. User confirmed `/run/llama` already exists on both `ai01` and `db02` with the same shape as `/run/vllm`. `vllm_socket_dir` is **not** renamed and **not** reused for llama.cpp sockets. |
| Freeing-cascade eviction of idle ACTIVE llama.cpp models | **Allowed.** If `activeRequests == 0` right now (regardless of elapsed idle duration), the memory-freeing cascade may kill an idle ACTIVE llama.cpp model to make room for another model's launch on the same GPU group — same as it already kills idle SLEEP2 vLLM models today. This is a real behavior addition to the freeing cascade, not just an engine-gated no-op. |

---

## 2. Config schema changes

### 2.1 New top-level `Config` fields (`config.go`)

```go
type Config struct {
    Listen            string        `yaml:"listen"`
    VLLMSocketDir     string        `yaml:"vllm_socket_dir"`
    LlamaCppSocketDir string        `yaml:"llama_cpp_socket_dir"` // new; required iff any model has engine: llama_cpp
    LlamaCppModelDir  string        `yaml:"llama_cpp_model_dir"`  // new; base dir for gguf_path resolution; required iff any model has engine: llama_cpp
    QueueDepth        int           `yaml:"queue_depth"`
    TTLActive         time.Duration `yaml:"ttl_active"`
    TTLInactive       time.Duration `yaml:"ttl_inactive"`
    TTLUnused         time.Duration `yaml:"ttl_unused"`
    GPUGroups         []GPUGroup    `yaml:"gpu_groups"`
    Models            []ModelConfig `yaml:"models"`
}
```

### 2.2 New `ModelConfig` fields

```go
const (
    engineVLLM     = "vllm"      // default when Engine is empty
    engineLlamaCpp = "llama_cpp"
)

type ModelConfig struct {
    Name             string        `yaml:"name"`
    Aliases          []string      `yaml:"aliases"`
    Engine           string        `yaml:"engine"`          // new; "" or "vllm" (default) | "llama_cpp"
    LoadAtStartup    bool          `yaml:"load_at_startup"`
    GPUGroup         string        `yaml:"gpu_group"`
    VRAMAllocationMB int64         `yaml:"vram_allocation"` // reused as-is for both engines
    KVCacheMemoryGB  float64       `yaml:"kv_cache_memory"` // vLLM only; validated as unset for llama_cpp
    TTLActive        time.Duration `yaml:"ttl_active"`      // accepted but unused by llama_cpp models
    TTLInactive      time.Duration `yaml:"ttl_inactive"`    // accepted but unused by llama_cpp models
    TTLUnused        time.Duration `yaml:"ttl_unused"`      // sole idle threshold for llama_cpp models
    VLLMArgs         []string      `yaml:"vllm_args"`       // vLLM only
    GGUFPath         string        `yaml:"gguf_path"`       // new; llama_cpp only; joined with llama_cpp_model_dir
    LlamaCppArgs     []string      `yaml:"llama_cpp_args"`  // new; llama_cpp only; raw passthrough, same pattern as vllm_args
}
```

### 2.3 `validateConfig` additions

- `m.Engine` must be `""`, `"vllm"`, or `"llama_cpp"` — reject anything else.
- If `m.Engine == "llama_cpp"`:
  - `cfg.LlamaCppSocketDir` and `cfg.LlamaCppModelDir` must be non-empty (checked once, not per-model).
  - `m.GGUFPath` must be non-empty.
  - `m.KVCacheMemoryGB` must be `0` (reject if set — meaningless for llama.cpp, avoids silent no-op confusion).
  - `m.VLLMArgs` must be empty (reject if set — same reasoning).
  - The resolved file `filepath.Join(cfg.LlamaCppModelDir, m.GGUFPath)` must exist and be a regular file (`os.Stat`) — fail fast at startup rather than at first request, consistent with existing validation philosophy.
- Else (vLLM, default):
  - `m.GGUFPath` and `m.LlamaCppArgs` must be empty (reject if set — catches config mistakes early).
- `cfg.LlamaCppSocketDir`, when non-empty, gets the same writability check (`os.MkdirAll` + temp-file write test) already applied to `cfg.VLLMSocketDir`. Applied only if at least one model uses `llama_cpp`.
- Existing TTL-ordering validation (`ttl_active < ttl_inactive < ttl_unused`, global and effective-per-model) is **unchanged** — still enforced even for llama_cpp models where two of the three values are semantically inert, to avoid special-casing validation logic for no real benefit.

### 2.4 Example `config.yaml` addition

```yaml
llama_cpp_socket_dir: "/run/llama"
llama_cpp_model_dir: "/data/llama/models"

models:
  - name: "unsloth/Qwen3.6-27B-MTP-GGUF"
    aliases: ["qwen3.6-27b-gguf"]
    engine: llama_cpp
    gguf_path: "Qwen3.6-27B-MTP-Q4_K_M.gguf"   # resolved as /data/llama/models/Qwen3.6-27B-MTP-Q4_K_M.gguf
    vram_allocation: 22000                     # MB; authoritative, no runtime measurement
    ttl_unused: 60m                            # sole idle-to-kill timer for this model
    llama_cpp_args:
      - "-ngl"
      - "auto"          # or an explicit layer count, or "all"/"0" for full-GPU/full-CPU
      - "-t"
      - "16"
      - "-fa"
      - "on"
```

---

## 3. Process launch (`process.go`)

### 3.1 New `launchLlamaCpp`

Sibling function to `launchVLLM`, same signature shape (returns `*vllmProcess`, reusing that struct as-is — it is already engine-agnostic: `cmd`, `socketPath`, `client`, `exited`, `onExit`; no rename needed for this change to land, though a future cosmetic rename to `engineProcess` could be considered separately and is out of scope here):

```go
func launchLlamaCpp(modelCfg ModelConfig, socketPath string, group *groupState, mem *modelMemory, modelDir string) (*vllmProcess, error) {
    // stale-socket check: identical logic to launchVLLM (checkSocketOwned / os.Remove)

    visibleDevs := ... // same CUDA_VISIBLE_DEVICES construction as launchVLLM, from group.gpus

    aliasNames := append([]string{modelCfg.Name}, modelCfg.Aliases...)
    args := append([]string{
        "-m", filepath.Join(modelDir, modelCfg.GGUFPath),
        "--host", socketPath,       // ends in .sock → llama-server binds a UDS (verified, §0.1)
        "-a", strings.Join(aliasNames, ","),
    }, modelCfg.LlamaCppArgs...)

    cmd := exec.Command("llama-server", args...)
    cmd.Env = buildEnvLlamaCpp(cudaVisible)

    // stdout/stderr pipes, cmd.Start(), reap goroutine: identical pattern to launchVLLM

    // VRAM accounting per user decision: authoritative immediately, no log parsing
    mem.fullKVVRAMMB = modelCfg.VRAMAllocationMB
    mem.measured = true

    go drainAndMeasure(stdout, modelCfg.Name, nil) // mem=nil: no regex parsing attempted, still logs
    go drainAndMeasure(stderr, modelCfg.Name, nil)
    // reap goroutine identical to launchVLLM

    return vp, nil
}
```

Notes:
- `--port` is deliberately **not** passed. Whether `llama-server` requires `--port` to be entirely absent (vs. harmlessly ignored) when `--host` ends in `.sock` is **not explicitly confirmed** by the fetched docs (only that `--host` ending in `.sock` triggers UDS binding) — this is a one-line behavior to confirm with a real test run before merging, not a redesign risk.
- `--sleep-idle-seconds` is **not** auto-injected. It is left entirely to the user via `llama_cpp_args` if desired (see §0 design consequence).

### 3.2 New `buildEnvLlamaCpp`

Deliberately minimal — unlike vLLM's `buildEnv`, we do **not** inject any of the PyTorch/vLLM-specific tuning vars (`VLLM_SERVER_DEV_MODE`, `OMP_NUM_THREADS`, `VLLM_CPU_OMP_THREADS_BIND`, `LD_PRELOAD` tcmalloc, `VLLM_USE_FASTOKENS`) — none of these are shown by the sourced docs to apply to or benefit `llama-server`, and injecting them would be an unverified guess. Only CUDA device visibility is set, mirroring the same filtering logic already in `buildEnv`:

```go
func buildEnvLlamaCpp(cudaVisible string) []string {
    base := os.Environ()
    out := make([]string, 0, len(base)+2)
    for _, kv := range base {
        k := kv
        if i := strings.IndexByte(kv, '='); i >= 0 { k = kv[:i] }
        if k == "CUDA_VISIBLE_DEVICES" || k == "CUDA_DEVICE_ORDER" { continue }
        out = append(out, kv)
    }
    return append(out, "CUDA_VISIBLE_DEVICES="+cudaVisible, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
}
```

Follow-up (not part of this change, noted for later): `llama-server`'s log level markers use single-letter prefixes (`D`, `I`, `W`, `E` — confirmed from real log excerpts in GitHub issue #24475, e.g. `0.00.497.696 D srv load_model: ...`), not the padded `" WARNING "` / `" ERROR "` strings vLLM uses. The existing `drainAndMeasure`'s "important line" filter (which promotes `tokens/s`/`WARNING`/`ERROR` lines to always-on logging) will **not** catch llama.cpp warnings/errors outside `--verbose` mode. This is a cosmetic logging gap, not a correctness issue (all lines still drain and log under `--verbose`), and is left as a documented follow-up rather than guessed at now.

### 3.3 Reused unchanged

- `waitForHealth` — reused as-is. `GET /health` semantics (503 loading → 200 ready) match vLLM's exactly (§0.1 item 8).
- `killProcess` — reused as-is (SIGTERM, 30s grace, SIGKILL, socket removal). Safe regardless of the unconfirmed SIGTERM-nuance in §0 item 12 — worst case, llama-server hits the SIGKILL fallback instead of exiting cleanly within the grace period.
- `checkSocketOwned` — reused as-is.
- `sleepModel` / `wakeModel` / `pollIsSleeping` — **never called for llama_cpp models**. These remain vLLM-only; call sites are gated by engine checks (§4), not by changes to these functions.

---

## 4. State machine changes (`state.go`)

### 4.1 Socket path construction (`newOrchestrator`)

```go
socketDir := cfg.VLLMSocketDir
if mc.Engine == engineLlamaCpp {
    socketDir = cfg.LlamaCppSocketDir
}
socketPath := socketDir + "/" + sanitizeModelName(mc.Name) + ".sock"
```

### 4.2 Launch dispatch (`handleRequest`, `stateUnloaded` branch)

Replace the single `launchVLLM(...)` call with:

```go
var proc *vllmProcess
var err error
if me.cfg.Engine == engineLlamaCpp {
    proc, err = launchLlamaCpp(me.cfg, me.socketPath, o.ms.groups[groupIdx], &me.mem, o.cfg.LlamaCppModelDir)
} else {
    proc, err = launchVLLM(me.cfg, me.socketPath, o.ms.groups[groupIdx], &me.mem)
}
```

Everything downstream (crash-handler registration, `waitForHealth`, `watchHealth`, queue draining) is unchanged and already engine-agnostic.

### 4.3 New small helper: `killAndUnload`

Justified as a genuine duplication-avoidance case (used at 2+ call sites with identical shape after this change — the existing inline block in the `stateSleep2` TTL case, and the new llama.cpp direct-kill path below), not a speculative abstraction:

```go
// killAndUnload terminates proc and marks me UNLOADED. Must be called without me.mu held.
func (o *orchestrator) killAndUnload(me *modelEntry, proc *vllmProcess, reason string) {
    me.mu.Lock()
    if me.proc != proc { me.mu.Unlock(); return }
    me.state = stateUnloaded
    me.proc = nil
    me.assignedGroupIdx = -1
    me.reservedVRAMMB = 0
    me.mu.Unlock()
    killProcess(proc, me.cfg.Name)
    log.Printf("[orchestrator] %s: → UNLOADED (%s)", me.cfg.Name, reason)
}
```

The existing `stateSleep2` case in `tickTTL` is refactored to call this helper instead of its current inline block (behavior-neutral refactor).

### 4.4 `tickTTL` — `stateActive` case gains an engine branch

```go
case stateActive:
    if me.cfg.Engine == engineLlamaCpp {
        ttlUnused := effTTL(me.cfg.TTLUnused, o.cfg.TTLUnused) // reuse existing effTTL closure
        if activeReqs > 0 || idle < ttlUnused {
            return
        }
        me.mu.Lock()
        if me.state != stateActive || me.activeRequests > 0 { me.mu.Unlock(); return }
        proc := me.proc
        me.mu.Unlock()
        o.killAndUnload(me, proc, "ttl_unused elapsed")
        return
    }
    // existing vLLM ACTIVE → SLEEP1/SLEEP2 logic, unchanged
```

The `stateSleep1` and `stateSleep2` cases in `tickTTL` are **never reached for llama_cpp models** (they never transition into those states), so no changes are needed there — they remain vLLM-only in practice without requiring an explicit guard.

---

## 5. Scheduler changes (`scheduler.go`)

### 5.1 `freeMemoryRules` — Rule 1 extended for llama.cpp

Per the locked-in decision (§1), idle ACTIVE llama.cpp models become eviction candidates. The existing Rule 1 loop (`o.activeModels(...)`) is extended: for each candidate, branch on engine.

```go
for _, me := range candidates {
    me.mu.Lock()
    if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs { me.mu.Unlock(); continue }
    if me.cfg.Engine == engineLlamaCpp {
        if me.activeRequests > 0 { me.mu.Unlock(); continue } // llama.cpp kill is destructive: require idle right now
        proc := me.proc
        me.mu.Unlock()

        freeBefore := gs.measuredFreeMB // under ms.mu.RLock, as existing code does
        expectedMB := me.reservedVRAMMB
        o.killAndUnload(me, proc, "evicted to free VRAM")
        waitVRAMStable(o.ms, gs, freeBefore, expectedMB) // reused as-is
        // check free >= neededMB, return true if so — same pattern as Rule 2 below
        continue
    }
    // existing vLLM sleepModel(...,1) logic, unchanged
}
```

This is placed as an additional branch inside the existing Rule 1 loop (not a new Rule 0), since it operates on the same candidate set (`o.activeModels(...)`) and the same "stop as soon as enough VRAM is freed" control flow — no new rule number, no new function, minimal diff.

### 5.2 Rules 2, 3, 4 — unchanged

These operate on `stateSleep2` and `stateSleep1` respectively. llama.cpp models are never in those states, so `idleModels(stateSleep2, ...)` / `idleModels(stateSleep1, ...)` naturally exclude them — no code changes needed.

### 5.3 `assignGroup` / `pickGroup` / `groupHasOtherModels` — unchanged

Already fully generic (operates on `me.mem.fullKVVRAMMB`, `me.cfg.VRAMAllocationMB`, `me.cfg.GPUGroup`). Since llama.cpp models set `mem.measured = true` and `mem.fullKVVRAMMB = vram_allocation` synchronously at launch (§3.1), the existing "placeholder before measurement" logic simply never applies to them (there is no unmeasured window) — no special-casing required.

---

## 6. Proxy layer (`proxy.go`) — no changes required

- `blockedPaths` (`/sleep`, `/wake_up`, `/is_sleeping`) continue to be blocked unconditionally for all models. These are meaningless for llama.cpp-backed models (llama.cpp doesn't implement them) but blocking them uniformly is harmless and requires no engine check.
- Routing, model-field extraction/rewriting, `forwardDirect` (Unix-socket reverse proxy), and the WebSocket tunnel all operate purely on `me.socketPath` and are already fully engine-agnostic.
- `serveModels` (`/v1/models` aggregation): `fetchLiveModels` already degrades gracefully — if a live `GET /v1/models` call to a llama.cpp instance fails or decodes unexpectedly, the code falls through to synthesizing a stub entry from config, exactly as it does today for any transient vLLM failure. Whether `llama-server` implements a plain (non-router-mode) `GET /v1/models` in single-model mode with the same response shape as vLLM is **not explicitly confirmed** by the fetched docs (only router-mode's `/models` management API was documented) — left as a "nice to have, verify at implementation time" item, not a blocker, precisely because of this existing graceful degradation.
- `/metrics?model=` passthrough works automatically via `forwardDirect` provided the user adds `--metrics` to `llama_cpp_args` (disabled by default per §0 item 13) — pure documentation note, no code change.

---

## 7. `main.go` changes

Startup stale-socket cleanup currently walks only `cfg.VLLMSocketDir`. Extend to also walk `cfg.LlamaCppSocketDir` when non-empty, with identical treatment (remove `.sock` files not owned by a live process):

```go
for _, dir := range []string{cfg.VLLMSocketDir, cfg.LlamaCppSocketDir} {
    if dir == "" { continue }
    entries, err := os.ReadDir(dir)
    if err != nil { continue }
    for _, e := range entries {
        if !strings.HasSuffix(e.Name(), ".sock") { continue }
        path := filepath.Join(dir, e.Name())
        if checkSocketOwned(path) != nil {
            os.Remove(path)
            log.Printf("startup: removed stale socket %s", path)
        }
    }
}
```

No other `main.go` changes — the startup model-loading loop (`load_at_startup`) is already generic across `o.models`.

---

## 8. Documentation updates

- `README.md`: new "Engines" section explaining `engine: vllm | llama_cpp`, the two-state (ACTIVE/UNLOADED) lifecycle for llama.cpp models vs. the five-state model for vLLM, the `gguf_path`/`llama_cpp_model_dir` resolution, `llama_cpp_socket_dir`, and the `llama_cpp_args` passthrough convention (mirroring the existing `vllm_args` explanation). Document that `ttl_active`/`ttl_inactive` are accepted-but-inert for llama.cpp models and only `ttl_unused` governs their idle-to-kill timing.
- `config.example.yaml`: add one worked llama.cpp model example (as in §2.4).
- Note the optional, unrecommended-by-default `--sleep-idle-seconds` llama.cpp flag as an available `llama_cpp_args` passthrough option, with the VRAM-residual caveat from issue #19379.

---

## 9. Implementation order

1. `config.go` — new fields, engine constant, validation additions (§2).
2. `process.go` — `launchLlamaCpp`, `buildEnvLlamaCpp` (§3).
3. `state.go` — socket path branch, launch dispatch, `killAndUnload` helper + refactor, `tickTTL` engine branch (§4).
4. `scheduler.go` — Rule 1 extension (§5.1).
5. `main.go` — stale-socket cleanup for the second directory (§7).
6. `README.md` / `config.example.yaml` — documentation (§8).
7. Tests, table-driven, parallel, following existing per-file test conventions (`config_test.go`, `process_test.go`, `state_test.go`, `scheduler_test.go` gain llama.cpp-engine cases alongside existing vLLM cases).

## 10. Pre-implementation verification checklist (real binary, not guessed)

Before merging, confirm against an actually-installed `llama-server` build:

- [ ] `--host <path>.sock` with `--port` omitted binds a UDS as expected (no port conflict, no requirement to also pass `--port`).
- [ ] `GET /health` over the UDS behaves identically to TCP (200/503 semantics unaffected by transport).
- [ ] `-a "name1,name2"` produces the expected `id` value(s) in whichever model-listing endpoint is checked.
- [ ] Confirm SIGTERM behavior in practice (clean exit vs. requiring SIGKILL fallback) — informs whether the existing 30s grace period is suffient or should be tuned down for llama.cpp specifically (cosmetic, not correctness-affecting either way).
- [ ] Capture one real startup log to see current warning/error line markers, to decide later (out of scope for this change) whether `drainAndMeasure`'s "important line" filter should be extended for llama.cpp.

---

## 11. Explicitly out of scope

- llama.cpp's own **router mode** (multi-model built-in management) — not used; see §0 design consequence.
- Structured/validated `n_gpu_layers`/`tensor_split`/`main_gpu`/`split_mode` config fields — raw passthrough only, per §1.
- Any orchestrator-driven use of `--sleep-idle-seconds` / `GET /props` polling — left fully opaque/optional to the user, no orchestrator code touches it.
- Runtime VRAM measurement via llama.cpp log parsing — `vram_allocation` is authoritative, per §1.
- Speculative decoding, multimodal (`mmproj`), LoRA, or any other llama.cpp feature beyond what's needed for basic OpenAI-compatible serving and GPU/CPU layer split — available to the user via `llama_cpp_args` passthrough without any orchestrator-side awareness, same as today's `vllm_args` philosophy.
