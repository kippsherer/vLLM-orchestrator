# AGENTS.md — vLLM-orchestrator

**Purpose**: Coordinate vLLM processes and routing to provide model-on-demand behavior (Ollama-like UX, vLLM performance).  
**Language**: Go (greenfield — no modules or code committed yet as of 2026-07-08).

---

## Critical Quirks

- **`go.work` is gitignored** — do not commit it. Any Go workspace file is local-only. When the project grows into a multi-module layout, each module must have its own `go.mod`; the workspace file stays off VCS.
- **`.env` is gitignored** — runtime config comes from environment variables. Never hardcode endpoints or secrets.
- **No Makefile, no CI, no build scripts exist yet.** Do not invent or assume commands beyond standard `go` toolchain.

## Module Path (to be created)

When initializing: `github.com/kippsherer/vLLM-orchestrator`

## Commands

Once `go.mod` exists:

```sh
go test ./...          # run all tests
go test -run TestFoo ./pkg/...   # run a single test
gofmt -w .             # format (required before commit)
goimports -w .         # fix imports
golangci-lint run      # lint (add .golangci.yml if needed)
```

`go build` and `go test -c` are **forbidden** per global engine constraints — use `go run` or `go test` directly.

## Global Constraints That Apply Here

The global OpenCode engine rules at `~/.config/opencode/AGENTS.md` govern this session. Key points:

- Never execute `git commit` or `git push`.
- Never deploy or trigger CI/CD.
- No hardcoding `127.0.0.1` / `localhost`.
- Schema/struct discovery: read Go source, not live DB queries.
- Database target: native PostgreSQL only (no ORM abstraction leaks).

## Architecture (TBD)

No code exists yet. When building:

- vLLM processes speak OpenAI-compatible HTTP — use that as the upstream interface.
- Model-on-demand implies a process lifecycle manager (start/stop/idle vLLM workers per model) and a reverse-proxy/router layer.
- Prefer flat package structure until complexity demands otherwise.
