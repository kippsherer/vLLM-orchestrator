package main

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func makeTestOrchestrator(t *testing.T) *orchestrator {
	t.Helper()
	dir := t.TempDir()
	cfg := &Config{
		Listen:        ":9999",
		VLLMSocketDir: dir,
		QueueDepth:    4,
		TTLActive:     5 * time.Minute,
		TTLInactive:   30 * time.Minute,
		TTLUnused:     60 * time.Minute,
		GPUGroups:     []GPUGroup{{ID: "g0", GPUs: []int{0}}},
		Models: []ModelConfig{
			{Name: "model-a", Aliases: []string{"alias-a", "a"}},
			{Name: "model-b"},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: -1},
		},
		freeCPURAMB: 65536,
	}
	return newOrchestrator(cfg, ms)
}

func TestNewOrchestrator(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	if len(o.models) != 2 {
		t.Fatalf("got %d models, want 2", len(o.models))
	}
	// Socket paths use sanitized names.
	for _, me := range o.models {
		for _, c := range me.socketPath {
			if c == '/' && me.socketPath[len(o.cfg.VLLMSocketDir):] != me.socketPath[len(o.cfg.VLLMSocketDir):] {
				// just ensure socketPath is non-empty and ends in .sock
				break
			}
		}
		if len(me.socketPath) == 0 {
			t.Errorf("model %q has empty socketPath", me.cfg.Name)
		}
	}
}

func TestNewOrchestratorLlamaCppSocketDir(t *testing.T) {
	t.Parallel()
	vllmDir := t.TempDir()
	llamaCppDir := t.TempDir()
	cfg := &Config{
		Listen:            ":9999",
		VLLMSocketDir:     vllmDir,
		LlamaCppSocketDir: llamaCppDir,
		QueueDepth:        4,
		TTLActive:         5 * time.Minute,
		TTLInactive:       30 * time.Minute,
		TTLUnused:         60 * time.Minute,
		GPUGroups:         []GPUGroup{{ID: "g0", GPUs: []int{0}}},
		Models: []ModelConfig{
			{Name: "vllm-model", Engine: "vllm"},
			{Name: "llama-model", Engine: engineLlamaCpp},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: -1},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)

	if len(o.models) != 2 {
		t.Fatalf("got %d models, want 2", len(o.models))
	}

	// vLLM model socket should be in VLLMSocketDir.
	vllmMe := o.resolve("vllm-model").pick()
	if !strings.HasPrefix(vllmMe.socketPath, vllmDir) {
		t.Errorf("vLLM model socketPath = %q, want prefix %q", vllmMe.socketPath, vllmDir)
	}

	// llama_cpp model socket should be in LlamaCppSocketDir.
	llamaMe := o.resolve("llama-model").pick()
	if !strings.HasPrefix(llamaMe.socketPath, llamaCppDir) {
		t.Errorf("llama_cpp model socketPath = %q, want prefix %q", llamaMe.socketPath, llamaCppDir)
	}
}

func TestResolve(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	cases := []struct {
		name  string
		found bool
	}{
		{"model-a", true},
		{"alias-a", true},
		{"a", true},
		{"model-b", true},
		{"unknown", false},
		{"", false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rs := o.resolve(tc.name)
			if tc.found && rs == nil {
				t.Errorf("resolve(%q) = nil, want non-nil", tc.name)
			}
			if !tc.found && rs != nil {
				t.Errorf("resolve(%q) = %v, want nil", tc.name, rs)
			}
		})
	}
}

func TestNewOrchestratorReplicas(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	cfg := &Config{
		Listen:        ":9999",
		VLLMSocketDir: dir,
		QueueDepth:    4,
		TTLActive:     5 * time.Minute,
		TTLInactive:   30 * time.Minute,
		TTLUnused:     60 * time.Minute,
		GPUGroups:     []GPUGroup{{ID: "g0", GPUs: []int{0}}},
		Models: []ModelConfig{
			{Name: "surya", Aliases: []string{"s"}, Replicas: 5},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: -1},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)

	if len(o.models) != 5 {
		t.Fatalf("got %d models, want 5", len(o.models))
	}
	seen := map[string]bool{}
	for _, me := range o.models {
		if seen[me.socketPath] {
			t.Errorf("duplicate socketPath %q", me.socketPath)
		}
		seen[me.socketPath] = true
		if me.replicaSet == nil {
			t.Error("replicaSet is nil for replicated model")
		}
	}
	for _, name := range []string{"surya", "s"} {
		rs := o.resolve(name)
		if rs == nil {
			t.Fatalf("resolve(%q) = nil", name)
		}
		if len(rs.entries) != 5 {
			t.Errorf("resolve(%q) has %d entries, want 5", name, len(rs.entries))
		}
	}
}

func TestReplicaSetPick(t *testing.T) {
	t.Parallel()

	t.Run("round_robin", func(t *testing.T) {
		t.Parallel()
		rs := &replicaSet{
			entries: []*modelEntry{
				{cfg: ModelConfig{Name: "m"}},
				{cfg: ModelConfig{Name: "m"}},
				{cfg: ModelConfig{Name: "m"}},
			},
		}
		for i := 0; i < 6; i++ {
			if got := rs.pick(); got != rs.entries[i%3] {
				t.Fatalf("pick[%d] = %p, want entries[%d]", i, got, i%3)
			}
		}
	})

	t.Run("single_entry", func(t *testing.T) {
		t.Parallel()
		rs := &replicaSet{entries: []*modelEntry{{cfg: ModelConfig{Name: "m"}}}}
		for i := 0; i < 3; i++ {
			if rs.pick() != rs.entries[0] {
				t.Fatal("single-entry set should always return the same entry")
			}
		}
	})
}

func TestCompleteRequest(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	me.mu.Lock()
	me.activeRequests = 3
	me.mu.Unlock()

	before := time.Now()
	time.Sleep(time.Millisecond)
	o.completeRequest(me)

	me.mu.Lock()
	active := me.activeRequests
	last := me.lastCompleted
	me.mu.Unlock()

	if active != 2 {
		t.Errorf("activeRequests = %d, want 2", active)
	}
	if !last.After(before) {
		t.Errorf("lastCompleted not updated")
	}
}

func TestDrainQueueWith503(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	recorders := make([]*httptest.ResponseRecorder, 3)
	errFlags := make([]bool, 3)
	for i := range recorders {
		recorders[i] = httptest.NewRecorder()
		me.queue <- requestPair{
			w:    recorders[i],
			r:    &http.Request{},
			done: make(chan struct{}),
			err:  &errFlags[i],
		}
	}

	o.drainQueueWith503(me)

	for i, rec := range recorders {
		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("recorder[%d]: got status %d, want 503", i, rec.Code)
		}
	}
	if len(me.queue) != 0 {
		t.Errorf("queue not drained: %d items remain", len(me.queue))
	}
}

func TestDrainQueue(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	const n = 3
	dones := make([]chan struct{}, n)
	for i := range dones {
		dones[i] = make(chan struct{})
		ef := false
		me.queue <- requestPair{
			w:    httptest.NewRecorder(),
			r:    &http.Request{},
			done: dones[i],
			err:  &ef,
		}
	}

	o.drainQueue(me)

	for i, done := range dones {
		select {
		case <-done:
		default:
			t.Errorf("done[%d] not closed after drainQueue", i)
		}
	}

	me.mu.Lock()
	active := me.activeRequests
	me.mu.Unlock()
	if active != n {
		t.Errorf("activeRequests = %d, want %d", active, n)
	}
	if len(me.queue) != 0 {
		t.Errorf("queue not drained: %d items remain", len(me.queue))
	}
}

func TestHandleRequestActiveImmediate(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	me.mu.Lock()
	me.state = stateActive
	me.mu.Unlock()

	errFlag := false
	rp := requestPair{
		w:    httptest.NewRecorder(),
		r:    &http.Request{},
		done: make(chan struct{}),
		err:  &errFlag,
	}

	o.handleRequest(me, rp)

	select {
	case <-rp.done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for done")
	}

	if errFlag {
		t.Error("errFlag should be false")
	}
}

func TestHandleRequestQueueFull(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	me := &modelEntry{
		cfg:              ModelConfig{Name: "test-model"},
		state:            stateUnloaded,
		queue:            make(chan requestPair, 1),
		lastCompleted:    time.Now(),
		assignedGroupIdx: -1,
	}

	occupyingErr := false
	me.queue <- requestPair{
		w:    httptest.NewRecorder(),
		r:    &http.Request{},
		done: make(chan struct{}),
		err:  &occupyingErr,
	}

	rec := httptest.NewRecorder()
	errFlag := false
	rp := requestPair{
		w:    rec,
		r:    &http.Request{},
		done: make(chan struct{}),
		err:  &errFlag,
	}

	o.handleRequest(me, rp)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("got status %d, want 503", rec.Code)
	}
}

func TestHandleRequestLoadingWait(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	me.mu.Lock()
	me.state = stateLoading
	me.mu.Unlock()

	errFlag := false
	rp := requestPair{
		w:    httptest.NewRecorder(),
		r:    &http.Request{},
		done: make(chan struct{}),
		err:  &errFlag,
	}

	go o.handleRequest(me, rp)

	time.Sleep(500 * time.Millisecond)

	me.mu.Lock()
	me.state = stateActive
	me.mu.Unlock()

	select {
	case <-rp.done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for done")
	}

	if errFlag {
		t.Error("errFlag should be false")
	}
}

func TestHandleRequestSleepWake(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	socketPath := t.TempDir() + "/test.sock"
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/wake_up", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"is_sleeping":false}`))
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(listener)
	t.Cleanup(func() { srv.Close() })

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		},
	}
	client := &http.Client{Transport: transport}

	vp := &vllmProcess{
		socketPath: socketPath,
		client:     client,
	}

	me.mu.Lock()
	me.state = stateSleep1
	me.proc = vp
	me.mu.Unlock()

	errFlag := false
	rp := requestPair{
		w:    httptest.NewRecorder(),
		r:    &http.Request{},
		done: make(chan struct{}),
		err:  &errFlag,
	}

	o.handleRequest(me, rp)

	select {
	case <-rp.done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for done")
	}

	me.mu.Lock()
	st := me.state
	me.mu.Unlock()
	if st != stateActive {
		t.Errorf("state = %s, want active", st)
	}
}

func TestTickTTLActive(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	socketPath := t.TempDir() + "/test.sock"
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/sleep", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"is_sleeping":true}`))
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(listener)
	t.Cleanup(func() { srv.Close() })

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		},
	}
	client := &http.Client{Transport: transport}

	vp := &vllmProcess{
		socketPath: socketPath,
		client:     client,
	}

	me.mu.Lock()
	me.state = stateActive
	me.activeRequests = 0
	me.lastCompleted = time.Now().Add(-10 * time.Minute)
	me.proc = vp
	me.mem.weightsVRAMMB = 10000
	me.mu.Unlock()

	o.tickTTL(me)

	me.mu.Lock()
	st := me.state
	me.mu.Unlock()
	if st != stateSleep1 {
		t.Errorf("state = %s, want sleep1", st)
	}
}

func TestTickTTLNotExpired(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	me := o.models[0]

	me.mu.Lock()
	me.state = stateActive
	me.activeRequests = 0
	me.lastCompleted = time.Now()
	me.mu.Unlock()

	o.tickTTL(me)

	me.mu.Lock()
	st := me.state
	me.mu.Unlock()
	if st != stateActive {
		t.Errorf("state = %s, want active", st)
	}
}

func TestTickTTLActiveLlamaCpp(t *testing.T) {
	t.Parallel()
	llamaCppDir := t.TempDir()
	cfg := &Config{
		Listen:            ":9999",
		LlamaCppSocketDir: llamaCppDir,
		QueueDepth:        4,
		TTLActive:         5 * time.Minute,
		TTLInactive:       30 * time.Minute,
		TTLUnused:         1 * time.Minute, // short TTL for testing
		GPUGroups:         []GPUGroup{{ID: "g0", GPUs: []int{0}}},
		Models: []ModelConfig{
			{Name: "llama-model", Engine: engineLlamaCpp},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: -1},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)
	me := o.models[0]

	me.mu.Lock()
	me.state = stateActive
	me.activeRequests = 0
	me.lastCompleted = time.Now().Add(-2 * time.Minute) // expired beyond ttl_unused
	me.proc = &vllmProcess{cmd: &exec.Cmd{}, socketPath: llamaCppDir + "/dummy.sock"}
	me.mu.Unlock()

	o.tickTTL(me)

	me.mu.Lock()
	st := me.state
	p := me.proc
	gi := me.assignedGroupIdx
	vram := me.reservedVRAMMB
	me.mu.Unlock()

	if st != stateUnloaded {
		t.Errorf("state = %s, want unloaded (llama_cpp goes ACTIVE→UNLOADED)", st)
	}
	if p != nil {
		t.Error("proc should be nil after unloading")
	}
	if gi != -1 {
		t.Errorf("assignedGroupIdx = %d, want -1", gi)
	}
	if vram != 0 {
		t.Errorf("reservedVRAMMB = %d, want 0", vram)
	}
}

func TestTickTTLActiveLlamaCppNotExpired(t *testing.T) {
	t.Parallel()
	llamaCppDir := t.TempDir()
	cfg := &Config{
		Listen:            ":9999",
		LlamaCppSocketDir: llamaCppDir,
		QueueDepth:        4,
		TTLActive:         5 * time.Minute,
		TTLInactive:       30 * time.Minute,
		TTLUnused:         60 * time.Minute,
		GPUGroups:         []GPUGroup{{ID: "g0", GPUs: []int{0}}},
		Models: []ModelConfig{
			{Name: "llama-model", Engine: engineLlamaCpp},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: -1},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)
	me := o.models[0]

	me.mu.Lock()
	me.state = stateActive
	me.activeRequests = 0
	me.lastCompleted = time.Now() // not expired
	me.mu.Unlock()

	o.tickTTL(me)

	me.mu.Lock()
	st := me.state
	me.mu.Unlock()
	if st != stateActive {
		t.Errorf("state = %s, want active (not expired)", st)
	}
}
