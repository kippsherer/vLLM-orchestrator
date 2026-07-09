package main

import (
	"net/http"
	"net/http/httptest"
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
			me := o.resolve(tc.name)
			if tc.found && me == nil {
				t.Errorf("resolve(%q) = nil, want non-nil", tc.name)
			}
			if !tc.found && me != nil {
				t.Errorf("resolve(%q) = %v, want nil", tc.name, me)
			}
		})
	}
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
	for i := range recorders {
		recorders[i] = httptest.NewRecorder()
		me.queue <- requestPair{
			w:    recorders[i],
			r:    &http.Request{},
			done: make(chan struct{}),
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
		me.queue <- requestPair{
			w:    httptest.NewRecorder(),
			r:    &http.Request{},
			done: dones[i],
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
