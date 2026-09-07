package main

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"sync"
	"testing"
	"time"
)

func TestPickGroup(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		groups   []*groupState
		neededMB int64
		wantIdx  int
		wantErr  bool
	}{
		{
			name: "no group qualifies",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 10000, measuredFreeMB: 1000},
			},
			neededMB: 2000,
			wantIdx:  -1,
			wantErr:  true,
		},
		{
			name: "single qualifying group",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
			},
			neededMB: 8000,
			wantIdx:  0,
			wantErr:  false,
		},
		{
			name: "picks smallest total among qualifying",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 40960, measuredFreeMB: 40960},
				{id: "g1", measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
				{id: "g2", measuredTotalVRAMMB: 80000, measuredFreeMB: 80000},
			},
			neededMB: 8000,
			wantIdx:  1, // g1 has smallest total
			wantErr:  false,
		},
		{
			name: "exact fit",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 8000, measuredFreeMB: 8000},
			},
			neededMB: 8000,
			wantIdx:  0,
			wantErr:  false,
		},
		{
			name: "only larger group qualifies",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 10000, measuredFreeMB: 1000},
				{id: "g1", measuredTotalVRAMMB: 40960, measuredFreeMB: 40960},
			},
			neededMB: 8000,
			wantIdx:  1,
			wantErr:  false,
		},
		{
			name: "measured free too low",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 24576, measuredFreeMB: 3000},
			},
			neededMB: 8000,
			wantIdx:  -1,
			wantErr:  true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ms := &memoryState{groups: tc.groups}
			o := &orchestrator{ms: ms}

			// pickGroup requires ms.mu held; lock it for the call.
			ms.mu.Lock()
			idx, err := o.pickGroup(tc.neededMB, nil)
			ms.mu.Unlock()

			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if idx != tc.wantIdx {
				t.Errorf("pickGroup returned idx %d, want %d", idx, tc.wantIdx)
			}
		})
	}
}

func TestPickGroupExclude(t *testing.T) {
	t.Parallel()

	groups := []*groupState{
		{id: "g0", measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
		{id: "g1", measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
		{id: "g2", measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
	}
	ms := &memoryState{groups: groups}
	o := &orchestrator{ms: ms}

	ms.mu.Lock()
	idx, err := o.pickGroup(8000, map[int]bool{0: true})
	ms.mu.Unlock()
	if err != nil {
		t.Fatalf("pickGroup: %v", err)
	}
	if idx != 1 {
		t.Errorf("pickGroup with group 0 excluded returned %d, want 1", idx)
	}
}

func TestSiblingGroups(t *testing.T) {
	t.Parallel()

	rs := &replicaSet{}
	entries := make([]*modelEntry, 3)
	for i := range entries {
		entries[i] = &modelEntry{replicaSet: rs, assignedGroupIdx: -1}
		rs.entries = append(rs.entries, entries[i])
	}
	// Siblings 0 and 2 occupy groups 0 and 2 respectively.
	entries[0].assignedGroupIdx = 0
	entries[2].assignedGroupIdx = 2

	o := &orchestrator{}
	exclude := o.siblingGroups(entries[1])
	if len(exclude) != 2 || !exclude[0] || !exclude[2] {
		t.Errorf("siblingGroups = %v, want {0:true, 2:true}", exclude)
	}

	// Non-replicated entry returns nil.
	o2 := &orchestrator{}
	if got := o2.siblingGroups(&modelEntry{}); got != nil {
		t.Errorf("siblingGroups for non-replicated entry = %v, want nil", got)
	}
}

func TestAssignGroupReplicasDistinctGroups(t *testing.T) {
	// Not t.Parallel: mocks the global nvidia-smi/meminfo vars, which parallel
	// tests in this package also mutate.
	dir := t.TempDir()
	cfg := &Config{
		Listen:        ":9999",
		VLLMSocketDir: dir,
		QueueDepth:    4,
		TTLActive:     5 * time.Minute,
		TTLInactive:   30 * time.Minute,
		TTLUnused:     60 * time.Minute,
		GPUGroups: []GPUGroup{
			{ID: "g0", GPUs: []int{0}},
			{ID: "g1", GPUs: []int{1}},
			{ID: "g2", GPUs: []int{2}},
		},
		Models: []ModelConfig{
			{Name: "surya", Replicas: 3, VRAMAllocationMB: 22000},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
			{id: "g1", gpus: []int{1}, measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
			{id: "g2", gpus: []int{2}, measuredTotalVRAMMB: 24576, measuredFreeMB: 24576},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)

	origSmi := queryNvidiaSmiFreeMB
	origMem := readMemAvailableMB
	t.Cleanup(func() {
		queryNvidiaSmiFreeMB = origSmi
		readMemAvailableMB = origMem
	})
	queryNvidiaSmiFreeMB = func() (string, error) {
		return "0, 24576\n1, 24576\n2, 24576", nil
	}
	readMemAvailableMB = func() (int64, error) { return 65536, nil }

	assigned := map[int]bool{}
	for _, me := range o.models {
		idx, err := o.assignGroup(me)
		if err != nil {
			t.Fatalf("assignGroup: %v", err)
		}
		if assigned[idx] {
			t.Errorf("replica assigned to already-used group %d", idx)
		}
		assigned[idx] = true
	}
	if len(assigned) != 3 {
		t.Errorf("expected 3 distinct groups, got %d", len(assigned))
	}
}

func TestPlaceholderVRAM(t *testing.T) {
	t.Parallel()

	gs := &groupState{measuredTotalVRAMMB: 24576, measuredFreeMB: -1}

	t.Run("measured", func(t *testing.T) {
		t.Parallel()
		mem := &modelMemory{measured: true, fullKVVRAMMB: 12288}
		got := placeholderVRAM(gs, mem)
		if got != 12288 {
			t.Errorf("placeholderVRAM = %d, want 12288", got)
		}
	})

	t.Run("not_measured", func(t *testing.T) {
		t.Parallel()
		mem := &modelMemory{measured: false}
		got := placeholderVRAM(gs, mem)
		want := int64(float64(gs.measuredTotalVRAMMB) * 0.85)
		if got != want {
			t.Errorf("placeholderVRAM = %d, want %d", got, want)
		}
	})
}

func TestIdleModels(t *testing.T) {
	t.Parallel()

	lessByWeights := func(a, b *modelEntry) bool { return a.mem.weightsVRAMMB < b.mem.weightsVRAMMB }

	cases := []struct {
		name      string
		setup     func(o *orchestrator)
		state     modelState
		less      func(a, b *modelEntry) bool
		wantNames []string
	}{
		{
			name: "returns only models in requested state with zero active requests",
			setup: func(o *orchestrator) {
				o.models[0].mu.Lock()
				o.models[0].state = stateActive
				o.models[0].activeRequests = 0
				o.models[0].mem.weightsVRAMMB = 200
				o.models[0].mu.Unlock()
				o.models[1].mu.Lock()
				o.models[1].state = stateActive
				o.models[1].activeRequests = 0
				o.models[1].mem.weightsVRAMMB = 100
				o.models[1].mu.Unlock()
			},
			state:     stateActive,
			less:      lessByWeights,
			wantNames: []string{"model-b", "model-a"},
		},
		{
			name: "skips models with active requests",
			setup: func(o *orchestrator) {
				o.models[0].mu.Lock()
				o.models[0].state = stateActive
				o.models[0].activeRequests = 1
				o.models[0].mem.weightsVRAMMB = 100
				o.models[0].mu.Unlock()
				o.models[1].mu.Lock()
				o.models[1].state = stateActive
				o.models[1].activeRequests = 0
				o.models[1].mem.weightsVRAMMB = 200
				o.models[1].mu.Unlock()
			},
			state:     stateActive,
			less:      lessByWeights,
			wantNames: []string{"model-b"},
		},
		{
			name: "skips models in a different state",
			setup: func(o *orchestrator) {
				o.models[0].mu.Lock()
				o.models[0].state = stateActive
				o.models[0].activeRequests = 0
				o.models[0].mem.weightsVRAMMB = 100
				o.models[0].mu.Unlock()
				o.models[1].mu.Lock()
				o.models[1].state = stateSleep1
				o.models[1].activeRequests = 0
				o.models[1].mem.weightsVRAMMB = 200
				o.models[1].mu.Unlock()
			},
			state:     stateActive,
			less:      lessByWeights,
			wantNames: []string{"model-a"},
		},
		{
			name: "sorts by less comparator smallest first",
			setup: func(o *orchestrator) {
				o.models[0].mu.Lock()
				o.models[0].state = stateSleep1
				o.models[0].activeRequests = 0
				o.models[0].mem.weightsVRAMMB = 300
				o.models[0].mu.Unlock()
				o.models[1].mu.Lock()
				o.models[1].state = stateSleep1
				o.models[1].activeRequests = 0
				o.models[1].mem.weightsVRAMMB = 100
				o.models[1].mu.Unlock()
			},
			state:     stateSleep1,
			less:      lessByWeights,
			wantNames: []string{"model-b", "model-a"},
		},
		{
			name: "returns empty slice when nothing matches",
			setup: func(o *orchestrator) {
				o.models[0].mu.Lock()
				o.models[0].state = stateActive
				o.models[0].activeRequests = 1
				o.models[0].mu.Unlock()
				o.models[1].mu.Lock()
				o.models[1].state = stateSleep2
				o.models[1].activeRequests = 0
				o.models[1].mu.Unlock()
			},
			state:     stateSleep1,
			less:      lessByWeights,
			wantNames: nil,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			o := makeTestOrchestrator(t)
			tc.setup(o)

			got := o.idleModels(tc.state, tc.less)

			if len(got) != len(tc.wantNames) {
				t.Fatalf("idleModels returned %d models, want %d", len(got), len(tc.wantNames))
			}
			for i, me := range got {
				if me.cfg.Name != tc.wantNames[i] {
					t.Errorf("idleModels[%d] = %q, want %q", i, me.cfg.Name, tc.wantNames[i])
				}
			}
		})
	}
}

func TestModelStateString(t *testing.T) {
	t.Parallel()

	cases := []struct {
		state modelState
		want  string
	}{
		{stateUnloaded, "unloaded"},
		{stateLoading, "loading"},
		{stateActive, "active"},
		{stateSleep1, "sleep1"},
		{stateSleep2, "sleep2"},
		{modelState(99), "unknown"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.want, func(t *testing.T) {
			t.Parallel()
			got := tc.state.String()
			if got != tc.want {
				t.Errorf("modelState(%d).String() = %q, want %q", tc.state, got, tc.want)
			}
		})
	}
}

func TestWaitForActive(t *testing.T) {
	t.Parallel()

	t.Run("loading transitions to active drains queue", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		me.mu.Lock()
		me.state = stateLoading
		me.mu.Unlock()

		rec := httptest.NewRecorder()
		errFlag := false
		rp := requestPair{
			w:    rec,
			r:    &http.Request{},
			done: make(chan struct{}),
			err:  &errFlag,
		}
		me.queue <- rp

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			o.waitForActive(me)
		}()

		time.Sleep(50 * time.Millisecond)
		me.mu.Lock()
		me.state = stateActive
		me.mu.Unlock()

		wg.Wait()

		select {
		case <-rp.done:
		default:
			t.Fatal("done channel was not closed")
		}
		if errFlag {
			t.Error("errFlag set, expected drainQueue (not 503)")
		}
		me.mu.Lock()
		active := me.activeRequests
		me.mu.Unlock()
		if active != 1 {
			t.Errorf("activeRequests = %d, want 1", active)
		}
	})

	t.Run("loading transitions to unloaded drains with 503", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		me.mu.Lock()
		me.state = stateLoading
		me.mu.Unlock()

		rec := httptest.NewRecorder()
		errFlag := false
		rp := requestPair{
			w:    rec,
			r:    &http.Request{},
			done: make(chan struct{}),
			err:  &errFlag,
		}
		me.queue <- rp

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			o.waitForActive(me)
		}()

		time.Sleep(50 * time.Millisecond)
		me.mu.Lock()
		me.state = stateUnloaded
		me.mu.Unlock()

		wg.Wait()

		select {
		case <-rp.done:
		default:
			t.Fatal("done channel was not closed")
		}
		if !errFlag {
			t.Error("errFlag not set, expected drainQueueWith503")
		}
		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("response status = %d, want 503", rec.Code)
		}
	})
}

// startMockVLLMServer starts a Unix socket HTTP server at sockPath with the
// given handler. Returns a cleanup function.
func startMockVLLMServer(t *testing.T, sockPath string, handler http.Handler) {
	t.Helper()
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("listen unix %s: %v", sockPath, err)
	}
	srv := &http.Server{Handler: handler}
	go srv.Serve(ln)
	t.Cleanup(func() {
		srv.Close()
		ln.Close()
	})
}

func TestTransitionToSleep(t *testing.T) {
	t.Parallel()

	t.Run("proc nil returns immediately", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]
		me.mu.Lock()
		me.state = stateActive
		me.mem.weightsVRAMMB = 1024
		me.mu.Unlock()

		cpuBefore := o.ms.freeCPURAMB
		o.transitionToSleep(me, nil, 1)

		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateActive {
			t.Errorf("state = %s, want active (unchanged)", st)
		}
		if o.ms.freeCPURAMB != cpuBefore {
			t.Errorf("freeCPURAMB changed from %d to %d, want unchanged", cpuBefore, o.ms.freeCPURAMB)
		}
	})

	t.Run("active to sleep1", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		sockPath := t.TempDir() + "/test.sock"
		mux := http.NewServeMux()
		mux.HandleFunc("/sleep", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"is_sleeping":true}`)
		})
		startMockVLLMServer(t, sockPath, mux)

		vp := makeTestVLLMProcess(sockPath)

		me.mu.Lock()
		me.state = stateActive
		me.proc = vp
		me.mem.weightsVRAMMB = 1024
		me.mu.Unlock()

		cpuBefore := o.ms.freeCPURAMB
		o.transitionToSleep(me, vp, 1)

		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateSleep1 {
			t.Errorf("state = %s, want sleep1", st)
		}
		if o.ms.freeCPURAMB != cpuBefore-1024 {
			t.Errorf("freeCPURAMB = %d, want %d (decremented by weightsVRAM)", o.ms.freeCPURAMB, cpuBefore-1024)
		}
	})

	t.Run("active to sleep2", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		sockPath := t.TempDir() + "/test.sock"
		mux := http.NewServeMux()
		mux.HandleFunc("/sleep", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"is_sleeping":true}`)
		})
		startMockVLLMServer(t, sockPath, mux)

		vp := makeTestVLLMProcess(sockPath)

		me.mu.Lock()
		me.state = stateActive
		me.proc = vp
		me.mem.weightsVRAMMB = 1024
		me.mu.Unlock()

		cpuBefore := o.ms.freeCPURAMB
		o.transitionToSleep(me, vp, 2)

		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateSleep2 {
			t.Errorf("state = %s, want sleep2", st)
		}
		if o.ms.freeCPURAMB != cpuBefore {
			t.Errorf("freeCPURAMB = %d, want %d (unchanged)", o.ms.freeCPURAMB, cpuBefore)
		}
	})

	t.Run("sleep1 to sleep2", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		sockPath := t.TempDir() + "/test.sock"
		mux := http.NewServeMux()
		mux.HandleFunc("/sleep", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"is_sleeping":true}`)
		})
		startMockVLLMServer(t, sockPath, mux)

		vp := makeTestVLLMProcess(sockPath)

		me.mu.Lock()
		me.state = stateSleep1
		me.proc = vp
		me.mem.weightsVRAMMB = 1024
		me.mu.Unlock()

		cpuBefore := o.ms.freeCPURAMB
		o.transitionToSleep(me, vp, 2)

		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateSleep2 {
			t.Errorf("state = %s, want sleep2", st)
		}
		if o.ms.freeCPURAMB != cpuBefore+1024 {
			t.Errorf("freeCPURAMB = %d, want %d (incremented by weightsVRAM)", o.ms.freeCPURAMB, cpuBefore+1024)
		}
	})
}

func TestWakeAndActivate(t *testing.T) {
	t.Parallel()

	t.Run("proc nil returns nil", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]
		me.mu.Lock()
		me.state = stateSleep1
		me.mu.Unlock()

		err := o.wakeAndActivate(me)
		if err != nil {
			t.Errorf("wakeAndActivate with nil proc returned error: %v", err)
		}
		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateSleep1 {
			t.Errorf("state = %s, want sleep1 (unchanged)", st)
		}
	})

	t.Run("sleep1 to active", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		sockPath := t.TempDir() + "/test.sock"
		mux := http.NewServeMux()
		mux.HandleFunc("/wake_up", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"is_sleeping":false}`)
		})
		startMockVLLMServer(t, sockPath, mux)

		vp := makeTestVLLMProcess(sockPath)

		me.mu.Lock()
		me.state = stateSleep1
		me.proc = vp
		me.mem.weightsVRAMMB = 1024
		me.mu.Unlock()

		cpuBefore := o.ms.freeCPURAMB
		err := o.wakeAndActivate(me)
		if err != nil {
			t.Fatalf("wakeAndActivate: %v", err)
		}

		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateActive {
			t.Errorf("state = %s, want active", st)
		}
		if o.ms.freeCPURAMB != cpuBefore+1024 {
			t.Errorf("freeCPURAMB = %d, want %d (incremented by weightsVRAM)", o.ms.freeCPURAMB, cpuBefore+1024)
		}
	})

	t.Run("sleep2 to active", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		me := o.models[0]

		sockPath := t.TempDir() + "/test.sock"
		mux := http.NewServeMux()
		mux.HandleFunc("/wake_up", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"is_sleeping":false}`)
		})
		startMockVLLMServer(t, sockPath, mux)

		vp := makeTestVLLMProcess(sockPath)

		me.mu.Lock()
		me.state = stateSleep2
		me.proc = vp
		me.mem.weightsVRAMMB = 1024
		me.mu.Unlock()

		cpuBefore := o.ms.freeCPURAMB
		err := o.wakeAndActivate(me)
		if err != nil {
			t.Fatalf("wakeAndActivate: %v", err)
		}

		me.mu.Lock()
		st := me.state
		me.mu.Unlock()
		if st != stateActive {
			t.Errorf("state = %s, want active", st)
		}
		if o.ms.freeCPURAMB != cpuBefore {
			t.Errorf("freeCPURAMB = %d, want %d (unchanged)", o.ms.freeCPURAMB, cpuBefore)
		}
	})
}

func TestRule1LlamaCppEviction(t *testing.T) {
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
			{Name: "llama-dest-model", Engine: engineLlamaCpp},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: 10000},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)
	me := o.models[0]

	// Set model to ACTIVE with zero active requests and assigned to group 0.
	me.mu.Lock()
	me.state = stateActive
	me.activeRequests = 0
	me.assignedGroupIdx = 0
	me.reservedVRAMMB = 15000
	me.mem.fullKVVRAMMB = 15000
	me.proc = &vllmProcess{cmd: &exec.Cmd{}, socketPath: llamaCppDir + "/dummy.sock"}
	me.mu.Unlock()

	// Mock nvidia-smi to return freed VRAM (10000 + 15000 = 25000) after eviction.
	origSmi := queryNvidiaSmiFreeMB
	origMem := readMemAvailableMB
	t.Cleanup(func() {
		queryNvidiaSmiFreeMB = origSmi
		readMemAvailableMB = origMem
	})
	queryNvidiaSmiFreeMB = func() (string, error) { return "0, 25000", nil }
	readMemAvailableMB = func() (int64, error) { return 65536, nil }

	// Call freeMemoryRules — it should evict the llama_cpp model (activeReqs==0).
	result := o.freeMemoryRules(ms.groups[0], 12000)

	if !result {
		t.Error("expected freeMemoryRules to succeed after evicting llama_cpp model")
	}

	me.mu.Lock()
	st := me.state
	p := me.proc
	me.mu.Unlock()

	if st != stateUnloaded {
		t.Errorf("state = %s, want unloaded after Rule 1 llama_cpp eviction", st)
	}
	if p != nil {
		t.Error("proc should be nil after eviction via killAndUnload")
	}
}

func TestRule1LlamaCppSkippedWithActiveReqs(t *testing.T) {
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
			{Name: "llama-dest-model", Engine: engineLlamaCpp},
		},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: 5000},
		},
		freeCPURAMB: 65536,
	}
	o := newOrchestrator(cfg, ms)
	me := o.models[0]

	// Set model to ACTIVE with activeRequests > 0 — should NOT be evicted.
	me.mu.Lock()
	me.state = stateActive
	me.activeRequests = 3
	me.assignedGroupIdx = 0
	me.reservedVRAMMB = 15000
	me.mem.fullKVVRAMMB = 15000
	me.proc = &vllmProcess{cmd: &exec.Cmd{}, socketPath: llamaCppDir + "/dummy.sock"}
	me.mu.Unlock()

	// freeMemoryRules should skip this llama_cpp model because activeReqs > 0.
	result := o.freeMemoryRules(ms.groups[0], 12000)

	if result {
		t.Error("expected freeMemoryRules to fail: llama_cpp model with active requests should be skipped")
	}

	me.mu.Lock()
	st := me.state
	p := me.proc
	me.mu.Unlock()

	if st != stateActive {
		t.Errorf("state = %s, want active (should NOT be evicted with active requests)", st)
	}
	if p == nil {
		t.Error("proc should not be nil (model was not evicted)")
	}
}

// makePinnedOrchestrator builds an orchestrator with two models pinned to the
// same single-GPU group, for exercising the co-residency gate in assignGroup.
func makePinnedOrchestrator(t *testing.T, modelA ModelConfig, modelB ModelConfig) *orchestrator {
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
		Models:        []ModelConfig{modelA, modelB},
	}
	ms := &memoryState{
		groups: []*groupState{
			{id: "g0", gpus: []int{0}, measuredTotalVRAMMB: 24576, measuredFreeMB: -1},
		},
		freeCPURAMB: 65536,
	}
	return newOrchestrator(cfg, ms)
}

func TestAssignGroupPinnedCoResidency(t *testing.T) {
	t.Parallel()

	o := makePinnedOrchestrator(t,
		ModelConfig{Name: "model-a", GPUGroup: "g0", VRAMAllocationMB: 12000},
		ModelConfig{Name: "model-b", GPUGroup: "g0", VRAMAllocationMB: 9000},
	)

	// model-a is ACTIVE and already assigned to g0, occupying 12000 MB.
	a := o.models[0]
	a.mu.Lock()
	a.state = stateActive
	a.assignedGroupIdx = 0
	a.reservedVRAMMB = 12000
	a.mu.Unlock()

	// nvidia-smi reports 24576 - 12000 = 12576 MB free: enough for model-b's
	// 9000 MB reservation, so model-a must NOT be evicted.
	origSmi := queryNvidiaSmiFreeMB
	origMem := readMemAvailableMB
	t.Cleanup(func() {
		queryNvidiaSmiFreeMB = origSmi
		readMemAvailableMB = origMem
	})
	queryNvidiaSmiFreeMB = func() (string, error) { return "0, 12576", nil }
	readMemAvailableMB = func() (int64, error) { return 65536, nil }

	b := o.models[1]
	idx, err := o.assignGroup(b)
	if err != nil {
		t.Fatalf("assignGroup: %v", err)
	}
	if idx != 0 {
		t.Errorf("assignGroup returned idx %d, want 0", idx)
	}

	a.mu.Lock()
	st := a.state
	a.mu.Unlock()
	if st != stateActive {
		t.Errorf("model-a state = %s, want active (co-residency: must not be evicted)", st)
	}
}

func TestAssignGroupPinnedEvictsWhenInsufficientFree(t *testing.T) {
	t.Parallel()

	o := makePinnedOrchestrator(t,
		ModelConfig{Name: "model-a", Engine: engineLlamaCpp, GPUGroup: "g0", VRAMAllocationMB: 22000},
		ModelConfig{Name: "model-b", GPUGroup: "g0", VRAMAllocationMB: 9000},
	)

	// model-a (llama_cpp) is ACTIVE and assigned to g0, occupying 22000 MB.
	a := o.models[0]
	a.mu.Lock()
	a.state = stateActive
	a.assignedGroupIdx = 0
	a.reservedVRAMMB = 22000
	a.mem.fullKVVRAMMB = 22000
	a.proc = &vllmProcess{cmd: &exec.Cmd{}, socketPath: t.TempDir() + "/dummy.sock"}
	a.mu.Unlock()

	// First nvidia-smi read (the gate check) reports 24576 - 22000 = 2576 MB
	// free: insufficient for model-b's 9000 MB, so model-a must be evicted.
	// Subsequent reads (post-eviction) report the reclaimed 25000 MB.
	calls := 0
	origSmi := queryNvidiaSmiFreeMB
	origMem := readMemAvailableMB
	t.Cleanup(func() {
		queryNvidiaSmiFreeMB = origSmi
		readMemAvailableMB = origMem
	})
	queryNvidiaSmiFreeMB = func() (string, error) {
		calls++
		if calls == 1 {
			return "0, 2576", nil
		}
		return "0, 25000", nil
	}
	readMemAvailableMB = func() (int64, error) { return 65536, nil }

	b := o.models[1]
	idx, err := o.assignGroup(b)
	if err != nil {
		t.Fatalf("assignGroup: %v", err)
	}
	if idx != 0 {
		t.Errorf("assignGroup returned idx %d, want 0", idx)
	}

	a.mu.Lock()
	st := a.state
	p := a.proc
	a.mu.Unlock()
	if st != stateUnloaded {
		t.Errorf("model-a state = %s, want unloaded (insufficient free VRAM must evict)", st)
	}
	if p != nil {
		t.Error("model-a proc should be nil after eviction")
	}
}
