package main

import (
	"context"
	"log"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// modelState enumerates the states from §3.
type modelState int

const (
	stateUnloaded modelState = iota
	stateLoading
	stateActive
	stateSleep1
	stateSleep2
)

func (s modelState) String() string {
	switch s {
	case stateUnloaded:
		return "unloaded"
	case stateLoading:
		return "loading"
	case stateActive:
		return "active"
	case stateSleep1:
		return "sleep1"
	case stateSleep2:
		return "sleep2"
	default:
		return "unknown"
	}
}

// requestPair holds the writer and request for a queued HTTP request.
type requestPair struct {
	w http.ResponseWriter
	r *http.Request
	// done is closed by the controller when the request may be forwarded or rejected.
	// err is set before closing done when the request is rejected with a 503.
	done chan struct{}
	err  *bool // true means 503 was already written; caller must not forward
}

// modelEntry is the full runtime record for one model.
type modelEntry struct {
	cfg        ModelConfig
	socketPath string
	mem        modelMemory

	mu               sync.Mutex
	state            modelState
	proc             *vllmProcess
	activeRequests   int
	queue            chan requestPair
	lastCompleted    time.Time
	assignedGroupIdx int   // index into memoryState.groups; -1 if unassigned
	reservedVRAMMB   int64 // VRAM MB reserved at launch time (placeholder or actual)
	replicaSet       *replicaSet
}

// replicaSet groups the N modelEntry instances of a replicated model and owns
// the round-robin cursor plus a mutex serializing group assignment so siblings
// land on distinct GPU groups.
type replicaSet struct {
	entries  []*modelEntry
	next     atomic.Uint64
	assignMu sync.Mutex
}

// pick returns the next entry in round-robin order. Single-entry sets always
// return that entry without touching the counter.
func (rs *replicaSet) pick() *modelEntry {
	if len(rs.entries) == 1 {
		return rs.entries[0]
	}
	n := rs.next.Add(1) - 1
	return rs.entries[n%uint64(len(rs.entries))]
}

// orchestrator is the top-level runtime that owns all model entries and memory.
type orchestrator struct {
	cfg    *Config
	ms     *memoryState
	models []*modelEntry
	// modelByName maps canonical name and aliases → *replicaSet
	modelByName map[string]*replicaSet
}

// newOrchestrator constructs the orchestrator from config and memory state.
func newOrchestrator(cfg *Config, ms *memoryState) *orchestrator {
	o := &orchestrator{
		cfg:         cfg,
		ms:          ms,
		modelByName: make(map[string]*replicaSet),
	}
	for _, mc := range cfg.Models {
		socketDir := cfg.VLLMSocketDir
		if mc.Engine == engineLlamaCpp {
			socketDir = cfg.LlamaCppSocketDir
		}
		n := mc.Replicas
		if n <= 0 {
			n = 1
		}
		rs := &replicaSet{}
		for i := 0; i < n; i++ {
			sockName := sanitizeModelName(mc.Name)
			if n > 1 {
				sockName += "-" + strconv.Itoa(i)
			}
			me := &modelEntry{
				cfg:              mc,
				socketPath:       socketDir + "/" + sockName + ".sock",
				state:            stateUnloaded,
				queue:            make(chan requestPair, cfg.QueueDepth),
				lastCompleted:    time.Now(),
				assignedGroupIdx: -1,
				replicaSet:       rs,
			}
			o.models = append(o.models, me)
			rs.entries = append(rs.entries, me)
		}
		o.modelByName[mc.Name] = rs
		for _, a := range mc.Aliases {
			o.modelByName[a] = rs
		}
	}
	return o
}

// resolve returns the replicaSet for name-or-alias, or nil.
func (o *orchestrator) resolve(name string) *replicaSet {
	return o.modelByName[name]
}

// startTTLLoops launches one TTL goroutine per model.
func (o *orchestrator) startTTLLoops() {
	for _, me := range o.models {
		go o.ttlLoop(me)
	}
}

// ttlLoop is the per-model goroutine that fires TTL transitions and drains the queue.
func (o *orchestrator) ttlLoop(me *modelEntry) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		o.tickTTL(me)
	}
}

// tickTTL evaluates TTL conditions and fires transitions as appropriate.
// All transitions that touch shared state use me.mu + ms.mu in consistent order.
func (o *orchestrator) tickTTL(me *modelEntry) {
	me.mu.Lock()
	state := me.state
	activeReqs := me.activeRequests
	idle := time.Since(me.lastCompleted)
	me.mu.Unlock()

	// Resolve effective TTL: per-model override takes precedence over global.
	effTTL := func(modelVal, globalVal time.Duration) time.Duration {
		if modelVal > 0 {
			return modelVal
		}
		return globalVal
	}
	ttlActive := effTTL(me.cfg.TTLActive, o.cfg.TTLActive)
	ttlInactive := effTTL(me.cfg.TTLInactive, o.cfg.TTLInactive)
	ttlUnused := effTTL(me.cfg.TTLUnused, o.cfg.TTLUnused)

	switch state {
	case stateActive:
		if me.cfg.Engine == engineLlamaCpp {
			if activeReqs > 0 || idle < ttlUnused {
				return
			}
			me.mu.Lock()
			if me.state != stateActive || me.activeRequests > 0 {
				me.mu.Unlock()
				return
			}
			proc := me.proc
			me.mu.Unlock()
			o.killAndUnload(me, proc, "ttl_unused elapsed")
			return
		}
		if activeReqs > 0 || idle < ttlActive {
			return
		}
		// Transition ACTIVE → SLEEP1 or SLEEP2 depending on CPU RAM.
		o.ms.mu.RLock()
		cpuFree := o.ms.freeCPURAMB
		o.ms.mu.RUnlock()

		me.mu.Lock()
		// Re-check under lock.
		if me.state != stateActive || me.activeRequests > 0 {
			me.mu.Unlock()
			return
		}
		weightsVRAM := me.mem.weightsVRAMMB
		proc := me.proc
		me.mu.Unlock()

		if cpuFree >= weightsVRAM {
			o.transitionToSleep(me, proc, 1)
		} else {
			o.transitionToSleep(me, proc, 2)
		}

	case stateSleep1:
		if activeReqs > 0 || idle < ttlInactive {
			return
		}
		me.mu.Lock()
		if me.state != stateSleep1 || me.activeRequests > 0 {
			me.mu.Unlock()
			return
		}
		proc := me.proc
		me.mu.Unlock()
		o.transitionToSleep(me, proc, 2)

	case stateSleep2:
		if activeReqs > 0 || idle < ttlUnused {
			return
		}
		me.mu.Lock()
		if me.state != stateSleep2 || me.activeRequests > 0 {
			me.mu.Unlock()
			return
		}
		proc := me.proc
		me.mu.Unlock()
		o.killAndUnload(me, proc, "ttl_unused elapsed")
	}
}

// killAndUnload terminates proc and marks me UNLOADED. Must be called without me.mu held.
func (o *orchestrator) killAndUnload(me *modelEntry, proc *vllmProcess, reason string) {
	me.mu.Lock()
	if me.proc != proc {
		me.mu.Unlock()
		return
	}
	me.state = stateUnloaded
	me.proc = nil
	me.assignedGroupIdx = -1
	me.reservedVRAMMB = 0
	me.mu.Unlock()
	killProcess(proc, me.cfg.Name)
	log.Printf("[orchestrator] %s: → UNLOADED (%s)", me.cfg.Name, reason)
}

// transitionToSleep calls /sleep?level on the vLLM process, updates accounting,
// and marks the new state. Must be called without me.mu held.
func (o *orchestrator) transitionToSleep(me *modelEntry, proc *vllmProcess, level int) {
	if proc == nil {
		return
	}
	if err := sleepModel(proc, me.cfg.Name, level); err != nil {
		log.Printf("[orchestrator] %s: sleep level %d error: %v", me.cfg.Name, level, err)
		return
	}

	me.mu.Lock()
	defer me.mu.Unlock()

	// Guard: state may have changed while the HTTP call was in-flight.
	if me.proc != proc {
		return
	}

	prevState := me.state
	weightsVRAM := me.mem.weightsVRAMMB

	o.ms.mu.Lock()
	switch {
	case prevState == stateActive && level == 1:
		o.ms.freeCPURAMB -= weightsVRAM
		me.state = stateSleep1
	case prevState == stateActive && level == 2:
		me.state = stateSleep2
	case prevState == stateSleep1 && level == 2:
		o.ms.freeCPURAMB += weightsVRAM
		me.state = stateSleep2
	}
	o.ms.mu.Unlock()

	log.Printf("[orchestrator] %s: → %s", me.cfg.Name, me.state)
}

// wakeAndActivate wakes a SLEEP1 or SLEEP2 model and marks it ACTIVE.
// Must be called without me.mu held. Returns error if wake fails.
func (o *orchestrator) wakeAndActivate(me *modelEntry) error {
	me.mu.Lock()
	proc := me.proc
	prevState := me.state
	me.mu.Unlock()

	if proc == nil {
		return nil
	}
	if err := wakeModel(proc, me.cfg.Name); err != nil {
		return err
	}

	me.mu.Lock()
	defer me.mu.Unlock()

	if me.proc != proc {
		return nil
	}

	weightsVRAM := me.mem.weightsVRAMMB

	o.ms.mu.Lock()
	if prevState == stateSleep1 {
		o.ms.freeCPURAMB += weightsVRAM
	}
	o.ms.mu.Unlock()

	me.state = stateActive
	me.lastCompleted = time.Now()
	log.Printf("[orchestrator] %s: → ACTIVE (wake)", me.cfg.Name)
	return nil
}

// handleRequest is the entry point from the HTTP handler for a model-routed request.
// It either forwards immediately (ACTIVE) or enqueues, triggers load/wake, and waits.
func (o *orchestrator) handleRequest(me *modelEntry, rp requestPair) {
	me.mu.Lock()
	state := me.state
	me.mu.Unlock()

	if state == stateActive {
		close(rp.done)
		return
	}

	// Enqueue.
	select {
	case me.queue <- rp:
	default:
		http.Error(rp.w, "service unavailable: queue full", http.StatusServiceUnavailable)
		return
	}

	me.mu.Lock()
	state = me.state
	me.mu.Unlock()

	switch state {
	case stateActive:
		// Raced to ACTIVE between enqueue and here — drain queue.
		o.drainQueue(me)
	case stateSleep1, stateSleep2:
		if err := o.wakeAndActivate(me); err != nil {
			log.Printf("[orchestrator] %s: wake error: %v", me.cfg.Name, err)
			o.drainQueueWith503(me)
			return
		}
		o.drainQueue(me)
	case stateLoading:
		// Another goroutine is already loading; wait for it.
		o.waitForActive(me)
	case stateUnloaded:
		// Claim the LOADING state.
		me.mu.Lock()
		if me.state != stateUnloaded {
			s := me.state
			me.mu.Unlock()
			// Someone else started loading; handle accordingly.
			if s == stateActive {
				o.drainQueue(me)
			} else if s == stateLoading {
				o.waitForActive(me)
			}
			return
		}
		me.state = stateLoading
		me.mu.Unlock()

		groupIdx, err := o.assignGroup(me)
		if err != nil {
			log.Printf("[orchestrator] %s: assign group failed: %v", me.cfg.Name, err)
			me.mu.Lock()
			me.state = stateUnloaded
			me.mu.Unlock()
			o.drainQueueWith503(me)
			return
		}

		var proc *vllmProcess
		if me.cfg.Engine == engineLlamaCpp {
			proc, err = launchLlamaCpp(me.cfg, me.socketPath, o.ms.groups[groupIdx], &me.mem, o.cfg.LlamaCppModelDir)
		} else {
			proc, err = launchVLLM(me.cfg, me.socketPath, o.ms.groups[groupIdx], &me.mem)
		}
		if err != nil {
			log.Printf("[orchestrator] %s: launch failed: %v", me.cfg.Name, err)
			me.mu.Lock()
			me.state = stateUnloaded
			me.assignedGroupIdx = -1
			me.reservedVRAMMB = 0
			me.mu.Unlock()
			o.drainQueueWith503(me)
			return
		}
		log.Printf("[orchestrator] %s: process started (pid %d), waiting for health", me.cfg.Name, proc.cmd.Process.Pid)

		me.mu.Lock()
		me.proc = proc
		me.assignedGroupIdx = groupIdx
		me.mu.Unlock()

		// Register crash handler: if vLLM exits unexpectedly, release VRAM and mark UNLOADED.
		proc.onExit = func() {
			me.mu.Lock()
			if me.proc != proc {
				me.mu.Unlock()
				return
			}
			me.state = stateUnloaded
			me.proc = nil
			me.assignedGroupIdx = -1
			me.reservedVRAMMB = 0
			me.mu.Unlock()
			log.Printf("[orchestrator] %s: process exited unexpectedly → UNLOADED", me.cfg.Name)
			o.drainQueueWith503(me)
		}

		if err := waitForHealth(proc, me.cfg.Name); err != nil {
			log.Printf("[orchestrator] %s: health poll failed: %v", me.cfg.Name, err)
			me.mu.Lock()
			me.state = stateUnloaded
			me.proc = nil
			me.assignedGroupIdx = -1
			me.reservedVRAMMB = 0
			me.mu.Unlock()
			o.drainQueueWith503(me)
			return
		}
		log.Printf("[orchestrator] %s: ready", me.cfg.Name)

		me.mu.Lock()
		if me.mem.measured {
			me.reservedVRAMMB = me.mem.fullKVVRAMMB
		}
		// If not yet measured, reservedVRAMMB retains the placeholder set by assignGroup.
		me.state = stateActive
		me.lastCompleted = time.Now()
		me.mu.Unlock()
		log.Printf("[orchestrator] %s: → ACTIVE", me.cfg.Name)
		go o.watchHealth(me, proc)
		o.drainQueue(me)
	}
}

// watchHealth polls GET /health every 10s while proc is the active process for me.
// On 3 consecutive failures it invokes the crash handler (onExit).
func (o *orchestrator) watchHealth(me *modelEntry, proc *vllmProcess) {
	const interval = 10 * time.Second
	const maxFails = 3
	fails := 0
	for {
		time.Sleep(interval)
		me.mu.Lock()
		if me.proc != proc {
			me.mu.Unlock()
			return // a new process took over or model was unloaded deliberately
		}
		s := me.state
		me.mu.Unlock()

		if s == stateUnloaded || s == stateLoading {
			return
		}

		resp, err := proc.client.Get("http://vllm/health")
		if err == nil {
			resp.Body.Close()
			fails = 0
			continue
		}
		fails++
		log.Printf("[orchestrator] %s: health check failed (%d/%d): %v", me.cfg.Name, fails, maxFails, err)
		if fails >= maxFails {
			log.Printf("[orchestrator] %s: health check failed %d times → UNLOADED", me.cfg.Name, maxFails)
			if proc.onExit != nil {
				proc.onExit()
			}
			return
		}
	}
}

// placeholderVRAM returns group.measuredTotalVRAMMB * 0.85 as the placeholder
// reservation used before actual model memory is measured.
// Must be called with ms.mu held for gs, or before concurrent access begins.
func placeholderVRAM(gs *groupState, mem *modelMemory) int64 {
	if mem.measured {
		return mem.fullKVVRAMMB
	}
	return int64(float64(gs.measuredTotalVRAMMB) * 0.85)
}

// drainQueue unblocks all requestPairs waiting in me.queue.
func (o *orchestrator) drainQueue(me *modelEntry) {
	for {
		select {
		case rp := <-me.queue:
			me.mu.Lock()
			me.activeRequests++
			me.mu.Unlock()
			close(rp.done)
		default:
			return
		}
	}
}

// drainQueueWith503 writes 503 to all waiting requestPairs in me.queue
// and closes their done channels so callers unblock immediately.
func (o *orchestrator) drainQueueWith503(me *modelEntry) {
	for {
		select {
		case rp := <-me.queue:
			http.Error(rp.w, "service unavailable", http.StatusServiceUnavailable)
			*rp.err = true
			close(rp.done)
		default:
			return
		}
	}
}

// waitForActive blocks until me reaches stateActive, then drains the queue.
func (o *orchestrator) waitForActive(me *modelEntry) {
	for {
		time.Sleep(200 * time.Millisecond)
		me.mu.Lock()
		s := me.state
		me.mu.Unlock()
		if s == stateActive {
			o.drainQueue(me)
			return
		}
		if s == stateUnloaded {
			// Loading failed; drain with 503.
			o.drainQueueWith503(me)
			return
		}
	}
}

// loadModel triggers an internal load of me without an inbound HTTP request.
// Used for load_at_startup. Blocks until the model reaches ACTIVE or fails.
func (o *orchestrator) loadModel(me *modelEntry) {
	errFlag := false
	rp := requestPair{
		w:    discardResponseWriter{},
		r:    &http.Request{},
		done: make(chan struct{}),
		err:  &errFlag,
	}
	// Provide a non-nil context so routeRequest's r.Context().Done() doesn't panic.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rp.r = rp.r.WithContext(ctx)
	o.handleRequest(me, rp)
	<-rp.done
}

// discardResponseWriter is a minimal http.ResponseWriter for internal boot
// requests where no real client is waiting.
type discardResponseWriter struct{}

func (discardResponseWriter) Header() http.Header         { return http.Header{} }
func (discardResponseWriter) Write(b []byte) (int, error) { return len(b), nil }
func (discardResponseWriter) WriteHeader(int)             {}

// completeRequest decrements activeRequests and resets the TTL clock.
func (o *orchestrator) completeRequest(me *modelEntry) {
	me.mu.Lock()
	me.activeRequests--
	me.lastCompleted = time.Now()
	me.mu.Unlock()
}
