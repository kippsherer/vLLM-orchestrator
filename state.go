package main

import (
	"log"
	"net/http"
	"sync"
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
	// done is closed by the controller when the request may be forwarded.
	done chan struct{}
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
	assignedGroupIdx int // index into memoryState.groups; -1 if unassigned
}

// orchestrator is the top-level runtime that owns all model entries and memory.
type orchestrator struct {
	cfg    *Config
	ms     *memoryState
	models []*modelEntry
	// modelByName maps canonical name and aliases → *modelEntry
	modelByName map[string]*modelEntry
}

// newOrchestrator constructs the orchestrator from config and memory state.
func newOrchestrator(cfg *Config, ms *memoryState) *orchestrator {
	o := &orchestrator{
		cfg:         cfg,
		ms:          ms,
		modelByName: make(map[string]*modelEntry),
	}
	for _, mc := range cfg.Models {
		me := &modelEntry{
			cfg:              mc,
			socketPath:       cfg.VLLMSocketDir + "/" + sanitizeModelName(mc.Name) + ".sock",
			state:            stateUnloaded,
			queue:            make(chan requestPair, cfg.QueueDepth),
			lastCompleted:    time.Now(),
			assignedGroupIdx: -1,
		}
		o.models = append(o.models, me)
		o.modelByName[mc.Name] = me
		for _, a := range mc.Aliases {
			o.modelByName[a] = me
		}
	}
	return o
}

// resolve returns the modelEntry for name-or-alias, or nil.
func (o *orchestrator) resolve(name string) *modelEntry {
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

	switch state {
	case stateActive:
		if activeReqs > 0 || idle < o.cfg.TTLActive {
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
		if activeReqs > 0 || idle < o.cfg.TTLInactive {
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
		if activeReqs > 0 || idle < o.cfg.TTLUnused {
			return
		}
		me.mu.Lock()
		if me.state != stateSleep2 || me.activeRequests > 0 {
			me.mu.Unlock()
			return
		}
		proc := me.proc
		groupIdx := me.assignedGroupIdx
		me.state = stateUnloaded
		me.proc = nil
		me.assignedGroupIdx = -1
		me.mu.Unlock()

		if proc != nil {
			killProcess(proc, me.cfg.Name)
		}
		if groupIdx >= 0 {
			o.ms.mu.Lock()
			// SLEEP2 holds no VRAM (already released), nothing to restore here.
			o.ms.mu.Unlock()
		}
		log.Printf("[orchestrator] %s: SLEEP2 → UNLOADED (ttl_unused elapsed)", me.cfg.Name)
	}
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
	groupIdx := me.assignedGroupIdx

	o.ms.mu.Lock()
	switch {
	case prevState == stateActive && level == 1:
		// Release VRAM; charge CPU RAM.
		if groupIdx >= 0 {
			o.ms.groups[groupIdx].usedVRAMMB -= me.mem.fullKVVRAMMB
		}
		o.ms.freeCPURAMB -= me.mem.weightsVRAMMB
		me.state = stateSleep1
	case prevState == stateActive && level == 2:
		// Release VRAM; no CPU RAM charged.
		if groupIdx >= 0 {
			o.ms.groups[groupIdx].usedVRAMMB -= me.mem.fullKVVRAMMB
		}
		me.state = stateSleep2
	case prevState == stateSleep1 && level == 2:
		// Release CPU RAM.
		o.ms.freeCPURAMB += me.mem.weightsVRAMMB
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
	groupIdx := me.assignedGroupIdx
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

	o.ms.mu.Lock()
	if prevState == stateSleep1 {
		// Restore VRAM; release CPU RAM.
		if groupIdx >= 0 {
			o.ms.groups[groupIdx].usedVRAMMB += me.mem.fullKVVRAMMB
		}
		o.ms.freeCPURAMB += me.mem.weightsVRAMMB
	} else {
		// SLEEP2: restore VRAM only.
		if groupIdx >= 0 {
			o.ms.groups[groupIdx].usedVRAMMB += me.mem.fullKVVRAMMB
		}
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

		proc, err := launchVLLM(me.cfg, me.socketPath, o.ms.groups[groupIdx], &me.mem)
		if err != nil {
			log.Printf("[orchestrator] %s: launch failed: %v", me.cfg.Name, err)
			me.mu.Lock()
			me.state = stateUnloaded
			me.assignedGroupIdx = -1
			me.mu.Unlock()
			o.ms.mu.Lock()
			o.ms.groups[groupIdx].usedVRAMMB -= placeholderVRAM(o.ms.groups[groupIdx], &me.mem)
			o.ms.mu.Unlock()
			o.drainQueueWith503(me)
			return
		}

		me.mu.Lock()
		me.proc = proc
		me.assignedGroupIdx = groupIdx
		me.mu.Unlock()

		if err := waitForHealth(proc, me.cfg.Name); err != nil {
			log.Printf("[orchestrator] %s: health poll failed: %v", me.cfg.Name, err)
			me.mu.Lock()
			me.state = stateUnloaded
			me.proc = nil
			me.assignedGroupIdx = -1
			me.mu.Unlock()
			o.ms.mu.Lock()
			o.ms.groups[groupIdx].usedVRAMMB -= placeholderVRAM(o.ms.groups[groupIdx], &me.mem)
			o.ms.mu.Unlock()
			o.drainQueueWith503(me)
			return
		}

		// Refine VRAM reservation from placeholder to actual if now measured.
		o.refineMeasuredVRAM(me, groupIdx)

		me.mu.Lock()
		me.state = stateActive
		me.lastCompleted = time.Now()
		me.mu.Unlock()
		log.Printf("[orchestrator] %s: → ACTIVE", me.cfg.Name)
		o.drainQueue(me)
	}
}

// refineMeasuredVRAM replaces the placeholder VRAM reservation with the actual
// measured fullKVVRAMMB if it is now available.
func (o *orchestrator) refineMeasuredVRAM(me *modelEntry, groupIdx int) {
	me.mu.Lock()
	measured := me.mem.measured
	actual := me.mem.fullKVVRAMMB
	me.mu.Unlock()

	if !measured {
		return
	}
	o.ms.mu.Lock()
	gs := o.ms.groups[groupIdx]
	placeholder := placeholderVRAM(gs, &me.mem)
	gs.usedVRAMMB = gs.usedVRAMMB - placeholder + actual
	o.ms.mu.Unlock()
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

// drainQueueWith503 writes 503 to all waiting requestPairs in me.queue.
func (o *orchestrator) drainQueueWith503(me *modelEntry) {
	for {
		select {
		case rp := <-me.queue:
			http.Error(rp.w, "service unavailable", http.StatusServiceUnavailable)
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

// completeRequest decrements activeRequests and resets the TTL clock.
func (o *orchestrator) completeRequest(me *modelEntry) {
	me.mu.Lock()
	me.activeRequests--
	me.lastCompleted = time.Now()
	me.mu.Unlock()
}
