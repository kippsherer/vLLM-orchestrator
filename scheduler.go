package main

import (
	"fmt"
	"log"
	"sort"
)

// assignGroup implements §5: find the smallest-fitting GPU group for me,
// running freeing memory rules (§6) if needed. Reserves VRAM on success.
// Returns the index into ms.groups, or error if no group can be freed sufficiently.
func (o *orchestrator) assignGroup(me *modelEntry) (int, error) {
	o.ms.mu.Lock()
	defer o.ms.mu.Unlock()

	needed := placeholderVRAM(o.ms.groups[0], &me.mem) // initial estimate; groups[0] fine for scale
	if me.mem.measured {
		needed = me.mem.fullKVVRAMMB
	}

	idx, err := o.pickGroup(needed)
	if err != nil {
		// Run freeing memory rules and retry.
		if freeErr := o.freeMemoryRules(needed); freeErr != nil {
			return -1, fmt.Errorf("assign group: %w", freeErr)
		}
		idx, err = o.pickGroup(needed)
		if err != nil {
			return -1, fmt.Errorf("assign group: still insufficient after freeing: %w", err)
		}
	}

	// Reserve VRAM.
	o.ms.groups[idx].usedVRAMMB += needed
	return idx, nil
}

// pickGroup finds the qualifying group with the smallest measuredTotalVRAMMB.
// Must be called with ms.mu held.
func (o *orchestrator) pickGroup(neededMB int64) (int, error) {
	best := -1
	var bestTotal int64
	for i, gs := range o.ms.groups {
		if gs.freeVRAMMB() >= neededMB {
			if best < 0 || gs.measuredTotalVRAMMB < bestTotal {
				best = i
				bestTotal = gs.measuredTotalVRAMMB
			}
		}
	}
	if best < 0 {
		return -1, fmt.Errorf("no GPU group has %d MB free VRAM", neededMB)
	}
	return best, nil
}

// freeMemoryRules implements §6 cycle: Rule 1 → 2 → 3 → 4.
// Must be called with ms.mu held. Releases the lock while making HTTP calls,
// then re-acquires it.
func (o *orchestrator) freeMemoryRules(neededMB int64) error {
	// Rule 1: Move ACTIVE → SLEEP1 (smallest first; fallback to Rule 2 per model if CPU RAM low).
	o.rule1(neededMB)

	// Rule 2: Terminate SLEEP2 processes.
	o.rule2(neededMB)

	// Rule 3: Move SLEEP1 → SLEEP2 (frees CPU RAM so Rule 1 can retry).
	o.rule3()

	// Rule 4: Terminate SLEEP2 again.
	o.rule4(neededMB)

	// Check if any group now has enough.
	for _, gs := range o.ms.groups {
		if gs.freeVRAMMB() >= neededMB {
			return nil
		}
	}
	return fmt.Errorf("freeing memory rules exhausted; insufficient VRAM for %d MB", neededMB)
}

// idleActiveModels returns all ACTIVE modelEntry with no in-flight requests,
// sorted smallest fullKVVRAMMB first.
func (o *orchestrator) idleActiveModels() []*modelEntry {
	var out []*modelEntry
	for _, me := range o.models {
		me.mu.Lock()
		s := me.state
		active := me.activeRequests
		me.mu.Unlock()
		if s == stateActive && active == 0 {
			out = append(out, me)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].mem.fullKVVRAMMB < out[j].mem.fullKVVRAMMB
	})
	return out
}

// idleSleep2Models returns all SLEEP2 modelEntry with no in-flight requests,
// sorted smallest fullKVVRAMMB first.
func (o *orchestrator) idleSleep2Models() []*modelEntry {
	var out []*modelEntry
	for _, me := range o.models {
		me.mu.Lock()
		s := me.state
		active := me.activeRequests
		me.mu.Unlock()
		if s == stateSleep2 && active == 0 {
			out = append(out, me)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].mem.fullKVVRAMMB < out[j].mem.fullKVVRAMMB
	})
	return out
}

// idleSleep1Models returns all SLEEP1 modelEntry with no in-flight requests,
// sorted smallest weightsVRAMMB first.
func (o *orchestrator) idleSleep1Models() []*modelEntry {
	var out []*modelEntry
	for _, me := range o.models {
		me.mu.Lock()
		s := me.state
		active := me.activeRequests
		me.mu.Unlock()
		if s == stateSleep1 && active == 0 {
			out = append(out, me)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].mem.weightsVRAMMB < out[j].mem.weightsVRAMMB
	})
	return out
}

// rule1 moves idle ACTIVE models to SLEEP1 (smallest first).
// If CPU RAM insufficient for a model, calls rule2 inline to free SLEEP2 processes.
// Must be called with ms.mu held; releases lock around HTTP calls.
func (o *orchestrator) rule1(neededMB int64) {
	for _, gs := range o.ms.groups {
		if gs.freeVRAMMB() >= neededMB {
			continue
		}
		for _, me := range o.idleActiveModels() {
			if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
				continue
			}
			if o.ms.freeCPURAMB < me.mem.weightsVRAMMB {
				// CPU RAM insufficient: run Rule 2 to free SLEEP2 procs first.
				o.rule2ForCPU(me.mem.weightsVRAMMB)
			}
			if o.ms.freeCPURAMB < me.mem.weightsVRAMMB {
				continue // still not enough; skip this model
			}
			proc := func() *vllmProcess {
				me.mu.Lock()
				defer me.mu.Unlock()
				return me.proc
			}()
			o.ms.mu.Unlock()
			err := sleepModel(proc, me.cfg.Name, 1)
			o.ms.mu.Lock()
			if err != nil {
				log.Printf("[scheduler] rule1: sleep error %s: %v", me.cfg.Name, err)
				continue
			}
			me.mu.Lock()
			me.state = stateSleep1
			me.mu.Unlock()
			gs.usedVRAMMB -= me.mem.fullKVVRAMMB
			o.ms.freeCPURAMB -= me.mem.weightsVRAMMB
			log.Printf("[scheduler] rule1: %s ACTIVE → SLEEP1", me.cfg.Name)
			if gs.freeVRAMMB() >= neededMB {
				break
			}
		}
	}
}

// rule2 terminates idle SLEEP2 processes (smallest first) until neededMB freed per group.
// Must be called with ms.mu held; releases lock around process kill.
func (o *orchestrator) rule2(neededMB int64) {
	for _, gs := range o.ms.groups {
		if gs.freeVRAMMB() >= neededMB {
			continue
		}
		for _, me := range o.idleSleep2Models() {
			if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
				continue
			}
			proc := func() *vllmProcess {
				me.mu.Lock()
				defer me.mu.Unlock()
				return me.proc
			}()
			o.ms.mu.Unlock()
			killProcess(proc, me.cfg.Name)
			o.ms.mu.Lock()
			me.mu.Lock()
			me.state = stateUnloaded
			me.proc = nil
			me.assignedGroupIdx = -1
			me.mu.Unlock()
			// SLEEP2 holds no VRAM — no accounting change needed.
			log.Printf("[scheduler] rule2: %s SLEEP2 → UNLOADED", me.cfg.Name)
			if gs.freeVRAMMB() >= neededMB {
				break
			}
		}
	}
}

// rule2ForCPU terminates idle SLEEP2 processes until freeCPURAMB >= neededCPUMB.
// Must be called with ms.mu held.
func (o *orchestrator) rule2ForCPU(neededCPUMB int64) {
	for _, me := range o.idleSleep2Models() {
		if o.ms.freeCPURAMB >= neededCPUMB {
			break
		}
		proc := func() *vllmProcess {
			me.mu.Lock()
			defer me.mu.Unlock()
			return me.proc
		}()
		o.ms.mu.Unlock()
		killProcess(proc, me.cfg.Name)
		o.ms.mu.Lock()
		me.mu.Lock()
		me.state = stateUnloaded
		me.proc = nil
		me.assignedGroupIdx = -1
		me.mu.Unlock()
		log.Printf("[scheduler] rule2ForCPU: %s SLEEP2 → UNLOADED", me.cfg.Name)
	}
}

// rule3 moves idle SLEEP1 models to SLEEP2 to free CPU RAM.
// Must be called with ms.mu held; releases lock around HTTP calls.
func (o *orchestrator) rule3() {
	for _, me := range o.idleSleep1Models() {
		proc := func() *vllmProcess {
			me.mu.Lock()
			defer me.mu.Unlock()
			return me.proc
		}()
		o.ms.mu.Unlock()
		err := sleepModel(proc, me.cfg.Name, 2)
		o.ms.mu.Lock()
		if err != nil {
			log.Printf("[scheduler] rule3: sleep2 error %s: %v", me.cfg.Name, err)
			continue
		}
		me.mu.Lock()
		me.state = stateSleep2
		me.mu.Unlock()
		o.ms.freeCPURAMB += me.mem.weightsVRAMMB
		log.Printf("[scheduler] rule3: %s SLEEP1 → SLEEP2", me.cfg.Name)
	}
}

// rule4 is Rule 2 repeated after Rule 3 to terminate newly created SLEEP2 processes.
func (o *orchestrator) rule4(neededMB int64) {
	o.rule2(neededMB)
}
