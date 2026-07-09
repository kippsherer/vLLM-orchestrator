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
	var needed int64
	if me.mem.measured {
		needed = me.mem.fullKVVRAMMB
	} else {
		o.ms.mu.RLock()
		smallest := o.ms.groups[0].measuredTotalVRAMMB
		for _, gs := range o.ms.groups[1:] {
			if gs.measuredTotalVRAMMB < smallest {
				smallest = gs.measuredTotalVRAMMB
			}
		}
		o.ms.mu.RUnlock()
		needed = int64(float64(smallest) * 0.85)
	}

	o.ms.mu.Lock()
	idx, err := o.pickGroup(needed)
	o.ms.mu.Unlock()

	if err != nil {
		// Run freeing memory rules — they manage ms.mu internally.
		if freeErr := o.freeMemoryRules(needed); freeErr != nil {
			return -1, fmt.Errorf("assign group: %w", freeErr)
		}
		o.ms.mu.Lock()
		idx, err = o.pickGroup(needed)
		o.ms.mu.Unlock()
		if err != nil {
			return -1, fmt.Errorf("assign group: still insufficient after freeing: %w", err)
		}
	}

	// Reserve VRAM and snapshot the reserved amount on the model entry.
	o.ms.mu.Lock()
	o.ms.groups[idx].usedVRAMMB += needed
	o.ms.mu.Unlock()
	me.reservedVRAMMB = needed
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
// Does NOT require ms.mu to be held on entry; rules manage the lock internally.
func (o *orchestrator) freeMemoryRules(neededMB int64) error {
	o.rule1(neededMB)
	o.rule2(neededMB)
	o.rule3()
	o.rule4(neededMB)

	o.ms.mu.RLock()
	defer o.ms.mu.RUnlock()
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
// If CPU RAM insufficient for a model, calls rule2ForCPU inline.
// Acquires ms.mu only for accounting reads/writes; releases it around HTTP calls.
func (o *orchestrator) rule1(neededMB int64) {
	for _, gs := range o.ms.groups {
		o.ms.mu.RLock()
		groupFree := gs.freeVRAMMB()
		o.ms.mu.RUnlock()
		if groupFree >= neededMB {
			continue
		}
		for _, me := range o.idleActiveModels() {
			me.mu.Lock()
			if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
				me.mu.Unlock()
				continue
			}
			weightsVRAM := me.mem.weightsVRAMMB
			reserved := me.reservedVRAMMB
			proc := me.proc
			me.mu.Unlock()

			o.ms.mu.RLock()
			cpuFree := o.ms.freeCPURAMB
			o.ms.mu.RUnlock()

			if cpuFree < weightsVRAM {
				o.rule2ForCPU(weightsVRAM)
				o.ms.mu.RLock()
				cpuFree = o.ms.freeCPURAMB
				o.ms.mu.RUnlock()
			}
			if cpuFree < weightsVRAM {
				continue
			}

			if err := sleepModel(proc, me.cfg.Name, 1); err != nil {
				log.Printf("[scheduler] rule1: sleep error %s: %v", me.cfg.Name, err)
				continue
			}

			me.mu.Lock()
			me.state = stateSleep1
			me.mu.Unlock()

			o.ms.mu.Lock()
			gs.usedVRAMMB -= reserved
			o.ms.freeCPURAMB -= weightsVRAM
			groupFree = gs.freeVRAMMB()
			o.ms.mu.Unlock()

			log.Printf("[scheduler] rule1: %s ACTIVE → SLEEP1", me.cfg.Name)
			if groupFree >= neededMB {
				break
			}
		}
	}
}

// rule2 terminates idle SLEEP2 processes (smallest first) until neededMB freed per group.
func (o *orchestrator) rule2(neededMB int64) {
	for _, gs := range o.ms.groups {
		o.ms.mu.RLock()
		groupFree := gs.freeVRAMMB()
		o.ms.mu.RUnlock()
		if groupFree >= neededMB {
			continue
		}
		for _, me := range o.idleSleep2Models() {
			me.mu.Lock()
			if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
				me.mu.Unlock()
				continue
			}
			proc := me.proc
			me.mu.Unlock()

			killProcess(proc, me.cfg.Name)

			me.mu.Lock()
			me.state = stateUnloaded
			me.proc = nil
			me.reservedVRAMMB = 0
			me.assignedGroupIdx = -1
			me.mu.Unlock()
			// SLEEP2 holds no VRAM in accounting — no usedVRAMMB change needed.
			log.Printf("[scheduler] rule2: %s SLEEP2 → UNLOADED", me.cfg.Name)

			o.ms.mu.RLock()
			groupFree = gs.freeVRAMMB()
			o.ms.mu.RUnlock()
			if groupFree >= neededMB {
				break
			}
		}
	}
}

// rule2ForCPU terminates idle SLEEP2 processes until freeCPURAMB >= neededCPUMB.
func (o *orchestrator) rule2ForCPU(neededCPUMB int64) {
	for _, me := range o.idleSleep2Models() {
		o.ms.mu.RLock()
		cpuFree := o.ms.freeCPURAMB
		o.ms.mu.RUnlock()
		if cpuFree >= neededCPUMB {
			break
		}
		me.mu.Lock()
		proc := me.proc
		me.mu.Unlock()

		killProcess(proc, me.cfg.Name)

		me.mu.Lock()
		me.state = stateUnloaded
		me.proc = nil
		me.reservedVRAMMB = 0
		me.assignedGroupIdx = -1
		me.mu.Unlock()
		log.Printf("[scheduler] rule2ForCPU: %s SLEEP2 → UNLOADED", me.cfg.Name)
	}
}

// rule3 moves idle SLEEP1 models to SLEEP2 to free CPU RAM.
func (o *orchestrator) rule3() {
	for _, me := range o.idleSleep1Models() {
		me.mu.Lock()
		proc := me.proc
		weightsVRAM := me.mem.weightsVRAMMB
		me.mu.Unlock()

		if err := sleepModel(proc, me.cfg.Name, 2); err != nil {
			log.Printf("[scheduler] rule3: sleep2 error %s: %v", me.cfg.Name, err)
			continue
		}

		me.mu.Lock()
		me.state = stateSleep2
		me.mu.Unlock()

		o.ms.mu.Lock()
		o.ms.freeCPURAMB += weightsVRAM
		o.ms.mu.Unlock()

		log.Printf("[scheduler] rule3: %s SLEEP1 → SLEEP2", me.cfg.Name)
	}
}

// rule4 is Rule 2 repeated after Rule 3 to terminate newly created SLEEP2 processes.
func (o *orchestrator) rule4(neededMB int64) {
	o.rule2(neededMB)
}
