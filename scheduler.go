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
// effectiveFree is the minimum of freeVRAMMB() and measuredFreeMB (when measured).
func (o *orchestrator) pickGroup(neededMB int64) (int, error) {
	best := -1
	var bestTotal int64
	for i, gs := range o.ms.groups {
		free := gs.freeVRAMMB()
		if gs.measuredFreeMB >= 0 && gs.measuredFreeMB < free {
			free = gs.measuredFreeMB
		}
		if free >= neededMB {
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

// freeMemoryRules runs the §6 eviction cycle per GPU group.
// Each rule iterates smallest-model-first and calls groupHasEnough (refreshMemory
// + real VRAM check) after every individual model action, stopping that rule as
// soon as the group has enough. Rules run in order 1→2→3→4 per group.
// Returns nil as soon as any group reaches neededMB free; error if exhausted.
//
//	Rule 1: ACTIVE → SLEEP1 (free CPU RAM via rule2ForCPU first if needed)
//	Rule 2: SLEEP2 → UNLOADED (kill process)
//	Rule 3: SLEEP1 → SLEEP2
//	Rule 4: repeat Rule 2 (handles models newly in SLEEP2 after Rule 3)
func (o *orchestrator) freeMemoryRules(neededMB int64) error {
	for _, gs := range o.ms.groups {
		o.rule1ForGroup(gs, neededMB)
		o.rule2ForGroup(gs, neededMB)
		o.rule3ForGroup(gs, neededMB)
		o.rule2ForGroup(gs, neededMB) // rule 4 = repeat rule 2
		if o.groupHasEnough(gs, neededMB) {
			return nil
		}
	}
	return fmt.Errorf("freeing memory rules exhausted; insufficient VRAM for %d MB", neededMB)
}

// groupHasEnough calls refreshMemory to get real free VRAM, then checks whether
// gs has at least neededMB free. It is the only place that decides "enough freed".
func (o *orchestrator) groupHasEnough(gs *groupState, neededMB int64) bool {
	refreshMemory(o.ms)
	o.ms.mu.RLock()
	free := gs.measuredFreeMB
	o.ms.mu.RUnlock()
	return free >= neededMB
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

// rule1ForGroup moves idle ACTIVE models on gs to SLEEP1, smallest first.
// Before each model action, checks whether CPU RAM is sufficient; if not,
// calls rule2ForCPU to free CPU RAM first (by demoting SLEEP1→SLEEP2 on
// other groups). Stops as soon as gs has enough free VRAM.
func (o *orchestrator) rule1ForGroup(gs *groupState, neededMB int64) {
	for _, me := range o.idleActiveModels() {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		weightsVRAM := me.mem.weightsVRAMMB
		proc := me.proc
		me.mu.Unlock()

		// Ensure CPU RAM is available before moving to SLEEP1.
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
		groupIdx := me.assignedGroupIdx
		reserved := me.reservedVRAMMB
		me.state = stateSleep1
		me.mu.Unlock()
		o.ms.mu.Lock()
		if groupIdx >= 0 {
			o.ms.groups[groupIdx].usedVRAMMB -= reserved
		}
		o.ms.freeCPURAMB -= weightsVRAM
		o.ms.mu.Unlock()
		log.Printf("[scheduler] rule1: %s ACTIVE → SLEEP1", me.cfg.Name)

		if o.groupHasEnough(gs, neededMB) {
			return
		}
	}
}

// rule2ForGroup kills idle SLEEP2 processes on gs, smallest first, until gs
// has enough free VRAM. Real free VRAM is re-measured after each kill.
func (o *orchestrator) rule2ForGroup(gs *groupState, neededMB int64) {
	for _, me := range o.idleSleep2Models() {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		proc := me.proc
		me.state = stateUnloaded
		me.proc = nil
		me.reservedVRAMMB = 0
		me.assignedGroupIdx = -1
		me.mu.Unlock()

		killProcess(proc, me.cfg.Name)
		log.Printf("[scheduler] rule2: %s SLEEP2 → UNLOADED", me.cfg.Name)

		if o.groupHasEnough(gs, neededMB) {
			return
		}
	}
}

// rule2ForCPU demotes idle SLEEP1 models to SLEEP2 (smallest weightsVRAMMB first)
// until real free CPU RAM >= neededCPUMB. Reads /proc/meminfo after each action.
func (o *orchestrator) rule2ForCPU(neededCPUMB int64) {
	for _, me := range o.idleSleep1Models() {
		refreshMemory(o.ms)
		o.ms.mu.RLock()
		cpuFree := o.ms.freeCPURAMB
		o.ms.mu.RUnlock()
		if cpuFree >= neededCPUMB {
			return
		}

		me.mu.Lock()
		proc := me.proc
		me.mu.Unlock()

		if err := sleepModel(proc, me.cfg.Name, 2); err != nil {
			log.Printf("[scheduler] rule2ForCPU: sleep2 error %s: %v", me.cfg.Name, err)
			continue
		}
		me.mu.Lock()
		me.state = stateSleep2
		me.mu.Unlock()
		log.Printf("[scheduler] rule2ForCPU: %s SLEEP1 → SLEEP2", me.cfg.Name)
	}
}

// rule3ForGroup demotes idle SLEEP1 models on gs to SLEEP2, smallest first,
// until gs has enough free VRAM.
func (o *orchestrator) rule3ForGroup(gs *groupState, neededMB int64) {
	for _, me := range o.idleSleep1Models() {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		proc := me.proc
		me.mu.Unlock()

		if err := sleepModel(proc, me.cfg.Name, 2); err != nil {
			log.Printf("[scheduler] rule3: sleep2 error %s: %v", me.cfg.Name, err)
			continue
		}
		me.mu.Lock()
		me.state = stateSleep2
		me.mu.Unlock()
		log.Printf("[scheduler] rule3: %s SLEEP1 → SLEEP2", me.cfg.Name)

		if o.groupHasEnough(gs, neededMB) {
			return
		}
	}
}
