package main

import (
	"fmt"
	"log"
	"sort"
	"time"
)

// assignGroup finds the smallest GPU group whose total VRAM can hold the model,
// running freeMemoryRules on each candidate group until one has enough free VRAM.
// Reserves VRAM on success.
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

	refreshMemory(o.ms)

	o.ms.mu.Lock()
	idx, err := o.pickGroup(needed)
	o.ms.mu.Unlock()

	if err != nil {
		// No group has enough free VRAM right now. Run freeing rules on each
		// group whose total VRAM is large enough to hold the model, stopping
		// as soon as one group has enough free VRAM.
		idx = -1
		o.ms.mu.RLock()
		var candidates []*groupState
		for _, gs := range o.ms.groups {
			if gs.measuredTotalVRAMMB >= needed {
				candidates = append(candidates, gs)
			}
		}
		o.ms.mu.RUnlock()
		for i, gs := range candidates {
			if o.freeMemoryRules(gs, needed) {
				// Find the index of this group in ms.groups.
				o.ms.mu.RLock()
				for j, g := range o.ms.groups {
					if g == gs {
						idx = j
						break
					}
				}
				o.ms.mu.RUnlock()
				_ = i
				break
			}
		}
		if idx < 0 {
			return -1, fmt.Errorf("assign group: insufficient VRAM for %d MB after freeing", needed)
		}
	}

	me.reservedVRAMMB = needed
	return idx, nil
}

// pickGroup finds the qualifying group with the smallest measuredTotalVRAMMB.
// Must be called with ms.mu held.
func (o *orchestrator) pickGroup(neededMB int64) (int, error) {
	best := -1
	var bestTotal int64
	for i, gs := range o.ms.groups {
		if gs.measuredFreeMB >= neededMB {
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

// freeMemoryRules applies the four eviction rules to a single GPU group gs.
// Rules are applied in order. After each individual model action, refreshMemory
// is called and free VRAM is checked — returning true immediately if enough is
// freed. Returns false if all four rules are exhausted without freeing enough.
//
// Rule 1: idle ACTIVE → SLEEP1 (smallest fullKVVRAMMB first; free CPU RAM first if needed)
// Rule 2: idle SLEEP2 → UNLOADED, kill process (smallest fullKVVRAMMB first)
// Rule 3: idle SLEEP1 → SLEEP2 (smallest weightsVRAMMB first)
// Rule 4: repeat Rule 2
func (o *orchestrator) freeMemoryRules(gs *groupState, neededMB int64) bool {
	// Rule 1: ACTIVE → SLEEP1
	for _, me := range o.idleActiveModels() {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		weightsVRAM := me.mem.weightsVRAMMB
		proc := me.proc
		me.mu.Unlock()

		// Free CPU RAM first if needed (rule 1 spec: follow rule 2 if cpu ram needs freed).
		o.ms.mu.RLock()
		cpuFree := o.ms.freeCPURAMB
		o.ms.mu.RUnlock()
		if cpuFree < weightsVRAM {
			o.freeCPURam(weightsVRAM)
			o.ms.mu.RLock()
			cpuFree = o.ms.freeCPURAMB
			o.ms.mu.RUnlock()
		}
		if cpuFree < weightsVRAM {
			continue
		}

		o.ms.mu.RLock()
		freeBefore := gs.measuredFreeMB
		o.ms.mu.RUnlock()

		if err := sleepModel(proc, me.cfg.Name, 1); err != nil {
			log.Printf("[scheduler] rule1: sleep error %s: %v", me.cfg.Name, err)
			continue
		}
		me.mu.Lock()
		me.state = stateSleep1
		me.mu.Unlock()
		o.ms.mu.Lock()
		o.ms.freeCPURAMB -= weightsVRAM
		o.ms.mu.Unlock()

		refreshMemory(o.ms)
		o.ms.mu.RLock()
		free := gs.measuredFreeMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] rule1: free %dMB → %dMB  %s  ACTIVE → SLEEP1", freeBefore, free, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	// Rule 2: SLEEP2 → UNLOADED
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

		o.ms.mu.RLock()
		freeBefore := gs.measuredFreeMB
		o.ms.mu.RUnlock()

		killProcess(proc, me.cfg.Name)
		waitVRAMStable(o.ms, gs)
		o.ms.mu.RLock()
		free := gs.measuredFreeMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] rule2: free %dMB → %dMB  %s  SLEEP2 → UNLOADED", freeBefore, free, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	// Rule 3: SLEEP1 → SLEEP2
	for _, me := range o.idleSleep1Models() {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		proc := me.proc
		me.mu.Unlock()

		o.ms.mu.RLock()
		freeBefore := gs.measuredFreeMB
		o.ms.mu.RUnlock()

		if err := sleepModel(proc, me.cfg.Name, 2); err != nil {
			log.Printf("[scheduler] rule3: sleep2 error %s: %v", me.cfg.Name, err)
			continue
		}
		me.mu.Lock()
		me.state = stateSleep2
		me.mu.Unlock()

		refreshMemory(o.ms)
		o.ms.mu.RLock()
		free := gs.measuredFreeMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] rule3: free %dMB → %dMB  %s  SLEEP1 → SLEEP2", freeBefore, free, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	// Rule 4: repeat Rule 2
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

		o.ms.mu.RLock()
		freeBefore := gs.measuredFreeMB
		o.ms.mu.RUnlock()

		killProcess(proc, me.cfg.Name)
		waitVRAMStable(o.ms, gs)
		o.ms.mu.RLock()
		free := gs.measuredFreeMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] rule4: free %dMB → %dMB  %s  SLEEP2 → UNLOADED", freeBefore, free, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	return false
}

// freeCPURam demotes idle SLEEP1 models to SLEEP2 (smallest weightsVRAMMB first)
// until real free CPU RAM >= neededCPUMB. Called inline from rule 1.
func (o *orchestrator) freeCPURam(neededCPUMB int64) {
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
			log.Printf("[scheduler] freeCPURam: sleep2 error %s: %v", me.cfg.Name, err)
			continue
		}
		me.mu.Lock()
		me.state = stateSleep2
		me.mu.Unlock()
		refreshMemory(o.ms)
		o.ms.mu.RLock()
		cpuAfter := o.ms.freeCPURAMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] freeCPURam: cpu free %dMB → %dMB  %s  SLEEP1 → SLEEP2", cpuFree, cpuAfter, me.cfg.Name)
	}
}

// waitVRAMStable polls nvidia-smi after a process kill until measuredFreeMB
// stops increasing between consecutive reads (500ms apart), or 10 seconds elapse.
// This replaces a fixed sleep and ensures the GPU driver has finished reclaiming
// memory before the caller checks whether enough is free.
func waitVRAMStable(ms *memoryState, gs *groupState) {
	deadline := time.Now().Add(10 * time.Second)
	ms.mu.RLock()
	prev := gs.measuredFreeMB
	ms.mu.RUnlock()
	for time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
		refreshMemory(ms)
		ms.mu.RLock()
		cur := gs.measuredFreeMB
		ms.mu.RUnlock()
		if cur <= prev {
			return
		}
		prev = cur
	}
}

// idleActiveModels returns idle ACTIVE models sorted smallest fullKVVRAMMB first.
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

// idleSleep2Models returns idle SLEEP2 models sorted smallest fullKVVRAMMB first.
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

// idleSleep1Models returns idle SLEEP1 models sorted smallest weightsVRAMMB first.
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
