package main

import (
	"fmt"
	"log"
	"sort"
	"time"
)

// assignGroup finds the smallest GPU group whose total VRAM can hold the model,
// running freeMemoryRules on each candidate group until one has enough free VRAM.
// Reserves VRAM on success. If the model config specifies a gpu_group, that group
// is used directly, bypassing the smallest-fit selection logic.
func (o *orchestrator) assignGroup(me *modelEntry) (int, error) {
	// Per-model GPU group pin: skip scheduling logic entirely.
	if me.cfg.GPUGroup != "" {
		for i, gs := range o.ms.groups {
			if gs.id == me.cfg.GPUGroup {
				if o.groupHasOtherModels(i, me) {
					o.freeMemoryRules(gs, me.cfg.VRAMAllocationMB)
				}
				refreshMemory(o.ms)
				o.ms.mu.Lock()
				o.ms.groups[i].measuredFreeMB -= me.cfg.VRAMAllocationMB
				o.ms.mu.Unlock()
				me.reservedVRAMMB = me.cfg.VRAMAllocationMB
				log.Printf("[scheduler] model %s pinned to group %s", me.cfg.Name, me.cfg.GPUGroup)
				return i, nil
			}
		}
		return -1, fmt.Errorf("assign group: pinned group %q not found", me.cfg.GPUGroup)
	}

	var needed int64
	if me.mem.measured {
		needed = me.mem.fullKVVRAMMB
	} else if me.cfg.VRAMAllocationMB > 0 {
		needed = me.cfg.VRAMAllocationMB
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

	// Run freeing rules on any group that has other models assigned to it,
	// regardless of what nvidia-smi currently reports as free. A model in
	// SLEEP1 has its weights offloaded but its KV cache still allocated on
	// the GPU — nvidia-smi correctly shows that KV memory as used, but the
	// margin may be thin enough that pickGroup passes while the actual
	// per-device headroom is insufficient for the incoming model to start.
	// Evicting first guarantees a clean GPU before launch.
	for i, gs := range o.ms.groups {
		if gs.measuredTotalVRAMMB < needed {
			continue
		}
		if o.groupHasOtherModels(i, me) {
			o.freeMemoryRules(gs, needed)
		}
	}

	refreshMemory(o.ms)

	o.ms.mu.Lock()
	idx, err := o.pickGroup(needed)
	o.ms.mu.Unlock()

	if err != nil {
		idx = -1

		// Log every model's current state so failures are diagnosable.
		for _, me := range o.models {
			me.mu.Lock()
			log.Printf("[scheduler] model %s: state=%s activeRequests=%d reservedVRAM=%dMB assignedGroup=%d",
				me.cfg.Name, me.state, me.activeRequests, me.reservedVRAMMB, me.assignedGroupIdx)
			me.mu.Unlock()
		}
		o.ms.mu.RLock()
		type candidateGroup struct {
			idx int
			gs  *groupState
		}
		var candidates []candidateGroup
		for i, gs := range o.ms.groups {
			if gs.measuredTotalVRAMMB >= needed {
				candidates = append(candidates, candidateGroup{i, gs})
			}
		}
		o.ms.mu.RUnlock()
		for _, c := range candidates {
			if o.freeMemoryRules(c.gs, needed) {
				idx = c.idx
				break
			}
		}
		if idx < 0 {
			o.ms.mu.RLock()
			for _, gs := range o.ms.groups {
				log.Printf("[scheduler] assignGroup fail: group %s free=%dMB needed=%dMB",
					gs.id, gs.measuredFreeMB, needed)
			}
			o.ms.mu.RUnlock()
			return -1, fmt.Errorf("assign group: insufficient VRAM for %d MB after freeing", needed)
		}
	}

	me.reservedVRAMMB = needed
	return idx, nil
}

// groupHasOtherModels reports whether any model other than exclude is
// assigned to group index groupIdx.
func (o *orchestrator) groupHasOtherModels(groupIdx int, exclude *modelEntry) bool {
	for _, m := range o.models {
		if m == exclude {
			continue
		}
		m.mu.Lock()
		assigned := m.assignedGroupIdx
		m.mu.Unlock()
		if assigned == groupIdx {
			return true
		}
	}
	return false
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
	// Does not require activeRequests==0: sleep level 1 only offloads KV cache
	// to CPU RAM; vLLM itself defers the offload until in-flight requests drain.
	candidates := o.activeModels(func(a, b *modelEntry) bool { return a.mem.fullKVVRAMMB < b.mem.fullKVVRAMMB })
	log.Printf("[scheduler] rule1: %d ACTIVE model(s) on group %s", len(candidates), gs.id)
	for _, me := range candidates {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		if me.cfg.Engine == engineLlamaCpp {
			if me.activeRequests > 0 {
				me.mu.Unlock()
				continue
			}
			proc := me.proc
			expectedMB := me.reservedVRAMMB
			me.mu.Unlock()

			o.ms.mu.RLock()
			freeBefore := gs.measuredFreeMB
			o.ms.mu.RUnlock()

			o.killAndUnload(me, proc, "evicted to free VRAM")
			waitVRAMStable(o.ms, gs, freeBefore, expectedMB)
			o.ms.mu.RLock()
			free := gs.measuredFreeMB
			o.ms.mu.RUnlock()
			log.Printf("[scheduler] rule1: free %dMB → %dMB (needed %dMB)  %s  ACTIVE → UNLOADED (llama_cpp)", freeBefore, free, neededMB, me.cfg.Name)
			if free >= neededMB {
				return true
			}
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
		log.Printf("[scheduler] rule1: free %dMB → %dMB (needed %dMB)  %s  ACTIVE → SLEEP1", freeBefore, free, neededMB, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	// Rule 2: SLEEP2 → UNLOADED
	for _, me := range o.idleModels(stateSleep2, func(a, b *modelEntry) bool { return a.mem.fullKVVRAMMB < b.mem.fullKVVRAMMB }) {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		proc := me.proc
		expectedMB := me.reservedVRAMMB
		me.state = stateUnloaded
		me.proc = nil
		me.reservedVRAMMB = 0
		me.assignedGroupIdx = -1
		me.mu.Unlock()

		o.ms.mu.RLock()
		freeBefore := gs.measuredFreeMB
		o.ms.mu.RUnlock()

		killProcess(proc, me.cfg.Name)
		waitVRAMStable(o.ms, gs, freeBefore, expectedMB)
		o.ms.mu.RLock()
		free := gs.measuredFreeMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] rule2: free %dMB → %dMB (needed %dMB)  %s  SLEEP2 → UNLOADED", freeBefore, free, neededMB, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	// Rule 3: SLEEP1 → SLEEP2
	for _, me := range o.idleModels(stateSleep1, func(a, b *modelEntry) bool { return a.mem.weightsVRAMMB < b.mem.weightsVRAMMB }) {
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
		log.Printf("[scheduler] rule3: free %dMB → %dMB (needed %dMB)  %s  SLEEP1 → SLEEP2", freeBefore, free, neededMB, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	// Rule 4: repeat Rule 2
	for _, me := range o.idleModels(stateSleep2, func(a, b *modelEntry) bool { return a.mem.fullKVVRAMMB < b.mem.fullKVVRAMMB }) {
		me.mu.Lock()
		if me.assignedGroupIdx < 0 || o.ms.groups[me.assignedGroupIdx] != gs {
			me.mu.Unlock()
			continue
		}
		proc := me.proc
		expectedMB := me.reservedVRAMMB
		me.state = stateUnloaded
		me.proc = nil
		me.reservedVRAMMB = 0
		me.assignedGroupIdx = -1
		me.mu.Unlock()

		o.ms.mu.RLock()
		freeBefore := gs.measuredFreeMB
		o.ms.mu.RUnlock()

		killProcess(proc, me.cfg.Name)
		waitVRAMStable(o.ms, gs, freeBefore, expectedMB)
		o.ms.mu.RLock()
		free := gs.measuredFreeMB
		o.ms.mu.RUnlock()
		log.Printf("[scheduler] rule4: free %dMB → %dMB (needed %dMB)  %s  SLEEP2 → UNLOADED", freeBefore, free, neededMB, me.cfg.Name)
		if free >= neededMB {
			return true
		}
	}

	return false
}

// freeCPURam demotes idle SLEEP1 models to SLEEP2 (smallest weightsVRAMMB first)
// until real free CPU RAM >= neededCPUMB. Called inline from rule 1.
func (o *orchestrator) freeCPURam(neededCPUMB int64) {
	for _, me := range o.idleModels(stateSleep1, func(a, b *modelEntry) bool { return a.mem.weightsVRAMMB < b.mem.weightsVRAMMB }) {
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

// waitVRAMStable polls nvidia-smi after a process kill until the GPU group has
// reclaimed at least 75% of expectedMB (measured from freeBefore) AND free VRAM
// has not changed by more than 1% of expectedMB between consecutive 500ms reads.
// This avoids both false-early-exit (nvidia-smi momentary plateau before driver
// finishes releasing NCCL/graph-pool memory) and infinite wait on a busy server
// where VRAM usage is never perfectly flat. Deadline is 30 seconds.
func waitVRAMStable(ms *memoryState, gs *groupState, freeBefore, expectedMB int64) {
	threshold75 := freeBefore + (expectedMB*75)/100
	jitter := expectedMB / 100 // 1% of expected as noise floor
	if jitter < 10 {
		jitter = 10 // minimum 10 MB noise floor
	}
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
		diff := cur - prev
		if diff < 0 {
			diff = -diff
		}
		if cur >= threshold75 && diff <= jitter {
			return
		}
		prev = cur
	}
}

// idleModels returns idle models in the given state, sorted by less.
func (o *orchestrator) idleModels(state modelState, less func(a, b *modelEntry) bool) []*modelEntry {
	var out []*modelEntry
	for _, me := range o.models {
		me.mu.Lock()
		s := me.state
		active := me.activeRequests
		me.mu.Unlock()
		if s == state && active == 0 {
			out = append(out, me)
		}
	}
	sort.Slice(out, func(i, j int) bool { return less(out[i], out[j]) })
	return out
}

// activeModels returns all models in stateActive regardless of activeRequests,
// sorted by less. Used by Rule 1 (ACTIVE → SLEEP1) where sleep is safe while
// requests are in-flight because vLLM defers the KV offload until they drain.
func (o *orchestrator) activeModels(less func(a, b *modelEntry) bool) []*modelEntry {
	var out []*modelEntry
	for _, me := range o.models {
		me.mu.Lock()
		s := me.state
		me.mu.Unlock()
		if s == stateActive {
			out = append(out, me)
		}
	}
	sort.Slice(out, func(i, j int) bool { return less(out[i], out[j]) })
	return out
}
