package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// groupState tracks VRAM accounting for a single GPU group.
type groupState struct {
	id                  string
	gpus                []int
	measuredTotalVRAMMB int64 // sum of total VRAM across all GPUs in group
	measuredFreeMB      int64 // total - used across all GPUs; -1 until first refreshMemory call
}

// memoryState holds all VRAM and CPU RAM accounting.
type memoryState struct {
	mu          sync.RWMutex
	groups      []*groupState
	freeCPURAMB int64 // from /proc/meminfo MemAvailable, in MB
}

// modelMemory caches per-model VRAM measurements from vLLM startup logs.
type modelMemory struct {
	weightsVRAMMB int64
	fullKVVRAMMB  int64
	measured      bool
}

// queryNvidiaSmi is the function used to obtain nvidia-smi total-memory output.
// Replaced in tests via assignment.
var queryNvidiaSmi = func() (string, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,memory.total",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return "", fmt.Errorf("nvidia-smi: %w", err)
	}
	return string(out), nil
}

// queryNvidiaSmiFreeMB is the function used to obtain nvidia-smi free-memory output.
// Replaced in tests via assignment.
var queryNvidiaSmiFreeMB = func() (string, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,memory.free",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return "", fmt.Errorf("nvidia-smi free: %w", err)
	}
	return string(out), nil
}

// initMemory queries nvidia-smi and /proc/meminfo, validates that every
// configured GPU device ID exists, and returns an initialised memoryState.
func initMemory(cfg *Config) (*memoryState, error) {
	smiOut, err := queryNvidiaSmi()
	if err != nil {
		return nil, err
	}
	deviceMB, err := parseNvidiaSmi(smiOut)
	if err != nil {
		return nil, err
	}

	// Validate that every configured GPU ID is present.
	for _, g := range cfg.GPUGroups {
		for _, dev := range g.GPUs {
			if _, ok := deviceMB[dev]; !ok {
				return nil, fmt.Errorf("config: GPU device %d (group %q) not found in nvidia-smi output", dev, g.ID)
			}
		}
	}

	ms := &memoryState{}
	for _, g := range cfg.GPUGroups {
		var total int64
		for _, dev := range g.GPUs {
			total += deviceMB[dev]
		}
		gs := &groupState{
			id:                  g.ID,
			gpus:                g.GPUs,
			measuredTotalVRAMMB: total,
			measuredFreeMB:      -1, // not yet measured; refreshMemory fills this in
		}
		ms.groups = append(ms.groups, gs)
	}

	cpuMB, err := readMemAvailableMB()
	if err != nil {
		return nil, err
	}
	ms.freeCPURAMB = cpuMB
	return ms, nil
}

// parseNvidiaSmi parses the output of:
//
//	nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits
//
// Returns a map from device index to total VRAM in MB.
func parseNvidiaSmi(output string) (map[int]int64, error) {
	m := make(map[int]int64)
	sc := bufio.NewScanner(strings.NewReader(output))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("nvidia-smi: unexpected line %q", line)
		}
		idx, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi: bad index in %q: %w", line, err)
		}
		mb, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi: bad memory value in %q: %w", line, err)
		}
		m[idx] = mb
	}
	if len(m) == 0 {
		return nil, fmt.Errorf("nvidia-smi: no GPU entries found")
	}
	return m, nil
}

var readMemAvailableMB = func() (int64, error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, fmt.Errorf("read /proc/meminfo: %w", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		// Format: "MemAvailable:   12345678 kB"
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("/proc/meminfo: unexpected MemAvailable line %q", line)
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return 0, fmt.Errorf("/proc/meminfo: bad MemAvailable value %q: %w", fields[1], err)
		}
		return kb / 1024, nil
	}
	return 0, fmt.Errorf("/proc/meminfo: MemAvailable not found")
}

// startPeriodicMemoryRefresh launches a goroutine that re-queries nvidia-smi
// and /proc/meminfo every 60 seconds, logging a warning if measured free VRAM
// is less than accounted.
func startPeriodicMemoryRefresh(ms *memoryState) {
	go func() {
		t := time.NewTicker(60 * time.Second)
		defer t.Stop()
		for range t.C {
			refreshMemory(ms)
		}
	}()
}

func refreshMemory(ms *memoryState) {
	freeOut, err := queryNvidiaSmiFreeMB()
	if err != nil {
		log.Printf("memory refresh: nvidia-smi free query error: %v", err)
		return
	}
	freeByDev, err := parseNvidiaSmi(freeOut)
	if err != nil {
		log.Printf("memory refresh: parse free VRAM: %v", err)
		return
	}

	ms.mu.Lock()
	defer ms.mu.Unlock()

	for _, gs := range ms.groups {
		var freeMB int64
		for _, dev := range gs.gpus {
			if mb, ok := freeByDev[dev]; ok {
				freeMB += mb
			}
		}
		gs.measuredFreeMB = freeMB
	}

	cpuMB, err := readMemAvailableMB()
	if err != nil {
		log.Printf("memory refresh: /proc/meminfo: %v", err)
		return
	}
	ms.freeCPURAMB = cpuMB
}
