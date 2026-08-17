package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var (
	reWeights = regexp.MustCompile(`Model loading took\s+([\d.]+)\s*GiB`)
	reKVCache = regexp.MustCompile(`Available KV cache memory:\s+([\d.]+)\s*GiB`)
)

// vllmProcess represents a running vLLM subprocess for one model.
type vllmProcess struct {
	cmd        *exec.Cmd
	socketPath string
	client     *http.Client  // HTTP client dialling the Unix socket
	exited     chan struct{} // closed when the OS process exits, for any reason
	onExit     func()        // called exactly once when the process exits for any reason
}

// launchVLLM starts a vLLM subprocess for modelCfg on gpuGroup and returns
// the process handle. Memory measurement goroutines are started immediately;
// readiness polling is done separately via waitForHealth.
func launchVLLM(modelCfg ModelConfig, socketPath string, group *groupState, mem *modelMemory) (*vllmProcess, error) {
	// Remove stale socket if present and unowned.
	if _, err := os.Stat(socketPath); err == nil {
		if err := checkSocketOwned(socketPath); err == nil {
			return nil, fmt.Errorf("launch %q: socket %s is owned by a live process", modelCfg.Name, socketPath)
		}
		os.Remove(socketPath)
	}

	visibleDevs := make([]string, len(group.gpus))
	for i, d := range group.gpus {
		visibleDevs[i] = strconv.Itoa(d)
	}
	cudaVisible := strings.Join(visibleDevs, ",")
	tpSize := len(group.gpus)
	if modelCfg.TensorParallelSize > 0 {
		tpSize = modelCfg.TensorParallelSize
	}
	tpSizeStr := strconv.Itoa(tpSize)

	var memArg []string
	if modelCfg.KVCacheMemoryGB > 0 {
		memArg = []string{"--kv-cache-memory-bytes", fmt.Sprintf("%gg", modelCfg.KVCacheMemoryGB)}
	} else {
		// --gpu-memory-utilization is a per-device fraction, so the denominator
		// must reflect only the GPUs vLLM actually uses (tpSize devices), not
		// group.measuredTotalVRAMMB (all GPUs in the group). These are equal
		// when tpSize == len(group.gpus) (the pre-override default), so this
		// is a no-op change for any model not using TensorParallelSize.
		perGPUVRAMMB := float64(group.measuredTotalVRAMMB) / float64(len(group.gpus))
		memArg = []string{"--gpu-memory-utilization", fmt.Sprintf("%.2f", float64(modelCfg.VRAMAllocationMB)/(perGPUVRAMMB*float64(tpSize)))}
	}
	args := append(append([]string{"serve", modelCfg.Name,
		"--uds", socketPath,
		"--tensor-parallel-size", tpSizeStr,
	}, memArg...), modelCfg.VLLMArgs...)

	cmd := exec.Command("vllm", args...)
	cmd.Env = buildEnv(cudaVisible)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("launch %q: stdout pipe: %w", modelCfg.Name, err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("launch %q: stderr pipe: %w", modelCfg.Name, err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("launch %q: %w", modelCfg.Name, err)
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		},
	}
	client := &http.Client{Transport: transport}

	vp := &vllmProcess{cmd: cmd, socketPath: socketPath, client: client, exited: make(chan struct{})}

	go drainAndMeasure(stdout, modelCfg.Name, mem)
	go drainAndMeasure(stderr, modelCfg.Name, nil)
	// Reap the top-level process to avoid zombies and signal waitForHealth
	// (via vp.exited) the moment it exits, rather than leaving the health
	// poll to discover this only after its full timeout. VRAM accounting is
	// NOT driven by this — vLLM's worker children outlive the parent process.
	go func() {
		cmd.Wait()
		close(vp.exited)
	}()

	return vp, nil
}

// buildEnv constructs the subprocess environment from the current env,
// injecting CUDA_VISIBLE_DEVICES, VLLM_SERVER_DEV_MODE=1, OMP_NUM_THREADS=8,
// VLLM_CPU_OMP_THREADS_BIND=auto, LD_PRELOAD=libtcmalloc_minimal, and
// VLLM_USE_FASTOKENS=1.
// OMP_NUM_THREADS=8 allows each GPU worker to spread PyTorch CPU ops (input
// tensor prep, attention assembly, sampling, KV cache) across multiple cores.
// VLLM_CPU_OMP_THREADS_BIND=auto lets vLLM pin those threads to cores local to
// each worker's GPU NUMA node.
// LD_PRELOAD replaces glibc malloc with tcmalloc's per-thread cache allocator,
// reducing lock contention under multi-threaded CPU load.
// VLLM_USE_FASTOKENS=1 enables the Rust BPE tokenizer backend for all BPE
// models (Qwen, DeepSeek, etc.), reducing tokenization overhead.
func buildEnv(cudaVisible string) []string {
	base := os.Environ()
	out := make([]string, 0, len(base)+7)
	for _, kv := range base {
		k := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			k = kv[:i]
		}
		if k == "CUDA_VISIBLE_DEVICES" || k == "CUDA_DEVICE_ORDER" || k == "VLLM_SERVER_DEV_MODE" || k == "OMP_NUM_THREADS" || k == "VLLM_CPU_OMP_THREADS_BIND" || k == "LD_PRELOAD" || k == "VLLM_USE_FASTOKENS" {
			continue
		}
		out = append(out, kv)
	}
	out = append(out,
		"CUDA_VISIBLE_DEVICES="+cudaVisible,
		"CUDA_DEVICE_ORDER=PCI_BUS_ID",
		"VLLM_SERVER_DEV_MODE=1",
		"OMP_NUM_THREADS=8",
		"VLLM_CPU_OMP_THREADS_BIND=auto",
		"LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
		"VLLM_USE_FASTOKENS=1",
	)
	return out
}

// launchLlamaCpp starts a llama-server subprocess for modelCfg on gpuGroup and
// returns the process handle. VRAM accounting is set synchronously from
// modelCfg.VRAMAllocationMB (no log parsing).
func launchLlamaCpp(modelCfg ModelConfig, socketPath string, group *groupState, mem *modelMemory, modelDir string) (*vllmProcess, error) {
	// Remove stale socket if present and unowned.
	if _, err := os.Stat(socketPath); err == nil {
		if err := checkSocketOwned(socketPath); err == nil {
			return nil, fmt.Errorf("launch %q: socket %s is owned by a live process", modelCfg.Name, socketPath)
		}
		os.Remove(socketPath)
	}

	visibleDevs := make([]string, len(group.gpus))
	for i, d := range group.gpus {
		visibleDevs[i] = strconv.Itoa(d)
	}
	cudaVisible := strings.Join(visibleDevs, ",")

	args := append([]string{
		"-m", filepath.Join(modelDir, modelCfg.GGUFPath),
		"--host", socketPath,
		"-a", modelCfg.Name,
	}, modelCfg.LlamaCppArgs...)

	cmd := exec.Command("llama-server", args...)
	cmd.Env = buildEnvLlamaCpp(cudaVisible)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("launch %q: stdout pipe: %w", modelCfg.Name, err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("launch %q: stderr pipe: %w", modelCfg.Name, err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("launch %q: %w", modelCfg.Name, err)
	}

	// VRAM accounting: authoritative from config, no log parsing.
	mem.fullKVVRAMMB = modelCfg.VRAMAllocationMB
	mem.measured = true

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		},
	}
	client := &http.Client{Transport: transport}

	vp := &vllmProcess{cmd: cmd, socketPath: socketPath, client: client, exited: make(chan struct{})}

	go drainAndMeasure(stdout, modelCfg.Name, nil)
	go drainAndMeasure(stderr, modelCfg.Name, nil)
	go func() {
		cmd.Wait()
		close(vp.exited)
	}()

	return vp, nil
}

// buildEnvLlamaCpp constructs the subprocess environment with only CUDA
// device visibility (no vLLM-specific tuning vars).
func buildEnvLlamaCpp(cudaVisible string) []string {
	base := os.Environ()
	out := make([]string, 0, len(base)+2)
	for _, kv := range base {
		k := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			k = kv[:i]
		}
		if k == "CUDA_VISIBLE_DEVICES" || k == "CUDA_DEVICE_ORDER" {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "CUDA_VISIBLE_DEVICES="+cudaVisible, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
}

// drainAndMeasure reads r, logs each line tagged with modelName, and (when mem
// is non-nil) parses memory measurement lines into mem. Drains to EOF always.
func drainAndMeasure(r io.Reader, modelName string, mem *modelMemory) {
	warnDeadline := time.Now().Add(3600 * time.Second)
	warnFired := false
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if strings.Contains(line, "tokens/s") || strings.Contains(line, " WARNING ") || strings.Contains(line, " ERROR ") {
			log.Printf("[vllm/%s] %s", modelName, line)
		} else {
			if verbose {
				log.Printf("[vllm/%s] %s", modelName, line)
			}
		}
		if mem == nil || mem.measured {
			continue
		}
		if m := reWeights.FindStringSubmatch(line); m != nil {
			gb, err := strconv.ParseFloat(m[1], 64)
			if err == nil {
				mem.weightsVRAMMB = int64(gb * 1024)
			}
		}
		if m := reKVCache.FindStringSubmatch(line); m != nil {
			gb, err := strconv.ParseFloat(m[1], 64)
			if err == nil {
				mem.fullKVVRAMMB = mem.weightsVRAMMB + int64(gb*1024)
				mem.measured = true
			}
		}
		if !mem.measured && !warnFired && time.Now().After(warnDeadline) {
			log.Printf("[vllm/%s] WARNING: memory values not found within 3600s; retaining placeholder", modelName)
			warnFired = true
			// Continue draining — do not break.
		}
	}
}

// waitForHealth polls GET /health on the vLLM process until 200 or 3600s
// timeout. Returns nil when ready. Fails immediately (without waiting out the
// timeout) if the process exits before becoming healthy. On timeout with the
// process still alive, kills it and removes the socket.
func waitForHealth(vp *vllmProcess, modelName string) error {
	deadline := time.Now().Add(3600 * time.Second)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for time.Now().Before(deadline) {
		resp, err := vp.client.Get("http://vllm/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-vp.exited:
			return fmt.Errorf("vllm %q: process exited before becoming healthy", modelName)
		case <-ticker.C:
		}
	}
	killProcess(vp, modelName)
	return fmt.Errorf("vllm %q: health poll timed out after 3600s", modelName)
}

// killProcess sends SIGTERM and waits 30s before SIGKILL, then removes the socket.
func killProcess(vp *vllmProcess, modelName string) {
	if vp.cmd.Process == nil {
		return
	}
	vp.cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		vp.cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		vp.cmd.Process.Kill()
		<-done
	}
	os.Remove(vp.socketPath)
	log.Printf("[vllm/%s] process stopped", modelName)
}

// checkSocketOwned returns nil if a live process owns the socket at path,
// error otherwise.
func checkSocketOwned(path string) error {
	conn, err := net.DialTimeout("unix", path, time.Second)
	if err != nil {
		return err
	}
	conn.Close()
	return nil
}

// sleepModel calls POST /sleep?level=<level> and polls /is_sleeping until true.
func sleepModel(vp *vllmProcess, modelName string, level int) error {
	url := fmt.Sprintf("http://vllm/sleep?level=%d", level)
	resp, err := vp.client.Post(url, "application/json", nil)
	if err != nil {
		return fmt.Errorf("vllm %q: POST /sleep?level=%d: %w", modelName, level, err)
	}
	resp.Body.Close()

	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		sleeping, err := pollIsSleeping(vp)
		if err == nil && sleeping {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("vllm %q: /is_sleeping did not become true within 30s", modelName)
}

// wakeModel calls POST /wake_up and polls /is_sleeping until false.
func wakeModel(vp *vllmProcess, modelName string) error {
	resp, err := vp.client.Post("http://vllm/wake_up", "application/json", nil)
	if err != nil {
		return fmt.Errorf("vllm %q: POST /wake_up: %w", modelName, err)
	}
	resp.Body.Close()

	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		sleeping, err := pollIsSleeping(vp)
		if err == nil && !sleeping {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("vllm %q: /is_sleeping did not become false within 60s", modelName)
}

// pollIsSleeping calls GET /is_sleeping and returns the is_sleeping bool value.
func pollIsSleeping(vp *vllmProcess) (bool, error) {
	resp, err := vp.client.Get("http://vllm/is_sleeping")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, err
	}
	// Minimal parse: look for "true" or "false" in the value of is_sleeping.
	s := string(body)
	if strings.Contains(s, `"is_sleeping":true`) {
		return true, nil
	}
	if strings.Contains(s, `"is_sleeping":false`) {
		return false, nil
	}
	return false, fmt.Errorf("unexpected /is_sleeping body: %s", s)
}
