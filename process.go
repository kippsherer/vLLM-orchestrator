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
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var (
	reWeights = regexp.MustCompile(`Loading model weights took\s+([\d.]+)\s*GB`)
	reKVCache = regexp.MustCompile(`GPU KV cache size:\s+([\d.]+)\s*GB`)
)

// vllmProcess represents a running vLLM subprocess for one model.
type vllmProcess struct {
	cmd        *exec.Cmd
	socketPath string
	client     *http.Client // HTTP client dialling the Unix socket
	onExit     func()       // called exactly once when the process exits for any reason
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
	tpSize := strconv.Itoa(len(group.gpus))

	args := append([]string{"serve", modelCfg.Name,
		"--uds", socketPath,
		"--tensor-parallel-size", tpSize,
	}, modelCfg.VLLMArgs...)

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

	vp := &vllmProcess{cmd: cmd, socketPath: socketPath, client: client}

	go drainAndMeasure(stdout, modelCfg.Name, mem, true)
	go drainLog(stderr, modelCfg.Name)
	// Reap the top-level process to avoid zombies. VRAM accounting is NOT
	// driven by this — vLLM's worker children outlive the parent process.
	go cmd.Wait()

	return vp, nil
}

// buildEnv constructs the subprocess environment from the current env,
// injecting CUDA_VISIBLE_DEVICES and VLLM_SERVER_DEV_MODE=1.
func buildEnv(cudaVisible string) []string {
	base := os.Environ()
	out := make([]string, 0, len(base)+2)
	for _, kv := range base {
		k := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			k = kv[:i]
		}
		if k == "CUDA_VISIBLE_DEVICES" || k == "VLLM_SERVER_DEV_MODE" {
			continue
		}
		out = append(out, kv)
	}
	out = append(out, "CUDA_VISIBLE_DEVICES="+cudaVisible, "VLLM_SERVER_DEV_MODE=1")
	return out
}

// drainAndMeasure reads stdout, logs each line tagged with modelName, and
// parses memory measurement lines into mem. measured is set true when both
// values are found. Stdout is always drained to EOF regardless of measurement status.
func drainAndMeasure(r io.Reader, modelName string, mem *modelMemory, isMeasurement bool) {
	warnDeadline := time.Now().Add(3600 * time.Second)
	warnFired := false
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		log.Printf("[vllm/%s] %s", modelName, line)
		if !isMeasurement || mem.measured {
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
	// drain remaining lines even after measurement complete or deadline
	for sc.Scan() {
		log.Printf("[vllm/%s] %s", modelName, sc.Text())
	}
}

// drainLog reads r and logs each line tagged with modelName.
func drainLog(r io.Reader, modelName string) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		log.Printf("[vllm/%s] %s", modelName, sc.Text())
	}
}

// waitForHealth polls GET /health on the vLLM process until 200 or 3600s timeout.
// Returns nil when ready. On timeout, kills the process and removes the socket.
func waitForHealth(vp *vllmProcess, modelName string) error {
	deadline := time.Now().Add(3600 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := vp.client.Get("http://vllm/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	killProcess(vp, modelName)
	return fmt.Errorf("vllm %q: health poll timed out after 300s", modelName)
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

// runCommand executes a command and returns combined stdout, or an error.
func runCommand(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
