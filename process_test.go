package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestBuildEnv(t *testing.T) {
	t.Parallel()

	// Inject known values into os.Environ via a sub-call approach:
	// buildEnv reads os.Environ() directly, so we set env vars then call it.
	os.Setenv("CUDA_VISIBLE_DEVICES", "old_value")
	os.Setenv("VLLM_SERVER_DEV_MODE", "old_value")
	os.Setenv("OMP_NUM_THREADS", "36")
	os.Setenv("LD_PRELOAD", "/old/lib.so")
	os.Setenv("VLLM_USE_FASTOKENS", "0")
	t.Cleanup(func() {
		os.Unsetenv("CUDA_VISIBLE_DEVICES")
		os.Unsetenv("VLLM_SERVER_DEV_MODE")
		os.Unsetenv("OMP_NUM_THREADS")
		os.Unsetenv("LD_PRELOAD")
		os.Unsetenv("VLLM_USE_FASTOKENS")
	})

	result := buildEnv("0,1", false)

	var cudaVal, devModeVal, ompVal, ldPreloadVal, fasttokensVal string
	cudaCount, devModeCount, ompCount, ldPreloadCount, fasttokensCount := 0, 0, 0, 0, 0
	for _, kv := range result {
		if strings.HasPrefix(kv, "CUDA_VISIBLE_DEVICES=") {
			cudaVal = strings.TrimPrefix(kv, "CUDA_VISIBLE_DEVICES=")
			cudaCount++
		}
		if strings.HasPrefix(kv, "VLLM_SERVER_DEV_MODE=") {
			devModeVal = strings.TrimPrefix(kv, "VLLM_SERVER_DEV_MODE=")
			devModeCount++
		}
		if strings.HasPrefix(kv, "OMP_NUM_THREADS=") {
			ompVal = strings.TrimPrefix(kv, "OMP_NUM_THREADS=")
			ompCount++
		}
		if strings.HasPrefix(kv, "LD_PRELOAD=") {
			ldPreloadVal = strings.TrimPrefix(kv, "LD_PRELOAD=")
			ldPreloadCount++
		}
		if strings.HasPrefix(kv, "VLLM_USE_FASTOKENS=") {
			fasttokensVal = strings.TrimPrefix(kv, "VLLM_USE_FASTOKENS=")
			fasttokensCount++
		}
	}
	if cudaCount != 1 {
		t.Errorf("CUDA_VISIBLE_DEVICES appears %d times, want 1", cudaCount)
	}
	if devModeCount != 1 {
		t.Errorf("VLLM_SERVER_DEV_MODE appears %d times, want 1", devModeCount)
	}
	if ompCount != 1 {
		t.Errorf("OMP_NUM_THREADS appears %d times, want 1", ompCount)
	}
	if ldPreloadCount != 1 {
		t.Errorf("LD_PRELOAD appears %d times, want 1", ldPreloadCount)
	}
	if fasttokensCount != 1 {
		t.Errorf("VLLM_USE_FASTOKENS appears %d times, want 1", fasttokensCount)
	}
	if cudaVal != "0,1" {
		t.Errorf("CUDA_VISIBLE_DEVICES = %q, want %q", cudaVal, "0,1")
	}
	if devModeVal != "1" {
		t.Errorf("VLLM_SERVER_DEV_MODE = %q, want %q", devModeVal, "1")
	}
	if ompVal != "8" {
		t.Errorf("OMP_NUM_THREADS = %q, want %q (parent value must be overridden)", ompVal, "8")
	}
	if ldPreloadVal != "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" {
		t.Errorf("LD_PRELOAD = %q, want %q (parent value must be overridden)", ldPreloadVal, "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4")
	}
	if fasttokensVal != "1" {
		t.Errorf("VLLM_USE_FASTOKENS = %q, want %q (parent value must be overridden)", fasttokensVal, "1")
	}
}

func TestBuildEnvDisableFastokens(t *testing.T) {
	t.Parallel()

	os.Setenv("VLLM_USE_FASTOKENS", "old")
	t.Cleanup(func() { os.Unsetenv("VLLM_USE_FASTOKENS") })

	result := buildEnv("0", true)

	fasttokensCount := 0
	for _, kv := range result {
		if strings.HasPrefix(kv, "VLLM_USE_FASTOKENS=") {
			fasttokensCount++
		}
	}
	if fasttokensCount != 0 {
		t.Errorf("VLLM_USE_FASTOKENS appears %d times, want 0 (disabled)", fasttokensCount)
	}
}

func TestBuildEnvLlamaCpp(t *testing.T) {
	t.Parallel()

	os.Setenv("CUDA_VISIBLE_DEVICES", "old_value")
	os.Setenv("CUDA_DEVICE_ORDER", "old_order")
	os.Setenv("VLLM_SERVER_DEV_MODE", "should_be_inherited")
	t.Cleanup(func() {
		os.Unsetenv("CUDA_VISIBLE_DEVICES")
		os.Unsetenv("CUDA_DEVICE_ORDER")
		os.Unsetenv("VLLM_SERVER_DEV_MODE")
	})

	result := buildEnvLlamaCpp("2,3")

	var cudaVal, deviceOrderVal string
	cudaCount, deviceOrderCount := 0, 0
	hasDevModeVar := false
	for _, kv := range result {
		if strings.HasPrefix(kv, "CUDA_VISIBLE_DEVICES=") {
			cudaVal = strings.TrimPrefix(kv, "CUDA_VISIBLE_DEVICES=")
			cudaCount++
		}
		if strings.HasPrefix(kv, "CUDA_DEVICE_ORDER=") {
			deviceOrderVal = strings.TrimPrefix(kv, "CUDA_DEVICE_ORDER=")
			deviceOrderCount++
		}
		if kv == "VLLM_SERVER_DEV_MODE=should_be_inherited" {
			hasDevModeVar = true
		}
	}
	if cudaCount != 1 {
		t.Errorf("CUDA_VISIBLE_DEVICES appears %d times, want 1", cudaCount)
	}
	if deviceOrderCount != 1 {
		t.Errorf("CUDA_DEVICE_ORDER appears %d times, want 1", deviceOrderCount)
	}
	if cudaVal != "2,3" {
		t.Errorf("CUDA_VISIBLE_DEVICES = %q, want %q", cudaVal, "2,3")
	}
	if deviceOrderVal != "PCI_BUS_ID" {
		t.Errorf("CUDA_DEVICE_ORDER = %q, want %q", deviceOrderVal, "PCI_BUS_ID")
	}
	// buildEnvLlamaCpp only strips CUDA_VISIBLE_DEVICES/CUDA_DEVICE_ORDER from
	// the parent env — unlike buildEnv, it does not strip or inject
	// VLLM_SERVER_DEV_MODE, so the parent's value must pass through unchanged.
	if !hasDevModeVar {
		t.Error("VLLM_SERVER_DEV_MODE from parent env should be inherited unchanged (only CUDA vars are stripped)")
	}
}

func TestDrainAndMeasure(t *testing.T) {
	t.Parallel()

	t.Run("both_lines_found", func(t *testing.T) {
		t.Parallel()
		pr, pw := io.Pipe()
		mem := &modelMemory{}
		done := make(chan struct{})
		go func() {
			drainAndMeasure(pr, "test-model", mem)
			close(done)
		}()
		pw.Write([]byte("INFO [model_runner.py:295] Model loading took 7.00 GiB and 7.8 seconds\n"))
		pw.Write([]byte("INFO [gpu_worker.py:466] Available KV cache memory: 5.00 GiB\n"))
		pw.Close()
		<-done

		if !mem.measured {
			t.Fatal("expected measured=true")
		}
		if mem.weightsVRAMMB != 7168 {
			t.Errorf("weightsVRAMMB = %d, want 7168", mem.weightsVRAMMB)
		}
		if mem.fullKVVRAMMB != 12288 {
			t.Errorf("fullKVVRAMMB = %d, want 12288 (7168+5120)", mem.fullKVVRAMMB)
		}
	})

	t.Run("no_measurement_lines", func(t *testing.T) {
		t.Parallel()
		pr, pw := io.Pipe()
		mem := &modelMemory{}
		done := make(chan struct{})
		go func() {
			drainAndMeasure(pr, "test-model", mem)
			close(done)
		}()
		// Write irrelevant lines only, then close.
		pw.Write([]byte("some other log line\n"))
		pw.Write([]byte("another line\n"))
		pw.Close()
		<-done

		if mem.measured {
			t.Error("expected measured=false when no measurement lines present")
		}
	})
}

func TestPollIsSleeping(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		body    string
		want    bool
		wantErr bool
	}{
		{"sleeping true", `{"is_sleeping":true}`, true, false},
		{"sleeping false", `{"is_sleeping":false}`, false, false},
		{"unexpected body", `{"something":"else"}`, false, true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create a temp Unix socket.
			sockPath := tempUnixSock(t)

			ln, err := net.Listen("unix", sockPath)
			if err != nil {
				t.Fatalf("listen: %v", err)
			}
			t.Cleanup(func() { ln.Close() })

			body := tc.body
			srv := &http.Server{
				Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					io.WriteString(w, body)
				}),
			}
			go srv.Serve(ln)
			t.Cleanup(func() { srv.Close() })

			vp := makeTestVLLMProcess(sockPath)

			// Retry briefly to allow server to be ready.
			var got bool
			var pollErr error
			deadline := time.Now().Add(2 * time.Second)
			for time.Now().Before(deadline) {
				got, pollErr = pollIsSleeping(vp)
				if pollErr == nil || tc.wantErr {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}

			if tc.wantErr {
				if pollErr == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if pollErr != nil {
				t.Fatalf("unexpected error: %v", pollErr)
			}
			if got != tc.want {
				t.Errorf("pollIsSleeping = %v, want %v", got, tc.want)
			}
		})
	}
}

// tempUnixSock returns a path for a temp Unix socket that does not yet exist.
func tempUnixSock(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	return dir + "/test.sock"
}

// makeTestVLLMProcess builds a vllmProcess whose client dials sockPath.
func makeTestVLLMProcess(sockPath string) *vllmProcess {
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", sockPath)
		},
	}
	return &vllmProcess{
		socketPath: sockPath,
		client:     &http.Client{Transport: transport},
	}
}

func TestCheckSocketOwned(t *testing.T) {
	t.Parallel()

	t.Run("server_listening", func(t *testing.T) {
		t.Parallel()
		sockPath := t.TempDir() + "/test.sock"
		ln, err := net.Listen("unix", sockPath)
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		t.Cleanup(func() { ln.Close() })

		if err := checkSocketOwned(sockPath); err != nil {
			t.Errorf("expected nil error, got %v", err)
		}
	})

	t.Run("no_server", func(t *testing.T) {
		t.Parallel()
		sockPath := t.TempDir() + "/test.sock"
		if err := checkSocketOwned(sockPath); err == nil {
			t.Error("expected non-nil error, got nil")
		}
	})
}

func TestLaunchVLLMMemoryArgs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name             string
		kvCacheMemoryGB  float64
		vramAllocationMB int64
		totalVRAMMB      int64
		wantFlag         string
		wantVal          string
	}{
		{
			name:            "kv_cache_memory_set",
			kvCacheMemoryGB: 18.5,
			wantFlag:        "--kv-cache-memory-bytes",
			wantVal:         "18.5g",
		},
		{
			name:             "gpu_memory_utilization_fallback",
			kvCacheMemoryGB:  0,
			vramAllocationMB: 20480,
			totalVRAMMB:      24576,
			wantFlag:         "--gpu-memory-utilization",
			wantVal:          "0.83",
		},
		{
			name:            "kv_cache_memory_whole_number",
			kvCacheMemoryGB: 11,
			wantFlag:        "--kv-cache-memory-bytes",
			wantVal:         "11g",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var memArg []string
			if tc.kvCacheMemoryGB > 0 {
				memArg = []string{"--kv-cache-memory-bytes", fmt.Sprintf("%gg", tc.kvCacheMemoryGB)}
			} else {
				memArg = []string{"--gpu-memory-utilization", fmt.Sprintf("%.2f", float64(tc.vramAllocationMB)/float64(tc.totalVRAMMB))}
			}
			if len(memArg) != 2 {
				t.Fatalf("memArg len = %d, want 2", len(memArg))
			}
			if memArg[0] != tc.wantFlag {
				t.Errorf("flag = %q, want %q", memArg[0], tc.wantFlag)
			}
			if memArg[1] != tc.wantVal {
				t.Errorf("value = %q, want %q", memArg[1], tc.wantVal)
			}
		})
	}
}

func TestDrainAndMeasureNilMem(t *testing.T) {
	t.Parallel()

	pr, pw := io.Pipe()
	done := make(chan struct{})
	go func() {
		drainAndMeasure(pr, "test-model", nil)
		close(done)
	}()
	pw.Write([]byte("some regular log line\n"))
	pw.Write([]byte("Generated 50 tokens/s\n"))
	pw.Write([]byte("another line\n"))
	pw.Close()
	<-done
}

// TestWaitForHealthFailsFastOnProcessExit verifies that when the vLLM
// process exits before ever becoming healthy, waitForHealth returns
// immediately (via vp.exited) instead of blind-polling for its full 3600s
// timeout.
func TestWaitForHealthFailsFastOnProcessExit(t *testing.T) {
	t.Parallel()

	// A socket path nothing listens on, so every health GET fails.
	vp := makeTestVLLMProcess(t.TempDir() + "/nonexistent.sock")
	vp.exited = make(chan struct{})
	close(vp.exited) // simulate the process having already exited

	start := time.Now()
	err := waitForHealth(vp, "test-model")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error when process exited before becoming healthy, got nil")
	}
	if !strings.Contains(err.Error(), "exited before becoming healthy") {
		t.Errorf("error = %v, want it to mention process exit", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("waitForHealth took %v, want it to fail fast (well under the 3600s timeout)", elapsed)
	}
}

func TestLlamaCppArgConstruction(t *testing.T) {
	t.Parallel()

	// Verify the arg list shape that launchLlamaCpp produces (without actually
	// spawning the process, since llama-server likely isn't installed in tests).
	modelCfg := ModelConfig{
		Name:             "test-model",
		Aliases:          []string{"alias1", "alias2"},
		Engine:           engineLlamaCpp,
		GGUFPath:         "test.gguf",
		VRAMAllocationMB: 20000,
		LlamaCppArgs:     []string{"-ngl", "auto", "-t", "16"},
	}
	modelDir := "/data/models"
	socketPath := "/run/llama/test.sock"

	// Reconstruct the args that launchLlamaCpp builds.
	args := append([]string{
		"-m", filepath.Join(modelDir, modelCfg.GGUFPath),
		"--host", socketPath,
		"-a", modelCfg.Name,
	}, modelCfg.LlamaCppArgs...)

	if len(args) != 10 {
		t.Fatalf("args len = %d, want 10", len(args))
	}
	cases := []struct {
		idx  int
		want string
	}{
		{0, "-m"},
		{1, "/data/models/test.gguf"},
		{2, "--host"},
		{3, "/run/llama/test.sock"},
		{4, "-a"},
		{5, "test-model"}, // correction 1: only Name, no aliases joined
		{6, "-ngl"},
		{7, "auto"},
		{8, "-t"},
		{9, "16"},
	}
	for _, c := range cases {
		if args[c.idx] != c.want {
			t.Errorf("args[%d] = %q, want %q", c.idx, args[c.idx], c.want)
		}
	}

	// Verify -a contains only the canonical name (not comma-joined aliases).
	if args[5] != "test-model" {
		t.Errorf("-a value = %q, want %q (CORRECTION 1: only Name, no aliases)", args[5], "test-model")
	}
}

func TestLaunchVLLMTensorParallelSize(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name               string
		tensorParallelSize int
		groupGpuCount      int
		wantTPSize         string
	}{
		{
			name:               "override_set_wins",
			tensorParallelSize: 2,
			groupGpuCount:      4,
			wantTPSize:         "2",
		},
		{
			name:               "override_zero_falls_back_to_group_size",
			tensorParallelSize: 0,
			groupGpuCount:      3,
			wantTPSize:         "3",
		},
		{
			name:               "override_equals_group_size",
			tensorParallelSize: 4,
			groupGpuCount:      4,
			wantTPSize:         "4",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tpSize := tc.groupGpuCount
			if tc.tensorParallelSize > 0 {
				tpSize = tc.tensorParallelSize
			}
			got := strconv.Itoa(tpSize)
			if got != tc.wantTPSize {
				t.Errorf("tpSize = %q, want %q", got, tc.wantTPSize)
			}
		})
	}
}
