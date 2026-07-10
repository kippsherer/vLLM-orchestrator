package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"os"
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
	t.Cleanup(func() {
		os.Unsetenv("CUDA_VISIBLE_DEVICES")
		os.Unsetenv("VLLM_SERVER_DEV_MODE")
		os.Unsetenv("OMP_NUM_THREADS")
	})

	result := buildEnv("0,1")

	var cudaVal, devModeVal, ompVal string
	cudaCount, devModeCount, ompCount := 0, 0, 0
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
	if cudaVal != "0,1" {
		t.Errorf("CUDA_VISIBLE_DEVICES = %q, want %q", cudaVal, "0,1")
	}
	if devModeVal != "1" {
		t.Errorf("VLLM_SERVER_DEV_MODE = %q, want %q", devModeVal, "1")
	}
	if ompVal != "1" {
		t.Errorf("OMP_NUM_THREADS = %q, want %q (parent value must be overridden)", ompVal, "1")
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
