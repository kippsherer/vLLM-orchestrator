package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSanitizeModelName(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in   string
		want string
	}{
		{"meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama_Meta-Llama-3-8B-Instruct"},
		{"mistralai/Mistral-7B-Instruct-v0.3", "mistralai_Mistral-7B-Instruct-v0.3"},
		{"simple", "simple"},
		{"with spaces", "with_spaces"},
		{"a:b@c#d", "a_b_c_d"},
		{"already_clean.name-1", "already_clean.name-1"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.in, func(t *testing.T) {
			t.Parallel()
			if got := sanitizeModelName(tc.in); got != tc.want {
				t.Errorf("sanitizeModelName(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestLoadConfig(t *testing.T) {
	t.Parallel()

	t.Run("valid", func(t *testing.T) {
		t.Parallel()
		yaml := `
listen: ":8000"
vllm_socket_dir: "/tmp/vllm-test-sockets"
queue_depth: 10
ttl_active: 5m
ttl_inactive: 30m
ttl_unused: 60m
gpu_groups:
  - id: "g0"
    gpus: [0]
models:
  - name: "test-model"
`
		f := writeTempYAML(t, yaml)
		cfg, err := loadConfig(f)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if cfg.Listen != ":8000" {
			t.Errorf("Listen = %q, want :8000", cfg.Listen)
		}
		if cfg.QueueDepth != 10 {
			t.Errorf("QueueDepth = %d, want 10", cfg.QueueDepth)
		}
		if cfg.TTLActive != 5*time.Minute {
			t.Errorf("TTLActive = %v, want 5m", cfg.TTLActive)
		}
	})

	t.Run("missing_file", func(t *testing.T) {
		t.Parallel()
		_, err := loadConfig("/nonexistent/path/config.yaml")
		if err == nil {
			t.Fatal("expected error for missing file")
		}
	})

	t.Run("bad_yaml", func(t *testing.T) {
		t.Parallel()
		f := writeTempYAML(t, ":\t\tinvalid: [yaml: here")
		_, err := loadConfig(f)
		if err == nil {
			t.Fatal("expected error for bad YAML")
		}
	})
}

func TestValidateConfig(t *testing.T) {
	t.Parallel()

	socketDir := t.TempDir()

	good := func() *Config {
		return &Config{
			Listen:        ":8000",
			VLLMSocketDir: socketDir,
			QueueDepth:    10,
			TTLActive:     5 * time.Minute,
			TTLInactive:   30 * time.Minute,
			TTLUnused:     60 * time.Minute,
			GPUGroups:     []GPUGroup{{ID: "g0", GPUs: []int{0}}},
			Models:        []ModelConfig{{Name: "m1"}},
		}
	}

	cases := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{"valid", func(*Config) {}, ""},
		{"missing listen", func(c *Config) { c.Listen = "" }, "listen is required"},
		{"missing socket_dir", func(c *Config) { c.VLLMSocketDir = "" }, "vllm_socket_dir is required"},
		{"queue_depth zero", func(c *Config) { c.QueueDepth = 0 }, "queue_depth must be > 0"},
		{"ttl_active zero", func(c *Config) { c.TTLActive = 0 }, "ttl_active must be > 0"},
		{"ttl_inactive zero", func(c *Config) { c.TTLInactive = 0 }, "ttl_inactive must be > 0"},
		{"ttl_unused zero", func(c *Config) { c.TTLUnused = 0 }, "ttl_unused must be > 0"},
		{"ttl_active >= ttl_inactive", func(c *Config) { c.TTLActive = 30 * time.Minute }, "must be <"},
		{"ttl_inactive >= ttl_unused", func(c *Config) { c.TTLInactive = 60 * time.Minute }, "must be <"},
		{"no gpu_groups", func(c *Config) { c.GPUGroups = nil }, "at least one gpu_group"},
		{"no models", func(c *Config) { c.Models = nil }, "at least one model"},
		{"group missing id", func(c *Config) { c.GPUGroups[0].ID = "" }, "gpu_group missing id"},
		{"group no gpus", func(c *Config) { c.GPUGroups[0].GPUs = nil }, "has no gpus"},
		{"duplicate gpu across groups", func(c *Config) {
			c.GPUGroups = []GPUGroup{
				{ID: "g0", GPUs: []int{0, 1}},
				{ID: "g1", GPUs: []int{1, 2}},
			}
		}, "GPU device 1 appears in both"},
		{"model missing name", func(c *Config) { c.Models = []ModelConfig{{Name: ""}} }, "model entry missing name"},
		{"duplicate model name", func(c *Config) {
			c.Models = []ModelConfig{{Name: "m1"}, {Name: "m1"}}
		}, "duplicate model name"},
		{"duplicate alias matches name", func(c *Config) {
			c.Models = []ModelConfig{
				{Name: "m1", Aliases: []string{"m2"}},
				{Name: "m2"},
			}
		}, "duplicate model name"},
		{"duplicate alias across models", func(c *Config) {
			c.Models = []ModelConfig{
				{Name: "m1", Aliases: []string{"alias1"}},
				{Name: "m2", Aliases: []string{"alias1"}},
			}
		}, "duplicate model name/alias"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg := good()
			tc.mutate(cfg)
			err := validateConfig(cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

// writeTempYAML writes content to a temp file and returns its path.
func writeTempYAML(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return filepath.Clean(f.Name())
}
