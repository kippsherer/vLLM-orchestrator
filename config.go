package main

import (
	"fmt"
	"os"
	"regexp"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the top-level structure parsed from the YAML config file.
type Config struct {
	Listen        string        `yaml:"listen"`
	VLLMSocketDir string        `yaml:"vllm_socket_dir"`
	QueueDepth    int           `yaml:"queue_depth"`
	TTLActive     time.Duration `yaml:"ttl_active"`
	TTLInactive   time.Duration `yaml:"ttl_inactive"`
	TTLUnused     time.Duration `yaml:"ttl_unused"`
	GPUGroups     []GPUGroup    `yaml:"gpu_groups"`
	Models        []ModelConfig `yaml:"models"`
}

// GPUGroup declares a set of CUDA device IDs that form a scheduling unit.
type GPUGroup struct {
	ID   string `yaml:"id"`
	GPUs []int  `yaml:"gpus"`
}

// ModelConfig is the per-model entry from the YAML file.
type ModelConfig struct {
	Name             string        `yaml:"name"`
	Aliases          []string      `yaml:"aliases"`
	LoadAtStartup    bool          `yaml:"load_at_startup"`
	GPUGroup         string        `yaml:"gpu_group"`      // when set, pins this model to the named gpu_group
	VRAMAllocationMB int64         `yaml:"vram_allocation"` // authoritative VRAM this model is allowed to consume on the group; used to derive --gpu-memory-utilization
	KVCacheMemoryGB  float64       `yaml:"kv_cache_memory"` // when set, passed as --kv-cache-memory-bytes (GiB, e.g. "10G" binary suffix) and skips --gpu-memory-utilization
	TTLActive        time.Duration `yaml:"ttl_active"`      // overrides global ttl_active when > 0
	TTLInactive      time.Duration `yaml:"ttl_inactive"`    // overrides global ttl_inactive when > 0
	TTLUnused        time.Duration `yaml:"ttl_unused"`      // overrides global ttl_unused when > 0
	VLLMArgs         []string      `yaml:"vllm_args"`
}

// loadConfig reads and parses the YAML file at path.
func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}
	return &cfg, nil
}

// validateConfig checks all abort-on-failure conditions from §13.
// nvidia-smi device ID existence is checked separately in memory.go after the
// nvidia-smi query, because it requires the live measurement results.
func validateConfig(cfg *Config) error {
	if cfg.Listen == "" {
		return fmt.Errorf("config: listen is required")
	}
	if cfg.VLLMSocketDir == "" {
		return fmt.Errorf("config: vllm_socket_dir is required")
	}
	if cfg.QueueDepth <= 0 {
		return fmt.Errorf("config: queue_depth must be > 0")
	}
	if cfg.TTLActive <= 0 {
		return fmt.Errorf("config: ttl_active must be > 0")
	}
	if cfg.TTLInactive <= 0 {
		return fmt.Errorf("config: ttl_inactive must be > 0")
	}
	if cfg.TTLUnused <= 0 {
		return fmt.Errorf("config: ttl_unused must be > 0")
	}
	if cfg.TTLActive >= cfg.TTLInactive {
		return fmt.Errorf("config: ttl_active (%v) must be < ttl_inactive (%v)", cfg.TTLActive, cfg.TTLInactive)
	}
	if cfg.TTLInactive >= cfg.TTLUnused {
		return fmt.Errorf("config: ttl_inactive (%v) must be < ttl_unused (%v)", cfg.TTLInactive, cfg.TTLUnused)
	}
	if len(cfg.GPUGroups) == 0 {
		return fmt.Errorf("config: at least one gpu_group is required")
	}
	if len(cfg.Models) == 0 {
		return fmt.Errorf("config: at least one model is required")
	}

	// Duplicate GPU device IDs across groups.
	seen := make(map[int]string)
	for _, g := range cfg.GPUGroups {
		if g.ID == "" {
			return fmt.Errorf("config: gpu_group missing id")
		}
		if len(g.GPUs) == 0 {
			return fmt.Errorf("config: gpu_group %q has no gpus", g.ID)
		}
		for _, dev := range g.GPUs {
			if prior, ok := seen[dev]; ok {
				return fmt.Errorf("config: GPU device %d appears in both group %q and %q", dev, prior, g.ID)
			}
			seen[dev] = g.ID
		}
	}

	// Duplicate model names or aliases.
	groupIDs := make(map[string]struct{})
	for _, g := range cfg.GPUGroups {
		groupIDs[g.ID] = struct{}{}
	}
	names := make(map[string]struct{})
	for _, m := range cfg.Models {
		if m.Name == "" {
			return fmt.Errorf("config: model entry missing name")
		}
		if _, dup := names[m.Name]; dup {
			return fmt.Errorf("config: duplicate model name %q", m.Name)
		}
		names[m.Name] = struct{}{}
		for _, a := range m.Aliases {
			if _, dup := names[a]; dup {
				return fmt.Errorf("config: duplicate model name/alias %q", a)
			}
			names[a] = struct{}{}
		}
		// Per-model gpu_group pin must reference a declared group.
		if m.GPUGroup != "" {
			if _, ok := groupIDs[m.GPUGroup]; !ok {
				return fmt.Errorf("config: model %q: gpu_group %q not found in gpu_groups", m.Name, m.GPUGroup)
			}
		}
		// Per-model TTL overrides must be self-consistent when set.
		eff := func(modelVal, globalVal time.Duration) time.Duration {
			if modelVal > 0 {
				return modelVal
			}
			return globalVal
		}
		effActive := eff(m.TTLActive, cfg.TTLActive)
		effInactive := eff(m.TTLInactive, cfg.TTLInactive)
		effUnused := eff(m.TTLUnused, cfg.TTLUnused)
		if effActive >= effInactive {
			return fmt.Errorf("config: model %q: effective ttl_active (%v) must be < ttl_inactive (%v)", m.Name, effActive, effInactive)
		}
		if effInactive >= effUnused {
			return fmt.Errorf("config: model %q: effective ttl_inactive (%v) must be < ttl_unused (%v)", m.Name, effInactive, effUnused)
		}
	}

	// vllm_socket_dir writable check.
	if err := os.MkdirAll(cfg.VLLMSocketDir, 0700); err != nil {
		return fmt.Errorf("config: vllm_socket_dir %q not writable: %w", cfg.VLLMSocketDir, err)
	}
	f, err := os.CreateTemp(cfg.VLLMSocketDir, ".writetest")
	if err != nil {
		return fmt.Errorf("config: vllm_socket_dir %q not writable: %w", cfg.VLLMSocketDir, err)
	}
	f.Close()
	os.Remove(f.Name())

	return nil
}

// sanitizeModelName replaces characters outside [a-zA-Z0-9_.-] with _.
var sanitizeRe = regexp.MustCompile(`[^a-zA-Z0-9_.\-]`)

func sanitizeModelName(name string) string {
	return sanitizeRe.ReplaceAllString(name, "_")
}
