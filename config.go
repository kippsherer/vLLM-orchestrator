package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	engineVLLM     = "vllm"
	engineLlamaCpp = "llama_cpp"
)

// Config is the top-level structure parsed from the YAML config file.
type Config struct {
	Listen            string        `yaml:"listen"`
	VLLMSocketDir     string        `yaml:"vllm_socket_dir"`
	LlamaCppSocketDir string        `yaml:"llama_cpp_socket_dir"`
	LlamaCppModelDir  string        `yaml:"llama_cpp_model_dir"`
	QueueDepth        int           `yaml:"queue_depth"`
	TTLActive         time.Duration `yaml:"ttl_active"`
	TTLInactive       time.Duration `yaml:"ttl_inactive"`
	TTLUnused         time.Duration `yaml:"ttl_unused"`
	GPUGroups         []GPUGroup    `yaml:"gpu_groups"`
	Models            []ModelConfig `yaml:"models"`
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
	Engine           string        `yaml:"engine"` // "" or "vllm" (default) | "llama_cpp"
	LoadAtStartup    bool          `yaml:"load_at_startup"`
	GPUGroup         string        `yaml:"gpu_group"`       // when set, pins this model to the named gpu_group
	VRAMAllocationMB int64         `yaml:"vram_allocation"` // authoritative VRAM this model is allowed to consume on the group
	KVCacheMemoryGB  float64       `yaml:"kv_cache_memory"` // vLLM only; passed as --kv-cache-memory-bytes (GiB)
	TTLActive        time.Duration `yaml:"ttl_active"`      // overrides global ttl_active when > 0
	TTLInactive      time.Duration `yaml:"ttl_inactive"`    // overrides global ttl_inactive when > 0
	TTLUnused        time.Duration `yaml:"ttl_unused"`      // overrides global ttl_unused when > 0
	VLLMArgs         []string      `yaml:"vllm_args"`       // vLLM only
	GGUFPath         string        `yaml:"gguf_path"`       // llama_cpp only; joined with llama_cpp_model_dir
	LlamaCppArgs     []string      `yaml:"llama_cpp_args"`  // llama_cpp only; raw passthrough
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

	// Duplicate model names or aliases per-model validation.
	groupIDs := make(map[string]struct{})
	for _, g := range cfg.GPUGroups {
		groupIDs[g.ID] = struct{}{}
	}
	names := make(map[string]struct{})
	hasVLLM := false
	hasLlamaCpp := false
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
		// Engine validation.
		switch m.Engine {
		case "", engineVLLM:
			hasVLLM = true
		case engineLlamaCpp:
			hasLlamaCpp = true
		default:
			return fmt.Errorf("config: model %q: engine must be empty, %q, or %q; got %q", m.Name, engineVLLM, engineLlamaCpp, m.Engine)
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
		// Engine-specific field validation.
		if m.Engine == engineLlamaCpp {
			if m.GGUFPath == "" {
				return fmt.Errorf("config: model %q: gguf_path is required for llama_cpp engine", m.Name)
			}
			if m.KVCacheMemoryGB != 0 {
				return fmt.Errorf("config: model %q: kv_cache_memory must not be set for llama_cpp engine", m.Name)
			}
			if len(m.VLLMArgs) > 0 {
				return fmt.Errorf("config: model %q: vllm_args must not be set for llama_cpp engine", m.Name)
			}
		} else {
			if m.GGUFPath != "" {
				return fmt.Errorf("config: model %q: gguf_path must not be set for vllm engine", m.Name)
			}
			if len(m.LlamaCppArgs) > 0 {
				return fmt.Errorf("config: model %q: llama_cpp_args must not be set for vllm engine", m.Name)
			}
		}
	}

	// Conditional socket dir requirements.
	if hasVLLM {
		if cfg.VLLMSocketDir == "" {
			return fmt.Errorf("config: vllm_socket_dir is required when models use vllm engine")
		}
		if err := os.MkdirAll(cfg.VLLMSocketDir, 0700); err != nil {
			return fmt.Errorf("config: vllm_socket_dir %q not writable: %w", cfg.VLLMSocketDir, err)
		}
		f, err := os.CreateTemp(cfg.VLLMSocketDir, ".writetest")
		if err != nil {
			return fmt.Errorf("config: vllm_socket_dir %q not writable: %w", cfg.VLLMSocketDir, err)
		}
		f.Close()
		os.Remove(f.Name())
	}
	if hasLlamaCpp {
		if cfg.LlamaCppSocketDir == "" {
			return fmt.Errorf("config: llama_cpp_socket_dir is required when models use llama_cpp engine")
		}
		if cfg.LlamaCppModelDir == "" {
			return fmt.Errorf("config: llama_cpp_model_dir is required when models use llama_cpp engine")
		}
		if err := os.MkdirAll(cfg.LlamaCppSocketDir, 0700); err != nil {
			return fmt.Errorf("config: llama_cpp_socket_dir %q not writable: %w", cfg.LlamaCppSocketDir, err)
		}
		f, err := os.CreateTemp(cfg.LlamaCppSocketDir, ".writetest")
		if err != nil {
			return fmt.Errorf("config: llama_cpp_socket_dir %q not writable: %w", cfg.LlamaCppSocketDir, err)
		}
		f.Close()
		os.Remove(f.Name())
	}

	// File existence check for llama_cpp models (after socket dir checks ensure dirs are set).
	for _, m := range cfg.Models {
		if m.Engine == engineLlamaCpp {
			fullPath := filepath.Join(cfg.LlamaCppModelDir, m.GGUFPath)
			info, err := os.Stat(fullPath)
			if err != nil {
				return fmt.Errorf("config: model %q: GGUF file %s not found: %w", m.Name, fullPath, err)
			}
			if info.IsDir() {
				return fmt.Errorf("config: model %q: GGUF path %s is a directory, not a file", m.Name, fullPath)
			}
		}
	}

	return nil
}

// sanitizeModelName replaces characters outside [a-zA-Z0-9_.-] with _.
var sanitizeRe = regexp.MustCompile(`[^a-zA-Z0-9_.\-]`)

func sanitizeModelName(name string) string {
	return sanitizeRe.ReplaceAllString(name, "_")
}
