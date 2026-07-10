package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

var verbose bool

func main() {
	configPath := flag.String("config", "", "path to YAML config file")
	flag.BoolVar(&verbose, "verbose", false, "enable verbose logging")
	flag.Parse()

	if *configPath == "" {
		*configPath = os.Getenv("VLLM_ORCH_CONFIG")
	}
	if *configPath == "" {
		log.Fatal("config path required: use --config or VLLM_ORCH_CONFIG env var")
	}

	cfg, err := loadConfig(*configPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	if err := validateConfig(cfg); err != nil {
		log.Fatalf("validate config: %v", err)
	}

	ms, err := initMemory(cfg)
	if err != nil {
		log.Fatalf("init memory: %v", err)
	}

	// Correct VRAM accounting immediately for any pre-existing GPU usage
	// from processes outside this orchestrator instance.
	refreshMemory(ms)

	// Remove stale socket files from a prior crash.
	entries, err := os.ReadDir(cfg.VLLMSocketDir)
	if err == nil {
		for _, e := range entries {
			if !strings.HasSuffix(e.Name(), ".sock") {
				continue
			}
			path := filepath.Join(cfg.VLLMSocketDir, e.Name())
			if checkSocketOwned(path) != nil {
				os.Remove(path)
				log.Printf("startup: removed stale socket %s", path)
			}
		}
	}

	o := newOrchestrator(cfg, ms)
	o.startTTLLoops()
	startPeriodicMemoryRefresh(ms)

	// Load startup models.
	for _, me := range o.models {
		if me.cfg.LoadAtStartup {
			go func(m *modelEntry) {
				o.loadModel(m)
			}(me)
		}
	}

	srv := &http.Server{
		Addr:    cfg.Listen,
		Handler: http.HandlerFunc(o.serveHTTP),
	}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		log.Printf("vLLM orchestrator listening on %s", cfg.Listen)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-quit
	log.Println("shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("server shutdown error: %v", err)
	}

	// Terminate all running vLLM subprocesses.
	for _, me := range o.models {
		me.mu.Lock()
		proc := me.proc
		name := me.cfg.Name
		me.proc = nil
		me.mu.Unlock()
		if proc != nil {
			killProcess(proc, name)
		}
	}
	log.Println("shutdown complete")
}
