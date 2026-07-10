package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"strings"
	"time"
)

// blockedPaths return 403 to any external caller.
var blockedPaths = map[string]bool{
	"/sleep":       true,
	"/wake_up":     true,
	"/is_sleeping": true,
}

// serveHTTP is the top-level HTTP handler wired to the server's mux.
func (o *orchestrator) serveHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	// Block management endpoints.
	if blockedPaths[path] {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	// Orchestrator-owned endpoints.
	switch path {
	case "/health", "/ping":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ok"}`)
		return
	case "/version":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"version":"%s"}`, buildVersion)
		return
	case "/v1/models":
		if r.Method == http.MethodGet {
			o.serveModels(w, r)
			return
		}
	}

	// Routes needing ?model= routing (no body peek).
	if path == "/metrics" || path == "/docs" || path == "/redoc" || path == "/openapi.json" {
		if path == "/docs" || path == "/redoc" || path == "/openapi.json" {
			o.forwardToAnyActive(w, r)
			return
		}
		// /metrics: require ?model= param.
		modelParam := r.URL.Query().Get("model")
		if modelParam == "" {
			http.Error(w, "bad request: ?model= required for /metrics", http.StatusBadRequest)
			return
		}
		me := o.resolve(modelParam)
		if me == nil {
			http.Error(w, "model not found", http.StatusNotFound)
			return
		}
		o.forwardDirect(w, r, me)
		return
	}

	// WebSocket upgrade.
	if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		o.serveWebSocket(w, r)
		return
	}

	// All remaining routes: body peek for model field.
	if r.Method == http.MethodGet || r.Method == http.MethodHead || r.Method == http.MethodDelete || r.Method == http.MethodOptions {
		// No body on safe/idempotent methods; route by ?model= if present.
		modelParam := r.URL.Query().Get("model")
		if modelParam == "" {
			http.Error(w, "bad request: ?model= required", http.StatusBadRequest)
			return
		}
		me := o.resolve(modelParam)
		if me == nil {
			http.Error(w, "model not found", http.StatusNotFound)
			return
		}
		o.routeRequest(w, r, me)
		return
	}

	// POST/PUT/PATCH: peek body for "model" field.
	modelName, buf, err := peekModel(r)
	if err != nil || modelName == "" {
		http.Error(w, "bad request: could not extract model field", http.StatusBadRequest)
		return
	}
	me := o.resolve(modelName)
	if me == nil {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	// Rewrite alias to canonical name so vLLM recognises it.
	if modelName != me.cfg.Name {
		buf = rewriteModelField(buf, me.cfg.Name)
	}
	r.Body = io.NopCloser(bytes.NewReader(buf))
	r.ContentLength = int64(len(buf))

	o.routeRequest(w, r, me)
}

// routeRequest drives the state machine / queuing and then proxies the request.
func (o *orchestrator) routeRequest(w http.ResponseWriter, r *http.Request, me *modelEntry) {
	errFlag := false
	rp := requestPair{w: w, r: r, done: make(chan struct{}), err: &errFlag}
	o.handleRequest(me, rp)

	select {
	case <-rp.done:
	case <-r.Context().Done():
		return
	}

	// errFlag means 503 was already written by drainQueueWith503; do not forward.
	if errFlag {
		return
	}

	o.forwardDirect(w, r, me)
	o.completeRequest(me)
}

// forwardDirect creates a per-request ReverseProxy to me's Unix socket and serves.
func (o *orchestrator) forwardDirect(w http.ResponseWriter, r *http.Request, me *modelEntry) {
	socketPath := me.socketPath
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		},
	}
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = "http"
			req.URL.Host = "vllm"
		},
		Transport:     transport,
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			logVerbose("[proxy] %s upstream error: %v", me.cfg.Name, err)
			http.Error(w, "bad gateway", http.StatusBadGateway)
		},
	}
	proxy.ServeHTTP(w, r)
}

// forwardToAnyActive proxies to the first ACTIVE model instance, or 503.
func (o *orchestrator) forwardToAnyActive(w http.ResponseWriter, r *http.Request) {
	for _, me := range o.models {
		me.mu.Lock()
		s := me.state
		me.mu.Unlock()
		if s == stateActive {
			o.forwardDirect(w, r, me)
			return
		}
	}
	http.Error(w, "service unavailable: no vLLM instance currently active", http.StatusServiceUnavailable)
}

// peekModel reads r.Body entirely and extracts the top-level "model" string
// with a minimal byte-scan (no encoding/json unmarshal).
func peekModel(r *http.Request) (string, []byte, error) {
	if r.Body == nil {
		return "", nil, fmt.Errorf("no body")
	}
	buf, err := io.ReadAll(r.Body)
	if err != nil {
		return "", nil, err
	}
	model := extractModelField(buf)
	return model, buf, nil
}

// rewriteModelField returns a copy of buf with the top-level "model" string
// value replaced by canonicalName. All other bytes are preserved verbatim.
func rewriteModelField(buf []byte, canonicalName string) []byte {
	key := []byte(`"model"`)
	i := bytes.Index(buf, key)
	if i < 0 {
		return buf
	}
	i += len(key)
	// Skip whitespace and colon.
	for i < len(buf) && (buf[i] == ' ' || buf[i] == '\t' || buf[i] == '\n' || buf[i] == '\r') {
		i++
	}
	if i >= len(buf) || buf[i] != ':' {
		return buf
	}
	i++
	for i < len(buf) && (buf[i] == ' ' || buf[i] == '\t' || buf[i] == '\n' || buf[i] == '\r') {
		i++
	}
	if i >= len(buf) || buf[i] != '"' {
		return buf
	}
	// valueStart is the opening quote of the current model value.
	valueStart := i
	i++ // skip opening quote
	for i < len(buf) {
		c := buf[i]
		if c == '"' {
			break
		}
		if c == '\\' {
			i++ // skip escaped char
		}
		i++
	}
	if i >= len(buf) {
		return buf
	}
	// valueEnd is the closing quote (inclusive).
	valueEnd := i

	// Build replacement: everything before valueStart + quoted canonical name + everything after valueEnd.
	quoted := make([]byte, 0, len(canonicalName)+2)
	quoted = append(quoted, '"')
	quoted = append(quoted, []byte(canonicalName)...)
	quoted = append(quoted, '"')

	out := make([]byte, 0, len(buf)-int(valueEnd-valueStart)+len(quoted))
	out = append(out, buf[:valueStart]...)
	out = append(out, quoted...)
	out = append(out, buf[valueEnd+1:]...)
	return out
}

// extractModelField performs a minimal scan of a JSON byte slice to find the
// top-level "model" string value without using encoding/json (which would
// drop unknown fields on re-serialization).
func extractModelField(data []byte) string {
	// Find `"model"` key at top level (depth 0 after the opening `{`).
	key := []byte(`"model"`)
	i := bytes.Index(data, key)
	if i < 0 {
		return ""
	}
	i += len(key)
	// Skip whitespace and colon.
	for i < len(data) && (data[i] == ' ' || data[i] == '\t' || data[i] == '\n' || data[i] == '\r') {
		i++
	}
	if i >= len(data) || data[i] != ':' {
		return ""
	}
	i++
	for i < len(data) && (data[i] == ' ' || data[i] == '\t' || data[i] == '\n' || data[i] == '\r') {
		i++
	}
	if i >= len(data) || data[i] != '"' {
		return ""
	}
	i++ // skip opening quote
	var sb strings.Builder
	for i < len(data) {
		c := data[i]
		if c == '"' {
			break
		}
		if c == '\\' && i+1 < len(data) {
			i++
			sb.WriteByte(data[i])
		} else {
			sb.WriteByte(c)
		}
		i++
	}
	return sb.String()
}

// serveModels implements GET /v1/models: aggregates from running instances
// and synthesises stubs for UNLOADED/SLEEP models.
func (o *orchestrator) serveModels(w http.ResponseWriter, r *http.Request) {
	type modelEntry_ struct {
		ID                string `json:"id"`
		Object            string `json:"object"`
		Created           int64  `json:"created"`
		OwnedBy           string `json:"owned_by"`
		OrchestratorState string `json:"orchestrator_state"`
		EstimatedVRAMMB   int64  `json:"estimated_vram_mb,omitempty"`
	}
	type modelsResp struct {
		Object string        `json:"object"`
		Data   []interface{} `json:"data"`
	}

	created := time.Now().Unix()
	var data []interface{}

	for _, me := range o.models {
		me.mu.Lock()
		state := me.state
		mem := me.mem
		groupIdx := me.assignedGroupIdx
		proc := me.proc
		me.mu.Unlock()

		stateStr := state.String()
		if state == stateLoading {
			stateStr = "loading"
		}

		// For ACTIVE/LOADING/SLEEP: try to fetch live data from the instance.
		if (state == stateActive || state == stateSleep1 || state == stateSleep2) && proc != nil {
			liveData := fetchLiveModels(proc, me.cfg.Name)
			if liveData != nil {
				// Inject orchestrator_state into each entry via raw JSON merge.
				for _, entry := range liveData {
					data = append(data, injectState(entry, stateStr))
				}
				continue
			}
		}

		// Synthesize stub.
		var estimatedVRAM int64
		if mem.measured {
			estimatedVRAM = mem.fullKVVRAMMB
		} else if groupIdx >= 0 {
			o.ms.mu.RLock()
			estimatedVRAM = int64(float64(o.ms.groups[groupIdx].measuredTotalVRAMMB) * 0.85)
			o.ms.mu.RUnlock()
		} else {
			// UNLOADED and never launched: use smallest group as reference.
			o.ms.mu.RLock()
			if len(o.ms.groups) > 0 {
				smallest := o.ms.groups[0]
				for _, gs := range o.ms.groups[1:] {
					if gs.measuredTotalVRAMMB < smallest.measuredTotalVRAMMB {
						smallest = gs
					}
				}
				estimatedVRAM = int64(float64(smallest.measuredTotalVRAMMB) * 0.85)
			}
			o.ms.mu.RUnlock()
		}

		data = append(data, modelEntry_{
			ID:                me.cfg.Name,
			Object:            "model",
			Created:           created,
			OwnedBy:           "vllm-orchestrator",
			OrchestratorState: stateStr,
			EstimatedVRAMMB:   estimatedVRAM,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp{Object: "list", Data: data})
}

// fetchLiveModels calls GET /v1/models on the vLLM instance and returns the
// raw data array, or nil on error.
func fetchLiveModels(proc *vllmProcess, modelName string) []json.RawMessage {
	resp, err := proc.client.Get("http://vllm/v1/models")
	if err != nil {
		logVerbose("[proxy] fetchLiveModels %s: %v", modelName, err)
		return nil
	}
	defer resp.Body.Close()
	var out struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		logVerbose("[proxy] fetchLiveModels %s decode: %v", modelName, err)
		return nil
	}
	return out.Data
}

// injectState merges orchestrator_state into a raw JSON object.
func injectState(raw json.RawMessage, state string) json.RawMessage {
	// Trim trailing `}` and append the new field.
	trimmed := bytes.TrimRight(bytes.TrimSpace(raw), "}")
	extra := fmt.Sprintf(`,"orchestrator_state":%q}`, state)
	return json.RawMessage(append(trimmed, []byte(extra)...))
}

// serveWebSocket tunnels a WebSocket upgrade request to the vLLM Unix socket.
func (o *orchestrator) serveWebSocket(w http.ResponseWriter, r *http.Request) {
	modelName, buf, err := peekModel(r)
	if err != nil || modelName == "" {
		http.Error(w, "bad request: could not extract model field", http.StatusBadRequest)
		return
	}
	me := o.resolve(modelName)
	if me == nil {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	if modelName != me.cfg.Name {
		buf = rewriteModelField(buf, me.cfg.Name)
	}
	r.Body = io.NopCloser(bytes.NewReader(buf))
	r.ContentLength = int64(len(buf))

	errFlag2 := false
	rp := requestPair{w: w, r: r, done: make(chan struct{}), err: &errFlag2}
	o.handleRequest(me, rp)
	select {
	case <-rp.done:
	case <-r.Context().Done():
		return
	}
	if errFlag2 {
		return
	}
	defer o.completeRequest(me)

	// Dial upstream Unix socket.
	upstream, err := net.Dial("unix", me.socketPath)
	if err != nil {
		http.Error(w, "bad gateway", http.StatusBadGateway)
		return
	}
	defer upstream.Close()

	// Hijack the client connection.
	hj, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "websocket not supported", http.StatusInternalServerError)
		return
	}
	clientConn, _, err := hj.Hijack()
	if err != nil {
		http.Error(w, "hijack failed", http.StatusInternalServerError)
		return
	}
	defer clientConn.Close()

	// Write the original HTTP upgrade request to upstream.
	if err := r.Write(upstream); err != nil {
		return
	}

	// Bidirectional copy.
	done := make(chan struct{})
	go func() {
		io.Copy(upstream, clientConn)
		close(done)
	}()
	io.Copy(clientConn, upstream)
	<-done
}

// buildVersion is set at link time; default to "dev".
var buildVersion = "dev"
