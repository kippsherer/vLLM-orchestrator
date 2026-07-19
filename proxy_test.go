package main

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestExtractModelField(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"normal", `{"model":"gpt-4","prompt":"hi"}`, "gpt-4"},
		{"whitespace around colon", `{"model" : "llama3"}`, "llama3"},
		{"escaped quote in value", `{"model":"name\"with\"quotes"}`, `name"with"quotes`},
		{"model key absent", `{"prompt":"hi"}`, ""},
		{"empty body", `{}`, ""},
		{"empty input", ``, ""},
		{"value not a string", `{"model":123}`, ""},
		{"model in nested object", `{"data":{"model":"nested"},"model":"top"}`, "top"},
		{"model as array element before real key", `{"input":["file","function","package","constant","model","type","struct","variable","interface","project"],"model":"embedding","encoding_format":"float"}`, "embedding"},
		{"json escape sequences decoded correctly", `{"model":"a\nb\u002Fc"}`, "a\nb/c"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := extractModelField([]byte(tc.input))
			if got != tc.want {
				t.Errorf("extractModelField(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestServeHTTPOrchestratorOwned(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	cases := []struct {
		method     string
		path       string
		wantStatus int
		wantBody   string
	}{
		{http.MethodGet, "/health", http.StatusOK, `{"status":"ok"}`},
		{http.MethodGet, "/ping", http.StatusOK, `{"status":"ok"}`},
		{http.MethodGet, "/version", http.StatusOK, `"version"`},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.method+tc.path, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(tc.method, tc.path, nil)
			rec := httptest.NewRecorder()
			o.serveHTTP(rec, req)
			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tc.wantStatus)
			}
			if !strings.Contains(rec.Body.String(), tc.wantBody) {
				t.Errorf("body %q does not contain %q", rec.Body.String(), tc.wantBody)
			}
		})
	}
}

func TestServeHTTPBlocked(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	for _, path := range []string{"/sleep", "/wake_up", "/is_sleeping"} {
		path := path
		t.Run(path, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, path, nil)
			rec := httptest.NewRecorder()
			o.serveHTTP(rec, req)
			if rec.Code != http.StatusForbidden {
				t.Errorf("%s: status = %d, want 403", path, rec.Code)
			}
		})
	}
}

func TestServeHTTPMetrics(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	t.Run("missing_model_param", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("unknown_model_param", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/metrics?model=nonexistent", nil)
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Errorf("status = %d, want 404", rec.Code)
		}
	})
}

func TestServeHTTPBodyPeek(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	t.Run("no_body", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("body_missing_model_field", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"messages":[]}`))
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("unknown_model", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"no-such-model","messages":[]}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Errorf("status = %d, want 404", rec.Code)
		}
	})
}

func TestServeHTTPModels(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	o.serveHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "model-a") {
		t.Errorf("body %q missing model-a", body)
	}
	if !strings.Contains(body, "model-b") {
		t.Errorf("body %q missing model-b", body)
	}
	if !strings.Contains(body, "orchestrator_state") {
		t.Errorf("body %q missing orchestrator_state", body)
	}
	if !strings.Contains(body, "unloaded") {
		t.Errorf("body %q missing unloaded state", body)
	}
}

func TestInjectState(t *testing.T) {
	t.Parallel()
	raw := []byte(`{"id":"m","object":"model"}`)
	got := injectState(raw, "active")
	s := string(got)
	if !strings.Contains(s, `"orchestrator_state":"active"`) {
		t.Errorf("injectState result %q missing expected field", s)
	}
}

func TestForwardToAnyActiveNoInstances(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)
	// All models are UNLOADED (default).
	req := httptest.NewRequest(http.MethodGet, "/docs", nil)
	rec := httptest.NewRecorder()
	o.forwardToAnyActive(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", rec.Code)
	}
}

// peekModelHelper builds a minimal http.Request for peekModel.
func peekModelHelper(body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
	return req
}

func TestPeekModel(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		body      string
		wantModel string
		wantErr   bool
	}{
		{"normal", `{"model":"llama3","stream":true}`, "llama3", false},
		{"no model field", `{"stream":true}`, "", false},
		{"nil body", "", "", true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var req *http.Request
			if tc.body == "" && tc.wantErr {
				req = &http.Request{} // nil Body
			} else {
				req = peekModelHelper(tc.body)
			}
			model, buf, err := peekModel(req)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if model != tc.wantModel {
				t.Errorf("model = %q, want %q", model, tc.wantModel)
			}
			if tc.wantModel != "" && len(buf) == 0 {
				t.Error("buf should not be empty when model found")
			}
			_ = io.NopCloser // satisfy import
		})
	}
}

func TestRewriteModelField(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		input     string
		canonical string
		want      string
	}{
		{
			name:      "alias replaced",
			input:     `{"model":"llama3","stream":true}`,
			canonical: "meta-llama/Meta-Llama-3-8B-Instruct",
			want:      `{"model":"meta-llama/Meta-Llama-3-8B-Instruct","stream":true}`,
		},
		{
			name:      "already canonical no change",
			input:     `{"model":"meta-llama/Meta-Llama-3-8B-Instruct","stream":true}`,
			canonical: "meta-llama/Meta-Llama-3-8B-Instruct",
			want:      `{"model":"meta-llama/Meta-Llama-3-8B-Instruct","stream":true}`,
		},
		{
			name:      "model key absent returns unchanged",
			input:     `{"stream":true}`,
			canonical: "canonical",
			want:      `{"stream":true}`,
		},
		{
			name:      "other fields preserved",
			input:     `{"model":"alias","messages":[{"role":"user","content":"hi"}],"temperature":0.7}`,
			canonical: "Qwen/Qwen3-14B",
			want:      `{"model":"Qwen/Qwen3-14B","messages":[{"role":"user","content":"hi"}],"temperature":0.7}`,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := rewriteModelField([]byte(tc.input), tc.canonical)
			if string(got) != tc.want {
				t.Errorf("rewriteModelField(%q, %q)\n got  %q\n want %q", tc.input, tc.canonical, got, tc.want)
			}
		})
	}
}

func TestServeHTTPDocsRoutes(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	routes := []string{"/docs", "/redoc", "/openapi.json"}
	for _, route := range routes {
		route := route
		t.Run(route, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodGet, route, nil)
			rec := httptest.NewRecorder()
			o.serveHTTP(rec, req)
			if rec.Code != http.StatusServiceUnavailable {
				t.Errorf("%s: status = %d, want 503", route, rec.Code)
			}
		})
	}
}

func TestServeHTTPGetMethodRequiresModel(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	t.Run("no_model_param", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/v1/completions", nil)
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("unknown_model", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/v1/completions?model=unknown", nil)
		rec := httptest.NewRecorder()
		o.serveHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Errorf("status = %d, want 404", rec.Code)
		}
	})
}

func TestPeekAndResolve(t *testing.T) {
	t.Parallel()
	o := makeTestOrchestrator(t)

	t.Run("nil_body", func(t *testing.T) {
		t.Parallel()
		req := &http.Request{Body: nil}
		rec := httptest.NewRecorder()
		_, _, ok := o.peekAndResolve(rec, req)
		if ok {
			t.Error("expected ok=false for nil body")
		}
		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("missing_model_field", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"messages":[]}`))
		rec := httptest.NewRecorder()
		_, _, ok := o.peekAndResolve(rec, req)
		if ok {
			t.Error("expected ok=false for missing model field")
		}
		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("unknown_model", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"model":"no-such-model"}`))
		rec := httptest.NewRecorder()
		_, _, ok := o.peekAndResolve(rec, req)
		if ok {
			t.Error("expected ok=false for unknown model")
		}
		if rec.Code != http.StatusNotFound {
			t.Errorf("status = %d, want 404", rec.Code)
		}
	})

	t.Run("known_canonical", func(t *testing.T) {
		t.Parallel()
		body := `{"model":"model-a","prompt":"hi"}`
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
		rec := httptest.NewRecorder()
		me, buf, ok := o.peekAndResolve(rec, req)
		if !ok {
			t.Fatal("expected ok=true for known canonical model")
		}
		if me == nil {
			t.Fatal("expected non-nil modelEntry")
		}
		if me.cfg.Name != "model-a" {
			t.Errorf("model name = %q, want model-a", me.cfg.Name)
		}
		if string(buf) != body {
			t.Errorf("buf = %q, want %q (unchanged)", string(buf), body)
		}
	})

	t.Run("alias_model", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"model":"alias-a","prompt":"hi"}`))
		rec := httptest.NewRecorder()
		me, buf, ok := o.peekAndResolve(rec, req)
		if !ok {
			t.Fatal("expected ok=true for alias model")
		}
		if me == nil {
			t.Fatal("expected non-nil modelEntry")
		}
		if me.cfg.Name != "model-a" {
			t.Errorf("model name = %q, want model-a", me.cfg.Name)
		}
		if !strings.Contains(string(buf), `"model":"model-a"`) {
			t.Errorf("buf %q does not contain rewritten canonical name", string(buf))
		}
	})
}

func TestServeModelsVRAMBranches(t *testing.T) {
	t.Parallel()

	t.Run("allocated_vram_mb_from_config", func(t *testing.T) {
		t.Parallel()
		o := makeTestOrchestrator(t)
		o.models[0].cfg.VRAMAllocationMB = 19661

		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		rec := httptest.NewRecorder()
		o.serveModels(rec, req)
		body := rec.Body.String()
		if !strings.Contains(body, `"allocated_vram_mb":19661`) {
			t.Errorf("body %q missing allocated_vram_mb:19661", body)
		}
	})
}

func TestFetchLiveModels(t *testing.T) {
	t.Parallel()

	t.Run("success", func(t *testing.T) {
		t.Parallel()
		sockPath := t.TempDir() + "/test.sock"
		ln, err := net.Listen("unix", sockPath)
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		t.Cleanup(func() { ln.Close() })

		srv := &http.Server{
			Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/models" {
					w.Header().Set("Content-Type", "application/json")
					io.WriteString(w, `{"object":"list","data":[{"id":"m","object":"model"}]}`)
					return
				}
				http.NotFound(w, r)
			}),
		}
		go srv.Serve(ln)
		t.Cleanup(func() { srv.Close() })

		proc := makeTestVLLMProcess(sockPath)

		var result []json.RawMessage
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			result = fetchLiveModels(proc, "m")
			if result != nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}

		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if len(result) != 1 {
			t.Fatalf("got %d entries, want 1", len(result))
		}
	})

	t.Run("malformed_json", func(t *testing.T) {
		t.Parallel()
		sockPath := t.TempDir() + "/test.sock"
		ln, err := net.Listen("unix", sockPath)
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		t.Cleanup(func() { ln.Close() })

		srv := &http.Server{
			Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				io.WriteString(w, `not json`)
			}),
		}
		go srv.Serve(ln)
		t.Cleanup(func() { srv.Close() })

		proc := makeTestVLLMProcess(sockPath)

		var result []json.RawMessage
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			result = fetchLiveModels(proc, "m")
			if result == nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}

		if result != nil {
			t.Error("expected nil for malformed JSON")
		}
	})
}

func TestForwardToAnyActiveWithActive(t *testing.T) {
	t.Parallel()

	sockPath := t.TempDir() + "/test.sock"
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { ln.Close() })

	received := make(chan struct{}, 1)
	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			received <- struct{}{}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			io.WriteString(w, `{"upstream":"ok"}`)
		}),
	}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })

	o := makeTestOrchestrator(t)
	o.models[0].mu.Lock()
	o.models[0].state = stateActive
	o.models[0].socketPath = sockPath
	o.models[0].mu.Unlock()

	req := httptest.NewRequest(http.MethodGet, "/docs", nil)
	rec := httptest.NewRecorder()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		rec = httptest.NewRecorder()
		o.forwardToAnyActive(rec, req)
		if rec.Code == http.StatusOK {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"upstream":"ok"`) {
		t.Errorf("body %q missing upstream response", rec.Body.String())
	}
	select {
	case <-received:
	default:
		t.Error("upstream did not receive the request")
	}
}
