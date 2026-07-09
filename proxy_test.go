package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
		{"model in nested object", `{"data":{"model":"nested"},"model":"top"}`, "nested"},
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
