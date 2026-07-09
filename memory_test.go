package main

import (
	"strings"
	"testing"
)

func TestParseNvidiaSmi(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		input   string
		want    map[int]int64
		wantErr string
	}{
		{
			name:  "single gpu",
			input: "0, 24576\n",
			want:  map[int]int64{0: 24576},
		},
		{
			name:  "multi gpu",
			input: "0, 24576\n1, 40960\n2, 40960\n3, 40960\n",
			want:  map[int]int64{0: 24576, 1: 40960, 2: 40960, 3: 40960},
		},
		{
			name:    "empty output",
			input:   "",
			wantErr: "no GPU entries found",
		},
		{
			name:    "malformed line no comma",
			input:   "0 24576\n",
			wantErr: "unexpected line",
		},
		{
			name:    "bad index",
			input:   "x, 24576\n",
			wantErr: "bad index",
		},
		{
			name:    "bad mb value",
			input:   "0, abc\n",
			wantErr: "bad memory value",
		},
		{
			name:  "trailing whitespace and blank lines",
			input: "\n0, 24576  \n\n",
			want:  map[int]int64{0: 24576},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := parseNvidiaSmi(tc.input)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("error %q does not contain %q", err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %d entries, want %d", len(got), len(tc.want))
			}
			for k, wv := range tc.want {
				if got[k] != wv {
					t.Errorf("device %d: got %d MB, want %d MB", k, got[k], wv)
				}
			}
		})
	}
}

func TestInitMemory(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		GPUGroups: []GPUGroup{
			{ID: "g0", GPUs: []int{0, 1}},
			{ID: "g1", GPUs: []int{2}},
		},
	}

	t.Run("valid", func(t *testing.T) {
		t.Parallel()
		orig := queryNvidiaSmi
		origMem := readMemAvailableMB
		t.Cleanup(func() { queryNvidiaSmi = orig; readMemAvailableMB = origMem })

		queryNvidiaSmi = func() (string, error) { return "0, 24576\n1, 24576\n2, 40960\n", nil }
		readMemAvailableMB = func() (int64, error) { return 65536, nil }

		ms, err := initMemory(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(ms.groups) != 2 {
			t.Fatalf("got %d groups, want 2", len(ms.groups))
		}
		if ms.groups[0].measuredTotalVRAMMB != 49152 {
			t.Errorf("g0 total = %d, want 49152", ms.groups[0].measuredTotalVRAMMB)
		}
		if ms.groups[1].measuredTotalVRAMMB != 40960 {
			t.Errorf("g1 total = %d, want 40960", ms.groups[1].measuredTotalVRAMMB)
		}
		if ms.freeCPURAMB != 65536 {
			t.Errorf("freeCPURAMB = %d, want 65536", ms.freeCPURAMB)
		}
	})

	t.Run("nvidia_smi_error", func(t *testing.T) {
		t.Parallel()
		orig := queryNvidiaSmi
		t.Cleanup(func() { queryNvidiaSmi = orig })
		queryNvidiaSmi = func() (string, error) { return "", errTest("nvidia-smi failed") }

		_, err := initMemory(cfg)
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("parse_error", func(t *testing.T) {
		t.Parallel()
		orig := queryNvidiaSmi
		t.Cleanup(func() { queryNvidiaSmi = orig })
		queryNvidiaSmi = func() (string, error) { return "bad output no comma\n", nil }

		_, err := initMemory(cfg)
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("gpu_id_not_in_smi", func(t *testing.T) {
		t.Parallel()
		orig := queryNvidiaSmi
		origMem := readMemAvailableMB
		t.Cleanup(func() { queryNvidiaSmi = orig; readMemAvailableMB = origMem })
		// Only device 0 present; config asks for 0,1,2.
		queryNvidiaSmi = func() (string, error) { return "0, 24576\n", nil }
		readMemAvailableMB = func() (int64, error) { return 65536, nil }

		_, err := initMemory(cfg)
		if err == nil {
			t.Fatal("expected error for missing device ID")
		}
		if !strings.Contains(err.Error(), "not found in nvidia-smi") {
			t.Errorf("error %q missing expected substring", err.Error())
		}
	})

	t.Run("meminfo_error", func(t *testing.T) {
		t.Parallel()
		orig := queryNvidiaSmi
		origMem := readMemAvailableMB
		t.Cleanup(func() { queryNvidiaSmi = orig; readMemAvailableMB = origMem })
		queryNvidiaSmi = func() (string, error) { return "0, 24576\n1, 24576\n2, 40960\n", nil }
		readMemAvailableMB = func() (int64, error) { return 0, errTest("meminfo failed") }

		_, err := initMemory(cfg)
		if err == nil {
			t.Fatal("expected error")
		}
	})
}

// errTest is a simple error value for injection.
type errTest string

func (e errTest) Error() string { return string(e) }
