package main

import (
	"testing"
)

func TestPickGroup(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		groups   []*groupState
		neededMB int64
		wantIdx  int
		wantErr  bool
	}{
		{
			name: "no group qualifies",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 10000, usedVRAMMB: 9000, measuredFreeMB: -1},
			},
			neededMB: 2000,
			wantIdx:  -1,
			wantErr:  true,
		},
		{
			name: "single qualifying group",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 24576, usedVRAMMB: 0, measuredFreeMB: -1},
			},
			neededMB: 8000,
			wantIdx:  0,
			wantErr:  false,
		},
		{
			name: "picks smallest total among qualifying",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 40960, usedVRAMMB: 0, measuredFreeMB: -1},
				{id: "g1", measuredTotalVRAMMB: 24576, usedVRAMMB: 0, measuredFreeMB: -1},
				{id: "g2", measuredTotalVRAMMB: 80000, usedVRAMMB: 0, measuredFreeMB: -1},
			},
			neededMB: 8000,
			wantIdx:  1, // g1 has smallest total
			wantErr:  false,
		},
		{
			name: "exact fit",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 8000, usedVRAMMB: 0, measuredFreeMB: -1},
			},
			neededMB: 8000,
			wantIdx:  0,
			wantErr:  false,
		},
		{
			name: "only larger group qualifies",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 10000, usedVRAMMB: 9000, measuredFreeMB: -1}, // 1000 free, not enough
				{id: "g1", measuredTotalVRAMMB: 40960, usedVRAMMB: 0, measuredFreeMB: -1},    // 40960 free
			},
			neededMB: 8000,
			wantIdx:  1,
			wantErr:  false,
		},
		{
			name: "measured free caps accounted free",
			groups: []*groupState{
				{id: "g0", measuredTotalVRAMMB: 24576, usedVRAMMB: 0, measuredFreeMB: 3000},
			},
			neededMB: 8000,
			wantIdx:  -1,
			wantErr:  true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ms := &memoryState{groups: tc.groups}
			o := &orchestrator{ms: ms}

			// pickGroup requires ms.mu held; lock it for the call.
			ms.mu.Lock()
			idx, err := o.pickGroup(tc.neededMB)
			ms.mu.Unlock()

			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if idx != tc.wantIdx {
				t.Errorf("pickGroup returned idx %d, want %d", idx, tc.wantIdx)
			}
		})
	}
}

func TestPlaceholderVRAM(t *testing.T) {
	t.Parallel()

	gs := &groupState{measuredTotalVRAMMB: 24576, measuredFreeMB: -1}

	t.Run("measured", func(t *testing.T) {
		t.Parallel()
		mem := &modelMemory{measured: true, fullKVVRAMMB: 12288}
		got := placeholderVRAM(gs, mem)
		if got != 12288 {
			t.Errorf("placeholderVRAM = %d, want 12288", got)
		}
	})

	t.Run("not_measured", func(t *testing.T) {
		t.Parallel()
		mem := &modelMemory{measured: false}
		got := placeholderVRAM(gs, mem)
		want := int64(float64(gs.measuredTotalVRAMMB) * 0.85)
		if got != want {
			t.Errorf("placeholderVRAM = %d, want %d", got, want)
		}
	})
}
