#!/usr/bin/env python3
"""Reconstructs vLLM's per-engine periodic stats log line for ALL currently
running models, discovered dynamically from the socket directory rather than
hardcoded by name — so this works uniformly whether a model has
--data-parallel-size > 1 (where vLLM's own StatLoggerManager disables the
native log line entirely: vllm/v1/metrics/loggers.py:1340 — "AsyncLLM created
with api_server_count more than 1; disabling stats logging to avoid
incomplete stats") or not (single-engine models, where this is redundant with
vLLM's own native logging but harmless — same fields, same source data).

This is NOT a vLLM bug being patched — it's an external substitute using the
always-on Prometheus /metrics endpoint (no client_count gate, unlike the log
line), reformatted to match the disabled log line's exact fields: Avg
prompt/generation throughput, Running/Waiting reqs, GPU KV cache usage,
Prefix cache hit rate — per engine, per model, per interval.

Sockets are discovered by globbing <socket-dir>/*.sock every interval, so
adding or removing models from the running config requires no changes here
— no model name is hardcoded anywhere in this script.

Usage: sudo python3 dp_metrics_reconstruct.py [socket-dir] [interval-seconds]
Defaults: socket-dir=/run/vllm, interval=10
"""
import glob
import http.client
import os
import re
import socket
import sys
import time


class UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path, timeout=5):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def fetch_metrics(socket_path: str) -> str:
    conn = UnixSocketHTTPConnection(socket_path)
    try:
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        return resp.read().decode("utf-8")
    finally:
        conn.close()


METRIC_RE = re.compile(r'^(vllm:\w+)\{([^}]*)\}\s+([0-9.eE+-]+)$', re.MULTILINE)
ENGINE_RE = re.compile(r'engine="(\d+)"')
MODEL_RE = re.compile(r'model_name="([^"]+)"')


def parse_metrics(text: str) -> dict:
    """Returns {(model_name, engine_id): {metric_name: value}}."""
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        em = ENGINE_RE.search(labels)
        mm = MODEL_RE.search(labels)
        if not em or not mm:
            continue
        key = (mm.group(1), em.group(1))
        out.setdefault(key, {})[name] = float(value)
    return out


def discover_sockets(socket_dir: str) -> list:
    return sorted(glob.glob(os.path.join(socket_dir, "*.sock")))


def main():
    socket_dir = sys.argv[1] if len(sys.argv) > 1 else "/run/vllm"
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    prev: dict = {}
    prev_time = time.monotonic()

    while True:
        time.sleep(interval)
        now = time.monotonic()
        delta = now - prev_time
        prev_time = now

        sockets = discover_sockets(socket_dir)
        if not sockets:
            ts = time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime())
            print(f"{ts} [vllm-stats-all] no sockets found in {socket_dir}", flush=True)
            continue

        for sock_path in sockets:
            try:
                text = fetch_metrics(sock_path)
            except (OSError, http.client.HTTPException) as e:
                ts = time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime())
                print(f"{ts} [vllm-stats-all] {os.path.basename(sock_path)}: metrics fetch failed: {e}", flush=True)
                continue

            engines = parse_metrics(text)
            for (model_name, engine_id) in sorted(engines, key=lambda k: (k[0], int(k[1]))):
                m = engines[(model_name, engine_id)]
                running = int(m.get("vllm:num_requests_running", 0.0))
                waiting = int(m.get("vllm:num_requests_waiting", 0.0))
                cache_pct = m.get("vllm:kv_cache_usage_perc", 0.0) * 100.0
                prompt_total = m.get("vllm:prompt_tokens_total", 0.0)
                gen_total = m.get("vllm:generation_tokens_total", 0.0)
                pq_total = m.get("vllm:prefix_cache_queries_total", 0.0)
                ph_total = m.get("vllm:prefix_cache_hits_total", 0.0)

                key = (model_name, engine_id)
                prev_entry = prev.get(key, {})
                p_prev = prev_entry.get("prompt", prompt_total)
                g_prev = prev_entry.get("gen", gen_total)
                pq_prev = prev_entry.get("pq", pq_total)
                ph_prev = prev_entry.get("ph", ph_total)

                prompt_rate = max(0.0, prompt_total - p_prev) / delta if delta > 0 else 0.0
                gen_rate = max(0.0, gen_total - g_prev) / delta if delta > 0 else 0.0
                pq_delta = pq_total - pq_prev
                ph_delta = ph_total - ph_prev
                hit_rate = (ph_delta / pq_delta * 100.0) if pq_delta > 0 else 0.0

                prev[key] = {
                    "prompt": prompt_total,
                    "gen": gen_total,
                    "pq": pq_total,
                    "ph": ph_total,
                }

                ts = time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime())
                print(
                    f"{ts} [vllm/{model_name}] Engine {int(engine_id):03d}: "
                    f"Avg prompt throughput: {prompt_rate:.1f} tokens/s, "
                    f"Avg generation throughput: {gen_rate:.1f} tokens/s, "
                    f"Running: {running} reqs, Waiting: {waiting} reqs, "
                    f"GPU KV cache usage: {cache_pct:.1f}%, "
                    f"Prefix cache hit rate: {hit_rate:.1f}%",
                    flush=True,
                )


if __name__ == "__main__":
    main()
