import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from measurement.metrics_collector import MetricsCollector

METRICS_TEXT = """\
llamacpp:prompt_seconds_total {prefill}
llamacpp:tokens_predicted_seconds_total {decode}
llamacpp:prompt_tokens_total 0
llamacpp:tokens_predicted_total 0
llamacpp:n_decode_total 0
llamacpp:n_tokens_max 0
llamacpp:requests_processing 0
"""

class _FakeHandler(BaseHTTPRequestHandler):
    counter = [0]
    def do_GET(self):
        if self.path == "/metrics":
            i = self.counter[0]; self.counter[0] += 1
            body = METRICS_TEXT.format(prefill=i*0.1, decode=i*0.2).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        elif self.path == "/slots":
            body = b'[{"id":0,"is_processing":false,"n_prompt_tokens":0,"n_prompt_tokens_processed":0,"n_prompt_tokens_cache":0}]'
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

def test_collector_polls_and_slices():
    srv = _serve()
    port = srv.server_address[1]
    coll = MetricsCollector(f"http://127.0.0.1:{port}", interval_s=0.05)
    coll.start()
    t0 = time.perf_counter()
    time.sleep(0.3)
    t1 = time.perf_counter()
    coll.stop()

    samples = coll.slice(t0, t1)
    assert len(samples) >= 3
    # Cumulative counters must be monotone non-decreasing
    for a, b in zip(samples, samples[1:]):
        assert b.prefill_s_total >= a.prefill_s_total
        assert b.decode_s_total >= a.decode_s_total

def test_slice_includes_boundary_samples():
    srv = _serve()
    port = srv.server_address[1]
    coll = MetricsCollector(f"http://127.0.0.1:{port}", interval_s=0.05)
    coll.start()
    time.sleep(0.5)
    # Pick a window that's safely in the past so "after" bracket is non-empty.
    # With 50ms intervals + ~10 samples collected, mid 150ms in the past is
    # guaranteed to have at least 2 samples after it.
    mid = time.perf_counter() - 0.150
    samples = coll.slice(mid, mid + 0.001)
    coll.stop()
    # Even a 1 ms window must include bracket samples (one before, one after)
    assert len(samples) >= 2
