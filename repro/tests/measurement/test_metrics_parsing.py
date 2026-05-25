from measurement.metrics_collector import parse_metrics, parse_slots, PollSample

METRICS_SAMPLE = """\
# HELP llamacpp:prompt_seconds_total Prefill time
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 12.34
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 45.67
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1500
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 3200
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 3200
# TYPE llamacpp:n_tokens_max gauge
llamacpp:n_tokens_max 4096
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
"""

SLOTS_SAMPLE = [
    {
        "id": 0,
        "is_processing": True,
        "id_task": 12,
        "n_ctx": 32768,
        "n_prompt_tokens": 2048,
        "n_prompt_tokens_processed": 2048,
        "n_prompt_tokens_cache": 512,
        "prompt": "...",
    }
]

def test_parse_metrics():
    d = parse_metrics(METRICS_SAMPLE)
    assert d["llamacpp:prompt_seconds_total"] == 12.34
    assert d["llamacpp:tokens_predicted_total"] == 3200
    assert d["llamacpp:requests_processing"] == 1

def test_parse_slots():
    d = parse_slots(SLOTS_SAMPLE)
    assert d["is_processing"] is True
    assert d["n_prompt_tokens"] == 2048
    assert d["n_prompt_tokens_cache"] == 512

def test_poll_sample_construction():
    m = parse_metrics(METRICS_SAMPLE)
    s = parse_slots(SLOTS_SAMPLE)
    sample = PollSample.from_endpoints(t=1.23, metrics=m, slots=s)
    assert sample.prefill_s_total == 12.34
    assert sample.is_processing is True
    assert sample.n_prompt_tokens_cache == 512
