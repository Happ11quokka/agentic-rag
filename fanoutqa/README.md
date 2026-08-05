# FanOutQA local open-book traces

This directory supplies questions—not evaluation—from the official FanOutQA development
set. `dataset.py` downloads only `fanout-final-dev.json` at repository commit
`4e254877c0d28378e800171ab92781dbba893718` and verifies its byte size, Git blob SHA-1,
record count, required fields, and unique IDs. Only question ID, question text, and
categories enter agent inputs or traces. Gold answers, decompositions, and evidence do not.

This experiment is not an official FanOutQA result. FanOutQA questions refer to English
Wikipedia revisions from 2023-11-20; local retrieval uses Upstash's June 2024 BGE-M3
paragraph collection. No correctness scoring or leaderboard submission is included.

From repository root:

```bash
uv run download-models
uv run run-experiment
```

The interactive launcher offers simultaneous main/draft inference, separate tool-use
phases, synchronized draft-query prefetch, and a Qdrant-only storage benchmark. It asks
for task/query and repetition counts, prints progress and final metrics, and writes
manifests, JSONL traces, and summaries under `experiment/results/`.

Tool-use uses the Wikipedia bundle selected by `WIKIPEDIA_BUNDLE_DIR` or
`.wikipedia.local.toml`, loads BGE-M3 once, and runs main and draft phases sequentially.
It restarts local Docker Qdrant before every role/question/repetition unit. Prefetched
tool calls similarly restart Qdrant before every paired run, then let target and draft
share only that fresh backend. Restart time is excluded from end-to-end latency; host OS
page cache is not evicted. Both experiments print horizontal ASCII histograms for
end-to-end and Qdrant RPC latency. The parallel experiment does not require Qdrant. SSE
timing remains client-observed rather than a claimed GPU-token timestamp.
