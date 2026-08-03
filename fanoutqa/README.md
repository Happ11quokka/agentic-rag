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
phases, and a Qdrant-only storage benchmark. It asks for task/query and repetition counts,
then prints progress and final metrics to stdout. No experiment configuration or result
files are written.

Tool-use reuses the Wikipedia bundle selected by `WIKIPEDIA_BUNDLE_DIR` or
`.wikipedia.local.toml`, starts/reuses Qdrant, loads BGE-M3 once, and runs main and draft
model phases sequentially. The parallel experiment starts both model servers together and
does not require Qdrant. SSE timing remains client-observed rather than a claimed GPU-token
timestamp.
