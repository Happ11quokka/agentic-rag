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

# Direct non-interactive entrypoint:
./experiment/setup/run.sh
./experiment/setup/run.sh --limit 1
./experiment/setup/run.sh --question-id 7dcbbbdc7f1120cd
./experiment/setup/run.sh --role main
./experiment/setup/run.sh --role draft
./experiment/setup/run.sh --run-dir experiment/results/<existing-run> --resume
```

The command reuses the Wikipedia bundle selected by `WIKIPEDIA_BUNDLE_DIR` or
`.wikipedia.local.toml`, starts/reuses its Qdrant runtime, loads BGE-M3 once, and then runs
main and draft model phases sequentially on one `llama-server` slot. Large dataset, model,
and result artifacts are ignored by Git.

Each run writes `manifest.json`, fsynced `traces.jsonl`, `timing_summary.json`, and one
server log per selected role. SSE chunks are client-observed response chunks, not claimed
GPU-token timestamps.
