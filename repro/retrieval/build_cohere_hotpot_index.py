#!/usr/bin/env python
"""Build a FAISS index over Cohere's Wikipedia-2023-11 passage embeddings,
restricted to passages whose article title appears in HotpotQA dev (fullwiki).

The Cohere `en` split is ~41.5M passages across 415 parquet shard files
(~90 GB total, 1024-d embeddings already computed). We stream those shards one
file at a time (disk-light: pyarrow reads row groups over HTTP range requests),
keep only rows whose normalized title is a HotpotQA dev title, and checkpoint
the matched rows per shard so the (multi-hour) build is resumable and never
holds more than one shard in flight. Finally we assemble a FAISS
IndexFlatIP(1024) plus an aligned metadata parquet, and print a gold-title
coverage report.

Run from cwd = repro/ :
    python retrieval/build_cohere_hotpot_index.py                 # full build (415 shards)
    python retrieval/build_cohere_hotpot_index.py --max-shards 40 # partial (~10%, validation)
    python retrieval/build_cohere_hotpot_index.py --rebuild-index # reassemble from checkpoints only
"""
import argparse
import glob
import json
import os
import time
import unicodedata

REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
DIM = 1024

HERE = os.path.dirname(os.path.abspath(__file__))            # repro/retrieval
REPRO = os.path.dirname(HERE)                               # repro
AGENTBENCH = os.environ.get(
    "AGENTBENCH_PATH", os.path.join(os.path.dirname(REPRO), "AgentBench")
)
HOTPOT_JSON = os.path.join(AGENTBENCH, "dataset", "hotpot_dev_fullwiki_v1.json")

OUT_DIR = os.path.join(HERE, "index")
SHARD_DIR = os.path.join(OUT_DIR, "_shards")
INDEX_PATH = os.path.join(OUT_DIR, "cohere_hotpot.faiss")
META_PATH = os.path.join(OUT_DIR, "cohere_hotpot_meta.parquet")
TITLES_CACHE = os.path.join(OUT_DIR, "hotpot_titles.json")


def norm_title(t) -> str:
    return unicodedata.normalize("NFC", str(t)).strip().lower()


def load_hotpot_titles():
    """Return (gold_titles, all_titles) as sets of normalized titles.

    gold  = union of supporting_facts titles (the passages needed to answer)
    all   = gold union of all context titles offered by the fullwiki dev set
    """
    with open(HOTPOT_JSON) as f:
        data = json.load(f)
    gold, ctx = set(), set()
    for ex in data:
        for sf in ex.get("supporting_facts", []):
            if sf:
                gold.add(norm_title(sf[0]))
        for c in ex.get("context", []):
            if c:
                ctx.add(norm_title(c[0]))
    return gold, (gold | ctx)


def list_shards():
    """Resolve the ordered list of `en` parquet shard URLs (no data download)."""
    from datasets import load_dataset_builder
    b = load_dataset_builder(REPO, "en")
    files = list(b.config.data_files["train"])
    files.sort()
    return files


def process_shard(path, titles, fs):
    """Stream one parquet shard, return matched [(id, title, text, emb), ...]."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=16384,
                                     columns=["_id", "title", "text", "emb"]):
            tcol = batch.column("title").to_pylist()
            keep = [j for j, t in enumerate(tcol) if norm_title(t) in titles]
            if not keep:
                continue
            sub = batch.take(pa.array(keep, type=pa.int64())).to_pydict()
            rows.extend(zip(sub["_id"], sub["title"], sub["text"], sub["emb"]))
    return rows


def assemble():
    """Build the FAISS index + metadata parquet from shard checkpoints.

    Memory-safe: each shard's embeddings are added to the index incrementally
    (no giant in-memory concat), so peak RAM is ~the index itself (N*1024*4B),
    not N python lists. Passage _ids are unique across the corpus, so no dedup
    is needed. Returns the list of all matched titles (for the coverage report)
    or None if nothing matched.
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import faiss

    index = faiss.IndexFlatIP(DIM)
    ids_all, titles_all, texts_all = [], [], []
    for p in sorted(glob.glob(os.path.join(SHARD_DIR, "*.parquet"))):
        t = pq.read_table(p, columns=["_id", "title", "text", "emb"])
        if t.num_rows == 0:
            continue
        flat = t.column("emb").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        m = np.asarray(flat, dtype=np.float32).reshape(t.num_rows, DIM)
        faiss.normalize_L2(m)
        index.add(m)
        ids_all.extend(t.column("_id").to_pylist())
        titles_all.extend(t.column("title").to_pylist())
        texts_all.extend(t.column("text").to_pylist())
    if index.ntotal == 0:
        print("[assemble] no matched passages in any shard -> nothing to build")
        return None
    assert index.ntotal == len(ids_all), (index.ntotal, len(ids_all))
    faiss.write_index(index, INDEX_PATH)
    pd.DataFrame({"_id": ids_all, "title": titles_all, "text": texts_all}).to_parquet(META_PATH)
    return titles_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=None,
                    help="process only the first N shard files (partial/validation build)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="reassemble FAISS from existing shard checkpoints without streaming")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent shard downloads (per-shard checkpoints make this safe)")
    args = ap.parse_args()

    os.makedirs(SHARD_DIR, exist_ok=True)

    # Idempotency/resume is per-shard (cached checkpoints in _shards/), so the
    # build scales incrementally: re-running with a larger --max-shards skips
    # cached shards and only streams the new ones, then always reassembles.
    gold, titles = load_hotpot_titles()
    with open(TITLES_CACHE, "w") as f:
        json.dump({"gold": sorted(gold), "all": sorted(titles)}, f)
    print(f"[titles] gold={len(gold)} all={len(titles)}")

    if not args.rebuild_index:
        from huggingface_hub import HfFileSystem
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd

        shards = list_shards()
        if args.max_shards:
            shards = shards[:args.max_shards]

        def checkpoint(path):
            return os.path.join(SHARD_DIR, os.path.basename(path))

        todo = [p for p in shards if not os.path.exists(checkpoint(p))]
        print(f"[shards] {len(shards)} total | {len(shards)-len(todo)} cached | "
              f"{len(todo)} to stream | workers={args.workers}")
        fs = HfFileSystem()

        def work(path):
            # Tolerate transient network errors: retry a few times, and on
            # persistent failure return without writing a checkpoint so the
            # shard is simply retried on the next run (no whole-build crash).
            s = time.time()
            err = None
            for attempt in range(3):
                try:
                    rows = process_shard(path, titles, fs)
                    out = checkpoint(path)
                    if rows:
                        ids, tt, txt, embs = zip(*rows)
                        pd.DataFrame({"_id": list(ids), "title": list(tt), "text": list(txt),
                                      "emb": [list(e) for e in embs]}).to_parquet(out)
                    else:
                        pd.DataFrame({"_id": [], "title": [], "text": [], "emb": []}).to_parquet(out)
                    return os.path.basename(path), len(rows), time.time() - s, None
                except Exception as e:
                    err = repr(e)[:200]
                    time.sleep(2 * (attempt + 1))
            return os.path.basename(path), -1, time.time() - s, err

        t0 = time.time()
        done = 0
        failed = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(work, p) for p in todo]
            for fut in as_completed(futures):
                name, k, dt, err = fut.result()
                done += 1
                if k < 0:
                    failed.append(name)
                    print(f"[{done}/{len(todo)}] {name}: FAILED -> will retry on rerun ({err})", flush=True)
                else:
                    print(f"[{done}/{len(todo)}] {name}: matched={k} ({dt:.1f}s)", flush=True)
        print(f"[stream] done in {time.time()-t0:.0f}s | failed={len(failed)} {failed[:10]}")

    print("[assemble] building FAISS index from shard checkpoints ...")
    titles_all = assemble()
    if titles_all is None:
        return

    matched_titles = {norm_title(t) for t in titles_all}
    hit = len(gold & matched_titles)
    print(f"[done] passages={len(titles_all)} uniq_titles={len(matched_titles)} "
          f"gold_coverage={hit/max(len(gold),1):.1%} ({hit}/{len(gold)})")
    print(f"[out]  {INDEX_PATH}")
    print(f"[out]  {META_PATH}")


if __name__ == "__main__":
    main()
