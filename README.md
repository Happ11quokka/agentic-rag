# agentic_rag — "The Cost of Dynamic Reasoning" 재현 + decode-RAG 벡터DB baseline

KAIST 논문 **"The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective"** (arXiv [2506.04301v2](https://arxiv.org/abs/2506.04301), HPCA-2026)의 HotpotQA 실험을 **로컬 Apple Silicon 환경**에서 재현하고, 그 위에서 **decode-RAG**(검색 대기를 decode 뒤에 숨기는 기법) 연구의 정량 baseline을 구축하는 저장소입니다.

> **핵심 지표 — retrieval-time fraction (tool/e2e)**: 에이전트가 한 질문을 푸는 전체 시간 중 검색 응답을 기다리며 GPU가 노는(idle) 비율. 이게 decode-RAG가 숨길 수 있는 비효율의 상한입니다.

---

## 1. 배경 / 동기

- AI 에이전트(ReAct 등)는 `생각 → 검색 → 관찰`을 반복하는데, **검색을 기다리는 동안 GPU가 절반 가까이 논다**(논문: ReAct-HotpotQA GPU idle **54.5%**).
- 이 idle 구간 = 단일 요청 안 LLM↔도구의 순차 의존성 때문에 생기는 비효율 = **decode 중 retrieval을 미리 당겨와(prefetch/overlap) 숨길 수 있는 대상**.
- 본 repo는 ① 논문 패턴을 로컬에서 재현하고 ② decode-RAG baseline(`retrieval-time fraction`)을 재현 가능하게 측정한다.

**환경 격차상 절대 수치(초·Wh)는 비교 불가** → **비율·분포·순위·트레이드오프 형상**의 재현을 목표로 한다.

| | 논문 | 본 재현 |
|---|---|---|
| GPU | NVIDIA A100 (GCP) | Apple M3 Pro 36GB (Metal) |
| Backend | vLLM 0.6.6 · FP16 | llama.cpp · Q4_K_M GGUF |
| 모델 | Llama-3.1 8B/70B | Llama-3.1-8B-Instruct Q4_K_M |
| 벤치 | HotpotQA·WebShop·MATH·HumanEval | HotpotQA (decode-RAG에 가장 적합) |

---

## 2. 지금까지의 결과 (요약)

**(A) 논문 재현** — 패턴 수준 일치
| Figure | 검증 | 결과 |
|---|---|---|
| Fig 6 GPU idle (ReAct) | ≈ 54.5% | **54.98%** (Δ0.48%p) ✅ |
| Fig 4 LLM 호출 폭증 (LATS/ReAct) | ≥ 5× | 21.9× ✅ |
| Fig 7 heavy tail (p95/p50) | ≥ 2.0 | 3.34 ✅ |
| Fig 13 Pareto | 형상 | LATS 최정확·최고비용, LLMCompiler 최속 ✅(부분) |
| Fig 13 Spearman ρ | ≥ 0.6 | 0.40 (LATS n=3 한계) ⚠️ |
| Fig 8 tool tokens | ≥ 10% | 0% (handler 한계) ❌ |

**(B) decode-RAG baseline** — 검색 백엔드를 Live Wikipedia → Cohere 밀집 벡터DB(+rerank)로 교체해 재현 가능하게 측정.

`retrieval-time fraction`을 두 끝점으로 **bracket**:
- Live Wikipedia ReAct: **~45%** (상한, 비재현 — 라이브 API 드리프트)
- Cohere VDB dense-only: **~11%** (하한, 재현 가능 — 검색이 너무 빠름)
- Cohere VDB **+ rerank**: **~31%** (재현 가능, 논문 tool share 30%대와 일치)

**3자 비교(논문 vs 1차 Live Wiki vs 현재 VDB+rerank)** — `three_way_compare.html`:
| 지표 | 논문 | 1차 (n=50) | 현재 rerank (n=139) |
|---|---|---|---|
| ① tool/e2e | 30.2%* | 44.8% | 31.3% |
| ② LLM / tool | 69 / 30* | 55 / 45 | 68 / 31 |
| ③ calls/q (LLM·tool) | 9.2×* | 8.84·8.54 | 5.67·5.50 |
| ④ EM / F1 | ~25–30% EM* | 32.0 / 38.5 | 30.9 / 41.9 |
| ⑤ p95/p50 | — | 3.15× | 4.66× |
| ⑥ prefill/decode | 4.7 / 74* | 26 / 74 | 39 / 61 |

\* 논문값은 5종 에이전트·4벤치 평균(ReAct-HotpotQA 단독 아님). 정확도는 EM(논문 지표); F1은 본 재현이 보완 추가.

**hop별 retrieval≈decode 균형**: rerank로 hop별 `retrieval/decode-only` 중앙값 **0.23 → 1.08**(≈1:1) — 검색을 decode 뒤에 거의 다 숨길 수 있는 상태. (decode-RAG 기법 실험의 토대)

> ⚠️ **현재 진행 상태**: 전체 7,405문항 rerank 런은 Cohere **trial 키 월 1,000회 한도**(계정 단위)로 **139/7,405에서 일시 중단**. production 키로 교체 후 `sweep/run_vectordb_full.sh` 재실행 시 이어서 완주(진도 보존).

---

## 3. 저장소 구조

```
agentic_rag/
├── README.md                     # (이 문서)
├── 2506.04301v2.pdf              # 원본 논문
├── paper_analysis.md             # 논문 전체 분석 (한국어)
├── experiment_methodology.md     # 실험 설계·측정 프로토콜
├── presentation.html / .md       # 재현 발표 덱 + 대본
├── progress_report.html          # 진행 상황 보고
├── three_way_compare.html        # 논문 vs 1차 vs 현재 3자 비교 (메인 결과)
├── AgentBench/                   # 에이전트 프레임워크 (ReAct/Reflexion/LATS/LLMCompiler)
│   ├── run_react.py              # ReAct 엔트리 (vectorDB 지원 추가)
│   ├── dataset/hotpot_dev_fullwiki_v1.json   # HotpotQA dev 7,405문항
│   └── src/tools/hotpotqa_tools/
│       ├── wikipedia.py          # 라이브 Wikipedia 검색 (1차)
│       └── vector_search.py      # Cohere FAISS + cross-encoder rerank (현재)
└── repro/                        # 재현 인프라
    ├── pyproject.toml            # 의존성 (Python 3.13)
    ├── setup/                    # llama.cpp 빌드·모델 다운로드·서버 기동
    ├── models/                   # Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    ├── retrieval/                # Cohere 벡터DB 인덱스 빌더 + index/(faiss·parquet)
    ├── sweep/                    # 실험 오케스트레이션 (configs/, run_full.sh, sweep_runner.py …)
    ├── measurement/              # trace 스키마·콜백·llama-server 메트릭·EM/F1 채점
    ├── analysis/                 # plot_fig*.py + 3자비교·hop balance·decode-rag 분석
    ├── results/raw/*.jsonl       # 실험 원자료 (append+fsync, resume 가능)
    ├── results/figures/          # 생성 그래프
    └── tests/                    # pytest (measurement·sweep·integration)
```

---

## 4. 환경 / 하드웨어

- **Apple M3 Pro · 36GB 통합 메모리 · Metal**
- **Python 3.13.0** (`repro/.python-version`)
- **llama.cpp** (commit `b9310`, `-DGGML_METAL=ON`)으로 빌드, **Llama-3.1-8B-Instruct Q4_K_M**(~5GB) 서빙
- **AgentBench** (commit `ef5b195f6904`) + 3개 패치
- 주요 의존성: `langchain 1.0.x`, `langchain-openai`, `langgraph`, `faiss-cpu`, `cohere`, `transformers`+`torch`(rerank), `datasets`, `pydantic 2`, `matplotlib`, `pyyaml`

> 70B Q4(~40GB)는 36GB 초과로 불가, 8B FP16은 decode 2~3배 느려 시간 예산 초과 → **8B Q4_K_M** 채택.

---

## 5. 설치 (Setup)

```bash
cd repro

# 1) llama.cpp 빌드 (Metal) + 모델 다운로드
./setup/install_llamacpp.sh                 # cmake -B build -DGGML_METAL=ON && build
./setup/download_model.sh                   # bartowski/Meta-Llama-3.1-8B-Instruct-GGUF (Q4_K_M)

# 2) Python 환경
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3) AgentBench clone + 우리 수정분 패치 적용
#    (AgentBench/는 third-party라 이 repo에 미포함 → 직접 clone 후 단일 패치 적용)
git clone https://github.com/VIA-Research/AgentBench.git ../AgentBench
cd ../AgentBench
git checkout ef5b195f69048865abceed472971237805bb9bc8     # 패치가 맞춰진 base 커밋
git apply ../repro/patches/agentbench.patch               # vector_search.py(Cohere FAISS+rerank) 등 전체 수정분
cd ../repro

# 4) (decode-RAG baseline용) Cohere 벡터DB 인덱스 — 선택, ~4h, COHERE_API_KEY 필요
export COHERE_API_KEY="cohere_..."          # 런타임 질문 임베딩용
python retrieval/build_cohere_hotpot_index.py            # 전체 415샤드 → index/cohere_hotpot.faiss(3.4GB) + .parquet(237MB)
#   --max-shards 40  으로 ~10% 검증 빌드 가능
```

**API 키**: 로컬 LLM은 더미 키로 동작. 검색 백엔드만 Cohere 키 필요.
```bash
export OPENAI_API_KEY="sk-dummy-local"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export COHERE_API_KEY="cohere_..."          # 벡터DB 검색 시에만
```

---

## 6. 실행 (Run)

### 6.1 llama-server 기동
```bash
./setup/start_server.sh                     # :8000, cache OFF (기본)
./setup/start_server_cache_on.sh            # --cache-reuse 256 (Fig 9용)
```

### 6.2 sweep 실행 — 한 config씩
```bash
nohup sweep/run_full.sh <config.yaml> [cache_on] > results/sweep_logs/<name>.log 2>&1 &
```
`run_full.sh`가 venv·환경변수·llama-server·warmup·`--resume`·로깅을 처리. 출력: `results/raw/<run_id>.jsonl`.

| config | 내용 |
|---|---|
| `fig13_pareto.yaml` | 4종 에이전트 baseline (ReAct/Reflexion/LATS/LLMCompiler) |
| `fig13_pareto_cache_on.yaml` | prefix-cache 켠 변형 (Fig 9) |
| `fig14_iteration.yaml` / `fig15_fewshot.yaml` | ReAct iteration / few-shot 스윕 |
| `fig16a/b/c_*.yaml` | Reflexion·LATS test-time scaling 3패널 |
| `react_vectordb.yaml` | ReAct + Cohere dense VDB (50, decode-RAG 하한 11%) |
| `react_vectordb_rerank.yaml` | + cross-encoder rerank (50) |
| `react_vectordb_rerank_full.yaml` | 전체 7,405문항 (rerank) |

### 6.3 전체 자동 실행
```bash
nohup sweep/master_chain.sh & disown        # 7 sweeps + 9 plots (~60h)
tail -f results/sweep_logs/master_chain.log
```

### 6.4 전체 벡터DB 런 (crash-resilient driver)
```bash
nohup sweep/run_vectordb_full.sh > results/sweep_logs/vectordb_full.log 2>&1 &
```
`run_vectordb_full.sh` 기능: `caffeinate`(맥 안 잠) + 자동 재시작 루프 + embed 재시도/backoff + **circuit breaker**(API 한도 8연속 실패 시 깨끗이 정지). 언제든 `kill $(cat .vectordb_full.pid)`로 멈추고 재실행하면 이어짐. (resume = append+fsync JSONL, 완료 문항 자동 skip.)

### 6.5 분석/그래프
```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-dummy-local OPENAI_BASE_URL=http://127.0.0.1:8000/v1
python -m analysis.plot_fig13           # Pareto → results/figures/fig13_pareto.png
python -m analysis.plot_three_way_compare   # ①~⑥ 3자 비교 12장
python -m analysis.plot_hop_balance         # hop별 retrieval/decode 균형
python -m analysis.decode_rag_ratio         # retrieval/e2e 비율 집계
#  그 외 plot_fig{4,5,7,8,9,14,15,16}, plot_decode_rag_compare
```

### 6.6 테스트
```bash
pytest repro/tests/                      # measurement·sweep 단위 + integration 스모크
```

---

## 7. 알려진 제약 / 트러블슈팅

- **Cohere trial 한도**: 월 1,000회(계정 단위). 전체 7,405(≈34k embed 호출)엔 **production 키 필요**. trial이면 circuit breaker가 429에서 정지(진도 보존).
- **faiss + torch libomp 충돌 (macOS)**: rerank 시 worker 스레드에서 faiss OMP search가 중복 libomp와 충돌해 segfault → `faiss.omp_set_num_threads(1)`(`vector_search.py`) + `KMP_DUPLICATE_LIB_OK=TRUE`(`run_full.sh`)로 해결.
- **절대값 비교 금지**: A100/vLLM vs M3/Q4 — 비율·형상만 비교.
- **표본 크기**: 1차 50문항(0.68%), 현재 139문항(1.9%) — 정확도는 표본 노이즈 큼.

---

## 8. 다음 단계 — decode-RAG prefetch (Step 4)

주 LLM(8B)이 decode하는 동안 **작은 보조 LLM(Llama-3.2-1B, 별도 :8001 서버)이 다음 검색어를 예측해 retrieval을 미리 실행** → 검색 대기를 숨긴다. 단계: ① trace에 쿼리 로깅 → ② 오프라인 예측 가능성·숨김 상한 측정(go/no-go) → ③ 라이브 prefetch 프로토타입(config로 on/off, 예측이 실제와 정확 일치 시에만 사용 → 정확도 불변) → ④ e2e 절감·2-모델 대역폭 경합세·정확도 불변 측정. 상세 설계는 [`repro/experiments/decode_rag_prefetch/PLAN.md`](repro/experiments/decode_rag_prefetch/PLAN.md) 참조.

---

## 9. 참고
- 논문: Kim et al., *The Cost of Dynamic Reasoning*, KAIST, arXiv 2506.04301v2 (HPCA-2026)
- 데이터셋: HotpotQA dev-fullwiki (7,405문항)
- 코퍼스: Cohere `wikipedia-2023-11-embed-multilingual-v3` (HotpotQA gold-title 필터, 880,777 passages)

---

## 10. Upstash Wikipedia vector DB workspace

Root project is a uv workspace for loading Upstash's English Wikipedia BGE-M3
embeddings onto an external USB HDD. Existing `repro/` project remains excluded
and independent.

**The backend is Milvus with a `DISKANN` index.** Qdrant/HNSW remains in the code
and is still selectable, but it does not work on this storage medium — see
[why below](#왜-diskann인가--qdrant-hnsw는-왜-기각됐나).

```bash
uv sync

# First run: choose one persistent bundle directory on the HDD.
# --max-shards 100 ≈ 10M records; the full 471 shards are 47,018,430.
uv run wikipedia-ingest milvus \
  --bundle-dir /Volumes/agentic_rag/wikipedia_diskann \
  --max-shards 100 --disable-xet --download-timeout 30

# Interrupted run: bundle path, Milvus settings, downloads, and checkpoints are reused.
uv run wikipedia-ingest milvus
uv run wikipedia-inspect --backend milvus
```

Ingest starts the Milvus stack itself, loads the shards, calls `flush()`, then
waits for the index to finish before returning. It reports the final
`index_type` / `indexed_rows` so a run that silently fell back to brute force is
visible.

Full English dataset contains 47,018,430 1024-dimensional records and transfers
roughly 205 GB; 100 shards is about 43 GB. Milvus and embedded Qdrant (`--path`)
ingest serially, while remote Qdrant runs one download-and-upsert pipeline per
worker. Each worker checkpoints and deletes its current Parquet shard before
claiming another, so raw source storage holds at most `--max-workers` complete
shards plus resumable partial files and the retained BGE-M3 model. Storage must
use a POSIX-compatible filesystem with block-level access, such as APFS or ext4;
ExFAT, NTFS, and network filesystems are rejected before the backend starts.

### 왜 DiskANN인가 — Qdrant HNSW는 왜 기각됐나

47M을 Qdrant HNSW로 이 HDD에 적재하는 것까지는 성공했으나(209 GB, 13시간),
**검색 1회가 48분에도 완료되지 않았다.** 원인은 버그가 아니라 자료구조와 매체의
불일치다.

```
HNSW가 요구하는 것 : 랜덤 4 KB 접근 수만 회를 밀리초 안에
HDD가 제공하는 것   : 랜덤 68 IOPS, 건당 16 KB 강제 (iostat 실측)
```

쿼리당 디스크 읽기 횟수가 갈린다. HNSW는 탐색 중 방문 노드마다 원본 벡터를 읽어야
해서 `O(세그먼트 수 × ef × 홉)`이고, DiskANN은 PQ 압축본을 램에 두고 탐색하므로
`O(search_list)` — 수백 회이고 코퍼스 크기에 거의 무관하다. 68 IOPS 매체에서는
후자만 성립한다.

| | Qdrant HNSW (47M) | Milvus DiskANN (1M 실측) |
|---|---|---|
| 콜드 검색 | **48분에도 미완료** | **17.7초** |
| 컬렉션 로드 | — | 25.5분 |

Docker VM 램을 8.2 → 27.4 GB로 늘리는 가설은 기각됐다 — Qdrant는 2.96 GB만 썼다.
병목이 캐시 용량이 아니라 랜덤 IOPS이기 때문이다. 세그먼트 96→4 병합과
`hnsw_on_disk: false`를 걸어도 40~90초로, 목표(수 초)에 못 미치면서 재색인 수 시간을
쓴다. 상세 진단은
[`repro/experiments/wikipedia_hdd/HNSW_HDD_MISMATCH.md`](repro/experiments/wikipedia_hdd/HNSW_HDD_MISMATCH.md),
결정 경위는
[`DEVLOG.md`](repro/experiments/wikipedia_hdd/DEVLOG.md),
설계는
[`docs/superpowers/specs/2026-08-03-milvus-diskann-design.md`](docs/superpowers/specs/2026-08-03-milvus-diskann-design.md)
에 있다.

### Milvus stack layout

Milvus standalone is three containers (`etcd`, `minio`, `milvus`) brought up by
compose project `wikipedia-milvus`. `ensure_milvus()` renders
`docker-compose.yml` and `milvus.yaml` into the storage directory, starts Docker
Desktop on macOS if needed, and waits on `http://localhost:9091/healthz` (300 s).

```
<bundle-dir>/
├── dataset/            # parquet shards, deleted as each one is ingested
├── models/bge-m3       # pinned encoder, query-side only
├── state/              # ingest checkpoints
├── manifest.json
└── milvus/             # --milvus-storage-dir, defaults here
    ├── docker-compose.yml
    ├── milvus.yaml     # queryNode.enableDisk: true
    └── volumes/{etcd,minio,milvus}
```

`queryNode.enableDisk` defaults to `false` upstream and a `DISKANN` index cannot
be loaded without it, so `milvus.yaml` sets it. Only `volumes/milvus` is read on
the query path; MinIO serves ingest and collection load. Measured footprint on
this hardware is about 25 GB per 1M records (minio 21 GB, milvus 3.8 GB, etcd
62 MB), so a 10M ingest needs roughly 250 GB — extrapolated from the 1M run, not
measured at 10M.

Images are pinned: `milvusdb/milvus:v2.5.27` (override with `MILVUS_IMAGE`),
`quay.io/coreos/etcd:v3.5.18`, `minio/minio:RELEASE.2024-05-28T17-19-04Z`.

### Milvus options

| Flag | Default | Notes |
|---|---|---|
| `--index-type` | `DISKANN` | Any Milvus index name; `AUTOINDEX` reverts to the in-memory path |
| `--search-list` | `100` | DiskANN candidate pool. Raises recall and latency together |
| `--milvus-storage-dir` | `<bundle-dir>/milvus` | Stack volumes. Put this on the medium under test |
| `--milvus-truncate-text` | off | See VARCHAR limit below |
| `--milvus-upsert` | off | See resume warning below |
| `--batch-size` | `1000` | Qdrant defaults lower; 256 would cost 39k round trips per 10M |

Both `--index-type` and `--search-list` are persisted to `manifest.json`, so a
resumed run reuses them without repeating the flags.

### Bundle selection and bounded runs

`wikipedia-ingest --bundle-dir` atomically writes untracked
`.wikipedia.local.toml` before transfer. Path must be absolute and becomes default
for later commands run from this source clone. Marker is anchored to clone's uv
workspace even if command is launched with another working directory. Resolution
order is explicit `--bundle-dir`, `WIKIPEDIA_BUNDLE_DIR`, then marker. Choosing a
new directory updates marker without moving or deleting old bundle.

```bash
# Small development ingest, marked partial but valid.
uv run wikipedia-ingest milvus \
  --bundle-dir "/Volumes/External Disk/wiki" \
  --max-shards 2

# Bound an import independently.
uv run wikipedia-ingest milvus --max-shards 1 --max-records 10000 --batch-size 256

# Select and remember another bundle directory.
uv run wikipedia-ingest milvus --bundle-dir /another/wiki
```

A partial bundle is a separate bundle, not a smaller version of an existing one:
`prepare_bundle` rewrites `manifest.json` on every run, so pointing a
`--max-shards` run at a completed bundle overwrites its record of what was
ingested. Use a distinct `--bundle-dir`.

Ingest honors `HF_TOKEN`, `--max-workers`, `--dataset-revision`, and
`--model-revision`. `--max-workers` defaults to 4 and controls remote Qdrant
shard pipelines plus initial model snapshot download. Before starting Qdrant or
Milvus, command checks required BGE-M3 files under `bundle_dir/models/bge-m3` and
downloads or resumes pinned model when files are incomplete or revision changed.
Complete matching model is reused without another snapshot transfer. It prints
30-second download and ingestion heartbeats. Parallel Hugging Face progress bars
are suppressed in favor of one aggregate ingest heartbeat showing checkpointed
and active shards plus records written. `Ctrl-C` stops all workers, aborts active
transfers, and retains incomplete shards for resume. For a faster transfer on a
machine with spare CPU, disk, and network capacity, enable hf-xet's
high-performance mode:

```bash
uv run wikipedia-ingest milvus --max-workers 16 --high-performance
```

`--float16` is Qdrant-only and stores vectors as native float16 to halve vector
storage. Repeat the flag on resumed runs; changing it requires cleaning up the
existing Qdrant storage and removing the `qdrant` section from `manifest.json`
first. It also halves the bytes read per query, so it changes the storage-medium
treatment this experiment measures — leave it off for HDD runs.

```bash
uv run wikipedia-ingest qdrant --float16
```

hf-xet stalled twice at ~270 MB with zero byte growth on this network. If that
happens, rerun with resumable HTTP and an explicit stall timeout; existing
completed and partial files are reused. **The flags are not persisted, so they
must be repeated on every resume.**

```bash
uv run wikipedia-ingest milvus --disable-xet --download-timeout 30
```

Completed shards are recorded per backend before deletion. A failed or
`--max-records`-limited shard remains on disk and is replayed on resume. Use
`--progress-interval 10` for more frequent heartbeat output or `0` to disable it.
When migrating an older multi-shard bundle, ingest retains up to the active
ingest-worker count and deletes surplus raw Parquet files to enforce the same
bound. Qdrant reads `QDRANT_URL`, `QDRANT_API_KEY`, and
`QDRANT_COLLECTION`; Milvus reads `MILVUS_URI`, `MILVUS_TOKEN`,
`MILVUS_DB_NAME`, and `MILVUS_COLLECTION`. Checkpoints live in bundle `state/`
unless `--checkpoint` is supplied.

### Measuring latency

```bash
uv run wikipedia-inspect --backend milvus \
  --bundle-dir /Volumes/agentic_rag/wikipedia_diskann \
  --runs 10 --search-list 100
```

`inspect` loads the collection explicitly and reports load time separately from
search latency. That separation matters here: on this HDD the 1M collection took
1,529 s to load and 17.7 s for the first query, and folding the two together had
previously been read as a 20 s search.

**Every run uses a different query.** Timing one query N times measures the page
cache from the second run onward, because the search re-walks the graph path the
previous one already faulted in — the wrong measurement for an on-disk index.
`--runs N` takes the first N of 20 built-in queries spread across unrelated
topics; `--query` repeated supplies your own set. Asking for more runs than
distinct queries is an error rather than a silent recycle.

```bash
uv run wikipedia-inspect --backend milvus --runs 3 \
  --query "What causes auroras?" \
  --query "Who wrote the Tale of Genji?" \
  --query "How are black holes detected?"
```

`summary:` reports `cold` (first query, nothing cached) alongside mean, median,
min, and max. Cite the cold number as the treatment; the spread across the
remaining queries is the steady state an agent would actually see.

Numbers measured before this change — including 71–80 ms warm on the 1M
bundle — came from one repeated query and are not comparable.

Useful while a query is in flight:

```bash
iostat -d disk6 1 4                       # tps = IOPS, KB/t = transfer unit
docker compose -f <bundle>/milvus/docker-compose.yml -p wikipedia-milvus ps
curl -s http://localhost:9091/healthz
```

#### Loading a large collection

`load_collection` on the 10M collection fails on stock settings, and the error
misdescribes itself — it reads `OOM if load, memUsage = 26192 MB` when real
process memory was 2,550 MB and nothing was short of RAM. The compose template
sets five values to get past it; the diagnosis is in
[`DEVLOG.md`](repro/experiments/wikipedia_hdd/DEVLOG.md#실측--10m-로드-거부의-정체-2026-08-05).

| Setting | Default | Here | Why |
|---|---|---|---|
| `queryCoord.taskExecutionCap` | 256 | 4 | Every admitted segment load reserves a fixed 128 MiB against the memory guard until it *finishes*. 176 admitted in 311 ms, none finishing, is what reached 23.6 GB |
| `queryCoord.loadTimeoutSeconds` | 600 | 36000 | The observer cancels a load that makes no segment-level progress inside the window |
| `queryCoord.segmentTaskTimeout` | 120 s | 3600 s | One segment load off this disk was measured at 14m26s; past the deadline the read dies with `context canceled` |
| `queryCoord.channelTaskTimeout` | 60 s | 600 s | Same problem, channel subscription |
| `queryCoord.overloadedMemoryThresholdPercentage` | 90 | 95 | Margin only; it fixes nothing on its own |

**Size the Docker VM for the search step, not the load step.** Loading only
needs the VM; searching needs the VM *and* BGE-M3 on the host. At 28 GB of a
36 GB host the two did not fit — the first search after a completed load
exhausted swap (25.4 GB of 26.6 GB) and macOS killed Docker Desktop. 22 GB
leaves the guard 4.8 GB over the measured peak (`RssAnon` 11.3 GB +
`committedMemSize` 4.8 GB) and the host 14 GB. A smaller VM also means less
page cache, which strengthens the HDD treatment rather than weakening it.

Two things that look like fixes and are not. Dropping the VM page cache does
nothing — `GetUsedMemoryCount` reads `RSS - Shared` from statm, so file-backed
pages are excluded by construction. Setting `mem_limit` makes it worse: Milvus
takes `min(cgroup, physical)`, so a limit can only lower the ceiling.

`DISKANN` is also not affected by the `mmap` flags — the index is always pulled
to local disk and its PQ codes always held in RAM. The flags matter for the
scalar payload, which is what they were added for.

### Prefetching the next retrieval

```bash
uv run wikipedia-prefetch \
  --bundle-dir ~/.cache/wikipedia_diskann \
  --episodes 4 --decode-seconds 2.0 --think-seconds 0.4 --wrong-hops 2
```

Runs each multi-hop episode twice — plain, then with a drafter speculating on
the next query — and compares. Alternates which arm goes first per episode, so
neither inherits the page cache the other warmed.

The two arms must return byte-identical passages; the report says `DIVERGENCE`
if they do not, which is a bug signal rather than a result. Prefetch is used
only on an exact normalized match, so it changes when a search happens and never
what it returns.

Without `--draft-url` the drafter is a perfect replay of the episode. That is a
**ceiling for the mechanism, not a measurement of a drafter** — it hits every
hop by construction. `--wrong-hops` forces misses so the cost of a wasted
speculative search on a busy disk shows up, and `--think-seconds` charges
prediction the time a real model would spend. With a server available:

```bash
uv run wikipedia-prefetch --draft-url http://localhost:8001 \
  --draft-model llama-3.2-1b-instruct
```

`--decode-seconds` stands in for the target model decoding the next thought. It
is the window a prefetch has to work in, so it bounds the result and is printed
with it. Report it; do not bury it.

### Python search API

```python
from wikipedia import Encoder, MilvusConfig, MilvusVectorDB, search_text

encoder = Encoder()  # Eagerly load pinned local BGE-M3 before serving queries.
with MilvusVectorDB(MilvusConfig(uri="http://localhost:19530")) as db:
    db.ensure_collection()
    db.load()  # Reads the on-disk index; do this before timing anything.
    vector_hits = db.search([0.0] * 1024, limit=5)
    text_hits = search_text(db, encoder, "What causes auroras?", limit=5)
```

`Encoder` supports pinned local BGE-M3 only and loads it eagerly, making model startup
explicit instead of adding cold-start work to the first text query. Ingestion and
vector-only search never load the encoder. Dataset embeddings are inserted unchanged.
No filters, reranking, sparse, or hybrid search are included.

`QdrantConfig` / `QdrantVectorDB` have the same shape and remain importable.

### Known limits

- **`insert` is not idempotent.** Ingest writes with `insert`, because Milvus
  implements `upsert` as delete + insert and a bulk load into a fresh collection
  then spends the disk compacting one tombstone per record — measured at 18 GB
  for 200k records and 680 MB/min of pure compaction. The cost is that resuming
  a partial ingest duplicates rows. Either wipe the volumes and checkpoints and
  start over, or resume with `--milvus-upsert`.
- **Milvus caps VARCHAR at 65535 bytes** and has no larger string type. A longer
  chunk fails the ingest by default. `--milvus-truncate-text` stores it cut to
  fit on a UTF-8 boundary and reports how many were truncated; embeddings are
  unaffected, so ranking does not change but the returned payload does. Two of
  1,000,000 records hit this.
- **Milvus requires NVMe SSD for DISKANN.** A USB HDD is outside the supported
  range. It works, but published figures will not reproduce — say so when
  reporting results.
- **Index state needs `indexed_rows`, not `state`.** Milvus 2.5.27 reports
  `state=Finished` with `indexed_rows=0` before any segment is built, and
  reports `pending_index_rows` equal to the total once everything is built.
  `wait_for_index()` keys on `indexed_rows >= total_rows`, confirmed twice.
- **Small ingests never build an index.** Milvus only indexes sealed segments,
  so a load that does not fill one stays searchable by brute force alone and
  DiskANN never engages. 200k records were not enough; 600k were.
- **etcd fsyncs on every write** and lives on the HDD. It has held so far; if it
  destabilizes at 10M, move only the etcd volume to internal SSD — that data is
  small and off the query path.
- **MinIO stores the source binlogs on top of the index**, roughly 21 GB per 1M
  records. Budget it separately from the index.
