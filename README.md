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
