# decode-RAG Speculative Prefetch — 기획안 (Step 4)

## 배경 / 목적
지금까지 측정으로 **검색(retrieval)이 e2e의 ~30%**, hop별 `retrieval ≈ decode`(rerank 후 **1:1**, 중앙값 0.23→1.08)임을 확인했다 → **검색을 decode 뒤에 숨길 여지가 충분**.

목표: 주 LLM(Llama-3.1-8B)이 step N을 decode하는 동안 **작고 빠른 보조 LLM(drafter, 별도 `:8001` llama-server)이 다음 검색어를 예측해 retrieval을 미리 실행** → step N+1 검색 시 캐시에서 즉시 반환 → 검색 대기를 숨긴다.

**drafter 모델은 고정하지 않고 Stage 1에서 후보 스윕으로 정한다**(2026-07-31 결정, 종전 Llama-3.2-1B 고정에서 변경). 근거: ① M3 Pro는 대역폭 바운드라 보조 모델 크기가 경합세에 직결, ② drafter 비용의 대부분은 짧은 생성이 아니라 **긴 ReAct prefix의 prefill**이라 파라미터 수에 거의 선형, ③ Stage 1은 오프라인이라 후보를 여러 개 돌려도 Cohere 쿼터·GPU 동시성 비용이 0.

**무결성 원칙(절대 규칙)**: prefetch는 **예측 쿼리가 실제 다음 쿼리와 정규화 후 정확히 일치할 때만** 사용. 불일치(miss)면 기존 경로 그대로 실행 → 반환 passage가 baseline과 **동일** → **정확도(EM/F1) 수학적 불변**. prefetch는 "언제 검색하느냐"만 바꾸고 "무엇을 받느냐"는 절대 안 바꾼다.

## 현재 코드 구조 (검증됨)
- ReAct 루프 = LangGraph 상태머신 `AgentBench/src/agents/ReAct/react.py`: `call_model`(LLM 스트리밍) → `execute_tool`(`Action: search[q]` 파싱 후 `tool.invoke()`). 전부 동기·블로킹. `run_react.py::run_single_query`가 구동, 워커는 `repro/sweep/agent_runner.py`의 `ThreadPoolExecutor`(context 복사).
- 검색 = `AgentBench/src/tools/hotpotqa_tools/vector_search.py::VectorSearchTool._run`(Cohere embed → FAISS → bge-reranker).
- 측정 = `repro/measurement/chat_wrapper.py::TraceCallbackHandler`(현재 검색 쿼리 문자열·결과 미저장).
- 서빙 = `:8000` 단일 인스턴스 `--parallel 1`(`repro/setup/start_server.sh`, 8B Q4_K_M 4.9GB). 보조 모델은 `:8001` 별도 서버. M3 Pro 36GB, 여유 ~25GB(대역폭 바운드 — 2모델 동시 시 주 decode가 느려질 수 있어 측정 필요).
- 검색 쿼리 문자열은 **기존 런에 남아 있지 않음**(확인: `results/raw/react_vectordb_rerank.jsonl`의 tool span 키 = `t_start`/`t_end`/`tool_name`/`error`) → Stage 1은 반드시 Stage 0 로깅을 켠 **신규 배치**를 전제로 한다.

---

## Stage 0 — Trace 로깅 (동작 변화 없음, Stage 1 잠금 해제)
검색 쿼리 문자열·prefetch 메타를 기록(전부 기본값 있는 optional → 기존 JSONL 하위호환).
- `repro/measurement/trace_schema.py`: `ToolCallSpan`에 `tool_input`, `run_id`, `result_id`(=`sha1(normalize(q))[:16]`, 본문 미저장), `prefetch_hit`, `predicted_query` 추가. `QueryTrace`에 `prefetch_hits/attempts`, `draft_predict_ms_total`.
- `repro/measurement/chat_wrapper.py`: `on_tool_start`에서 `input_str`→`tool_input` 저장, tool span 매칭을 **`run_id` 기준**(LLM span 방식)으로, 실패 시 기존 역순 폴백.
- **검증**: `pytest repro/tests/measurement/` 그린 + 구버전 JSONL 행이 `model_validate` 되는지(하위호환 가드).

## Stage 1 — 오프라인 타당성 (병렬 시스템 없음, 의사결정 게이트)
**전제**: Stage 0 로깅 켠 채 ~50–100쿼리 1배치(= 기존 `react_vectordb_rerank.yaml`, Cohere 키 의존).

> **쿼터 회계**: 현재 트라이얼 키는 월 1,000회. 관측된 tool calls/q ≈ 5.5 → **100쿼리 ≈ 550회**로 한 달 한도 안에 들어간다(139쿼리에서 막힌 이전 런이 ~765회). 즉 **Stage 1 배치는 월초에 트라이얼 키로 실행 가능**하며 production 키는 Stage 3(쌍대 런 + prefetch miss분 추가 embed)부터 필요하다.
- 신규 `repro/analysis/prefetch_feasibility.py`:
  - **(a) 숨김 상한**: `plot_hop_balance.py::load_hops` 재사용. hop별 `hidden_ceiling = min(retrieval_ms(N+1), decode_ms(step N))`. 쿼리별 합/e2e → 이상적 회수 가능 비율 + 분산.
  - **(b) 다음-쿼리 예측 가능성(게이트 지표)**: step N 직후 prefix를 작은 모델에 주고 다음 `search[...]` 예측 → 실제 로그 쿼리와 비교. **exact**(정규화 후=캐시 키와 동일=실제 적중률), fuzzy(rapidfuzz≥90), semantic(임베딩 코사인, opt-in).
  - **(c) drafter 후보 스윕(신규)**: (b)를 **모델별로 반복**. 같은 로그·같은 prefix에 대해 후보를 순차 실행하므로 GPU 동시성·Cohere 호출 없이 오프라인으로 끝난다. 후보(전부 Q4_K_M GGUF, `repro/models/`):
    | 후보 | 파라미터 | Q4 크기 | 비고 |
    |---|---|---|---|
    | `Qwen2.5-0.5B-Instruct` | 0.5B | ~0.4GB | 1순위 — 최소 경합세 |
    | `SmolLM2-360M-Instruct` | 0.36B | ~0.3GB | 하한 탐침, 포맷 추종 실패 가능성 높음 |
    | `Llama-3.2-1B-Instruct` | 1B | ~0.8GB | 상한/기준선(종전 계획값) |
    - 후보별 기록: `exact_hit_rate`, `predict_ms`(= prefix prefill + 생성, prefix 길이별로 분해), `format_valid_rate`(`Action: search[...]` 파싱 성공률).
    - **선정 기준**: `exact_hit_rate ≥ 0.4`를 만족하는 후보 중 `predict_ms` 최소. 어떤 후보도 0.4를 못 넘으면 Stage 1에서 중단.
    - **주의**: 소형 모델은 hit rate가 아니라 **포맷 추종**에서 먼저 무너질 수 있다. `format_valid_rate`가 낮으면 few-shot 프롬프트 1회 보강 후 재측정하고, 그래도 낮으면 그 후보는 탈락(프롬프트 튜닝 루프에 시간을 쓰지 않는다).
  - 출력 `results/figures/prefetch/`: `predictability.json`(후보별 배열), `ceiling.json`, `hit_rate_hist.png`, `ceiling_vs_hitrate.png`, `drafter_tradeoff.png`(x=`predict_ms`, y=`exact_hit_rate`).
- **의사결정 게이트**: 기대 절감 ≈ `retrieval비율(~0.30) × exact_hit_rate × 중첩효율 − contention_tax`. **exact_hit_rate ≥ ~0.4 미만이면 여기서 중단**(음성 결과도 유효한 측정).

## Stage 2 — 라이브 프로토타입 (config-gated, 끄면 baseline byte-for-byte 동일)
- **보조 서버**: `repro/setup/start_draft_server.sh`(= start_server.sh + `--port 8001 -c 8192`). **모델 경로는 하드코딩하지 않고 `DRAFT_MODEL` 환경변수/인자로 받는다**(Stage 1 스윕 우승 모델을 그대로 주입). 다운 시 prefetch는 무조건 miss로 graceful degrade.
- **Config**: `repro/sweep/configs/react_vectordb_prefetch.yaml`(= rerank + `prefetch_enabled: true`, `draft_base_url`, `draft_model`, `prefetch_match: exact`). `prefetch_enabled` 없으면 새 경로 미실행 = baseline 불변 보장.
- **prefetch 모듈**: `AgentBench/src/agents/ReAct/prefetch.py` — `PrefetchCache`(Lock 보호, **쿼리당 새로 생성**→교차오염 불가, contextvar `PREFETCH`), `normalize_query()`(캐시 writer·tool lookup 단일 진실원), `spawn_prefetch()`(daemon 스레드: `:8001` 예측→`extract_tool_calls` 파싱→`search_tool._run`→캐시 저장; 전부 try/except).
- **트리거 seam(채택)**: `react.py::call_model` 끝에서 step N+1용 prefetch 스레드 spawn(가드: 비활성 시 no-op). Stage 1 측정과 정확히 대응, 한 곳 국소 삽입. (대안 `on_llm_new_token` 토큰 seam은 핸들러 부재·baseline 위험으로 기각.)
- **캐시-적중 경로**: `vector_search.py::_run` 맨 앞에서 `cache.get(normalize(text))` 확인 → 적중 시 캐시 반환 + `prefetch_hit=True`; miss 시 기존 경로(=baseline 동일 passage). rerank(MPS)는 `_rerank`에 `threading.Lock` 직렬화.
- **검증**: (a) 정확도 불변 테스트(`repro/tests/integration/test_prefetch_correctness.py`: 어떤 예측이든 `_run` 반환 = baseline). (b) 게이트-off 테스트. (c) 양 서버 2–3쿼리 스모크(`prefetch_attempts>0`·무에러).

## Stage 3 — 정직한 측정
- **두 런(쌍대, sample_idx 1:1)**: baseline `react_vectordb_rerank.jsonl` ↔ treatment `react_vectordb_prefetch.jsonl`(같은 seed=42).
- **지표**: e2e 지연 Δ(Wilcoxon+bootstrap), 숨긴 retrieval(`Σ min(retrieval_ms, 중첩 decode_ms)`), hit-rate(=Stage1 예측 대조), **정확도 불변**(EM/F1 일치, 불일치 시 버그 신호).
- **경합세(contention tax)**: `:8000`·`:8001` 양쪽 decode tok/s 폴링(MetricsCollector 임의 base_url 지원). `repro/analysis/contention_tax.py`: 주모델 tok/s를 보조 idle vs 부하 비교 → `tax = 1 − loaded/idle`. **순절감 = 숨긴 retrieval − 경합세**. tax가 더 크면 net-negative(그대로 보고).
- **플롯**: `plot_hop_balance.py`에 `fig_prefetch` + 신규 `repro/analysis/plot_prefetch_summary.py`(헤드라인 4숫자: hit-rate·숨긴 retrieval·경합세·순 e2e 절감).

---

## 위험 & 결정 포인트
- **예측 정확도 임계(최우선 게이트)**: retrieval이 e2e의 ~30%라 hit 50%여도 tax 전 ~15% 회수. **exact_hit_rate ≥ ~0.4 floor**, 미달 시 Stage 1에서 중단.
- **2-모델 대역폭 경합세**: M3 Pro(대역폭 바운드)의 핵심 위험 — 반드시 측정(§Stage3). drafter를 0.5B급까지 줄이려는 이유가 이것이다. 단 **경합세가 0이 되지는 않는다**: 모델 가중치가 작아도 prefill은 연산 바운드라 GPU를 점유한다. "작게 만들면 tax가 사라진다"고 가정하지 말고 후보별로 측정한다.
- **MPS 부재(신규)**: 2모델 동시 실행 시 GPU 컨텍스트 스위칭을 내부적으로 조율해 주는 NVIDIA **MPS(Multi-Process Service)** 는 **CUDA 전용이며 Apple Metal에는 대응물이 없다.** 즉 본 재현 환경에서는 스케줄링 개선 여지가 없고, 경합세는 완화 대상이 아니라 **측정·보고 대상**이다. (논문 환경인 A100에서는 MPS/MIG로 이 항목이 달라질 수 있다는 점을 결과 해석에 명시.)
- **캐시 staleness/정확성**: 쿼리당 캐시 + exact-match-only + miss 폴백으로 구조적 차단. Stage 2 정확도 테스트가 증명.
- **Cohere 쿼터**: Stage 1·3 모두 production 키 소모. prefetch는 miss여도 embed해 쿼터 더 씀(`prefetch_attempts`로 회계).

---

## 부록 A — 검색 백엔드 이관(외장 디스크 트랙, 병행)
2026-07-31 결정: 코퍼스를 줄이는 대신 **외장 디스크를 붙여 Upstash BGE-M3 전량을 적재**한다. 본 prefetch 트랙과 **의존 관계 없이 병행**하며(Stage 0~2는 기존 Cohere+FAISS 경로로 진행 가능), Stage 3 전까지 준비되면 갈아탄다.

- **용량 산정**(47,018,430 × 1024-d): 벡터 fp32 **~193GB** / fp16 **~96GB**, HNSW 그래프 ~6GB, payload(본문 텍스트) ~25–40GB, 여기에 적재 중 raw Parquet 스테이징(`--max-workers` 개 샤드 분). → **`--float16` 사용 시 실사용 ~130–145GB**, 디스크는 **500GB 이상(1TB 권장)**.
- **포맷 제약(주의)**: Qdrant 스토리지는 블록 단위 접근이 되는 POSIX 파일시스템만 허용 — **APFS/ext4 가능, ExFAT·NTFS·네트워크 FS는 시작 전 거부됨.** 외장 SSD는 공장 출하 시 대부분 ExFAT이므로 **APFS로 재포맷 필요**.
- **명령**: `uv run wikipedia-ingest qdrant --bundle-dir /Volumes/<disk>/wikipedia --float16`(중단 시 같은 명령으로 재개, `--float16`은 재개 시에도 반복 지정).
- **연동 시 후속 과제**: 쿼리 인코딩이 Cohere API(네트워크) → BGE-M3 로컬로 바뀌면 **retrieval-time fraction의 정의가 달라진다**(네트워크 왕복 소멸 → fraction 급락). 갈아탈 때 `RELATED_WORK.md` §"우리 프로젝트의 함정" 의 재정의안을 반영하고, **baseline/treatment는 반드시 같은 백엔드끼리만 비교**한다.

---

## 빌드/테스트 순서
1. Stage 0 스키마+핸들러+테스트(서버·Cohere·디스크 불필요, 즉시) → 2. drafter 후보 GGUF 3종 다운로드(~1.5GB, 오프라인) → 3. Stage 1 로깅 배치(Cohere 트라이얼, 월초) → 4. Stage 1 분석+후보 스윕+게이트 → 5. Stage 2 인프라+테스트(목으로 지금 단위테스트) → 6. Stage 3 paired 런+경합세 벤치+플롯(Cohere production 키 + 양 서버).
- 부록 A(외장 디스크 적재)는 1~5와 **병행**, 6 전까지 완료 목표.

## 수정/생성 핵심 파일
- 수정: `repro/measurement/trace_schema.py`, `repro/measurement/chat_wrapper.py`, `AgentBench/src/agents/ReAct/react.py`, `AgentBench/src/tools/hotpotqa_tools/vector_search.py`, `AgentBench/run_react.py`, `repro/analysis/plot_hop_balance.py`, `repro/setup/download_model.sh`(drafter 후보 3종 추가)
- 신규: `AgentBench/src/agents/ReAct/prefetch.py`, `repro/analysis/prefetch_feasibility.py`, `repro/analysis/contention_tax.py`, `repro/analysis/plot_prefetch_summary.py`, `repro/setup/start_draft_server.sh`, `repro/sweep/configs/react_vectordb_prefetch.yaml`, 테스트 3종
