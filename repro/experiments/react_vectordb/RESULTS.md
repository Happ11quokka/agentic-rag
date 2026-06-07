# 실행 결과 — ReAct × Cohere 벡터DB

**작성일**: 2026-06-07 · **브랜치**: `experiment/react-vectordb`
개요는 `README.md`, 결정은 `DECISIONS.md`, 계획은 `PLAN.md`.

---

## 1. decode-RAG baseline (최종: 풀 415-샤드 인덱스, 50 샘플)

`results/raw/react_vectordb.jsonl` · 50/50 clean, timeout·error 0.
인덱스: **880,777 passages · 60,682 titles · gold 커버리지 82.9% (11,421/13,783)**.

| 지표 | 값 |
|---|---|
| **decode-RAG 비율 (tool/e2e)** | **median 9.7% · mean 11.1%** (범위 5.0–29.4%) |
| 검색 횟수/쿼리 | mean 4.96 (median 3) — multi-hop |
| 검색당 시간 | ~415 ms (Cohere API 네트워크 + faiss 880K) |
| 지연 분해 (median) | e2e ~10.5s (tool 1.2s / LLM 9.5s) |
| EM 정확도 | **36.0% (18/50)** |

상관: 정답 쿼리는 평균 **3.1회** 검색, 오답은 **6.0회** (근거 못 찾으면 iteration_limit까지 헤맴). 비율 자체는 정답/오답 무관(~11%).

## 2. 부분빌드(40샤드) 대조 + 비율 해석 정정

| | 40샤드 (cov 7.2%, 12q) | 415샤드 (cov 82.9%, 50q) |
|---|---|---|
| EM 정확도 | 8.3% | **36.0%** |
| decode-RAG 비율 | 6.5% | **~10%** (med 9.7%) |
| 검색/쿼리 | 10.7 | 4.96 |
| e2e median | ~51s | ~10.5s |

**정정**: 초기엔 "비율이 커버리지에 거의 무관(~6.5%)"으로 추정했으나, 실측은 **풀 코퍼스에서 비율이 오름(~10%)**.
이유 — 커버리지↑ → 적은 hop으로 정답 도달(검색 10.7→5.0) → 대화 컨텍스트가 짧아져 **LLM 호출당 시간↓**,
동시에 인덱스가 커져 **검색당 시간↑**(305→415ms). 둘 다 retrieval 비중을 키움. 즉 **현실적 코퍼스일수록
숨길 지연(retrieval fraction)이 더 크다** → decode-RAG 동기에 유리한 결과.

## 3. 구현 중 확인된 사실 (계획 대비 정정)

- **샤드 수 415개** (en split, `en/0000.parquet`~`0414.parquet`, 각 ~217MB). DECISIONS 부록의 "50샤드"는 오류.
- **Cohere 키는 `cohere_` 접두어 포함 전체**가 키 (`COHERE_API_KEY=cohere_…`). 접두어 빼면 401.
- **libomp 충돌**: faiss + 에이전트 스택 동시 로드 시 `OMP Error #15`로 abort →
  실행 환경에 **`KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` 필수**.
- **스트리밍/읽기는 디스크에 캐시 안 함** (신규 샤드 1개 읽어도 HF 캐시 증가 0). 산출물만 디스크 사용(415샤드: index 3.4GB + _shards 1.6GB + meta 0.24GB).
- **동시 다운로드는 무의미**: 네트워크 총 대역폭(~6.2MB/s)이 병목 → 풀 빌드 4.2h, 실패 샤드 0.
- 패키지: faiss-cpu 1.14.2 / cohere 7.0.3 / datasets 5.0.0 / pyarrow 24.0.0 (arm64·py3.13 wheel OK).
- 응답 형태: `co.embed(...).embeddings.float_` (방어적으로 `float_`→`float`→list 순 접근).
- 측정 경로: `search`가 표준 `BaseTool.invoke`라 `TraceCallbackHandler`가 tool span 자동 캡처(Cohere+faiss 시간 포함). sweep 코드 무수정(D10 확인).

## 4. 재현 명령

```bash
cd /Users/imdonghyeon/agentic_rag/repro && source .venv/bin/activate

# (1) 인덱스 빌드 — 재개 가능(샤드 단위), 디스크 안전, 메모리 안전 증분 assemble
python retrieval/build_cohere_hotpot_index.py            # 전체 415샤드 (~4h)
#   --max-shards 40   부분(검증)   |   --rebuild-index   체크포인트만 재조립   |   --workers N

# (2) sweep — libomp 환경변수 + OpenAI 로컬 라우팅 필수. llama-server 먼저 가동.
./setup/start_server.sh &
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=sk-dummy-local \
python -m sweep.sweep_runner --config sweep/configs/react_vectordb.yaml \
    --out results/raw/react_vectordb.jsonl
#   (COHERE_API_KEY는 AgentBench/.env에서 load_dotenv로 자동 주입)

# (3) 집계
python analysis/decode_rag_ratio.py results/raw/react_vectordb.jsonl
```

## 5. 산출물

- 인덱스: `retrieval/index/cohere_hotpot.faiss` (3.4GB) + `cohere_hotpot_meta.parquet` (237MB) — gitignore됨
- 결과: `results/raw/react_vectordb.jsonl` (50q, 풀), `results/raw/react_vectordb_smoke.jsonl` (12q, 40샤드)
- 신규 코드(모두 `AgentBench/`는 gitignore라 미추적): `vector_search.py`, `hotpotqa_vectordb.py`, `run_react.py` 분기
- repro/ 추적 변경: `retrieval/build_cohere_hotpot_index.py`, `sweep/configs/react_vectordb*.yaml`, `analysis/decode_rag_ratio.py`, `.gitignore`, 본 실험 문서
