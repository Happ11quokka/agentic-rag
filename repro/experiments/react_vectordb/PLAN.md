# 구현 계획 — ReAct × Cohere 벡터DB

> 이 문서는 **구현 계획**이다. 실행은 사용자가 별도로 지시할 때 시작한다.
> 개요/목적은 `README.md`, 결정 근거는 `DECISIONS.md` 참고.

## 아키텍처

```
[오프라인 1회]  Cohere en 스트리밍 ──filter(title ∈ HotpotQA)──▶ passages(text, title, emb)
                 │
                 ├─▶ FAISS IndexFlatIP   →  repro/retrieval/index/cohere_hotpot.faiss
                 └─▶ 메타데이터(정렬됨)   →  repro/retrieval/index/cohere_hotpot_meta.parquet

[런타임/sweep]  ReAct ──search[q]──▶ Cohere API embed(q) ──▶ FAISS top-k ──▶ top-k passages
                (LLM = 로컬 llama-server)     (네트워크)         (로컬)        └─ SystemMessage Observation

                tool span = embed(API) + FAISS  →  tool_total_ms  →  decode-RAG 비율
```

- **문서 벡터**: Cohere 사전계산(`emb`) 그대로. 재임베딩 없음.
- **쿼리 벡터**: 런타임에 Cohere API(`input_type="search_query"`)로 생성. 문서와 동일 모델이어야 같은 벡터공간에서 비교 가능.
- **측정**: 기존 `TraceCallbackHandler`가 `search`(표준 `BaseTool.invoke` 경로)를 자동 캡처 → tool 호출 수·시간이 그대로 기록됨.

## 파일 — 생성 / 수정

**신규**
- `repro/retrieval/build_cohere_hotpot_index.py` — 오프라인 인덱스 빌드 (스트리밍·필터·FAISS·커버리지 리포트)
- `AgentBench/src/tools/hotpotqa_tools/vector_search.py` — 싱글톤 retriever(Cohere 임베딩 + FAISS) + `VectorSearchTool(BaseTool, name="search")`
- `AgentBench/src/agents/ReAct/prompt/hotpotqa_vectordb.py` — passage 스타일 few-shot `get_system_prompt(fewshots)`
- `repro/sweep/configs/react_vectordb.yaml` — sweep config

**수정**
- `AgentBench/run_react.py::run_single_query` — `agent_kwargs["retrieval_backend"]` 분기 (아래 패턴)
- `repro/pyproject.toml` — deps 추가: `faiss-cpu`, `cohere`, `datasets` (`numpy`는 기존)
- `AgentBench/.env` (또는 셸 export) — `COHERE_API_KEY`

**재사용 (변경 없음)**
- `AgentBench/src/agents/ReAct/react.py::create_react_agent` — `search`가 표준 `BaseTool.invoke` 경로라 tool 호출/시간 자동 측정. 관찰값은 기존대로 `SystemMessage`로 주입(비교가능성 위해 유지).
- `repro/sweep/{sweep_runner,agent_runner,cells}.py` — **변경 불필요**. config `defaults`의 임의 필드가 `extra_kwargs` → `agent_kwargs`로 전달됨 (`cells.py:69-71`, `agent_runner.py:219-225`).
- `repro/measurement/*` (TraceCallbackHandler, `eval.hotpotqa_em`, trace_schema) — 그대로.
- `repro/sweep/run_full.sh`, `repro/setup/start_server.sh` — 그대로.

## 구현 단계

### 1. 의존성
`repro/pyproject.toml`의 `dependencies`에 `faiss-cpu`, `cohere`, `datasets` 추가 → `repro/.venv`에서 `pip install -e .`.
(arm64-mac / py3.13 wheel 가능 여부는 설치 시 확인 — 리스크 참고.)

### 2. 인덱스 빌드 — `repro/retrieval/build_cohere_hotpot_index.py`
1. **title 집합**: 디스크의 `AgentBench/dataset/hotpot_dev_fullwiki_v1.json`에서 모든 `supporting_facts` title(=gold) ∪ 모든 `context` title 수집 → 정규화(lowercase/strip/unicode NFC). gold title은 따로 보관(커버리지 측정용). (≈ 6.6만+ unique title)
2. **스트리밍 필터**: `datasets.load_dataset(..., "en", split="train", streaming=True)`로 영어 split 순회(≈90GB 전송, 저장은 매칭분만). `norm(row["title"]) ∈ titles`인 행의 `(_id, title, text, emb)` 수집. 진행률(스캔/매칭 수) 로그.
3. **FAISS**: `embs = np.asarray(emb, np.float32)` 적재 → `faiss.normalize_L2(embs)` → `IndexFlatIP(1024)` add → `index/cohere_hotpot.faiss` 저장. 메타데이터(인덱스 순서와 정렬된 `{id, title, text}`)는 `index/cohere_hotpot_meta.parquet`.
4. **리포트**: 매칭 passage 수, 매칭 unique title 수, **gold-title 커버리지**(매칭된 supporting_facts title 비율) 출력.
5. **idempotent**: 인덱스 존재 시 skip.

### 3. retriever + 도구 — `AgentBench/src/tools/hotpotqa_tools/vector_search.py`
```python
# 프로세스당 1회 로드되는 싱글톤 (sweep는 한 프로세스에서 다수 쿼리 실행)
_CACHE = {}
def get_retriever(index_path, meta_path, model="embed-multilingual-v3.0", top_k=5):
    key = (index_path, model)
    if key not in _CACHE:
        import faiss, cohere, os, pandas as pd
        idx  = faiss.read_index(index_path)
        meta = pd.read_parquet(meta_path).to_dict("records")
        co   = cohere.Client(os.environ["COHERE_API_KEY"])
        _CACHE[key] = (idx, meta, co, model, top_k)
    return _CACHE[key]

class VectorSearchTool(BaseTool):
    name: str = "search"
    description: str = "Search Wikipedia passages by meaning"
    # _run(query):
    #   v = co.embed([query], model, input_type="search_query").embeddings[0]
    #   faiss.normalize_L2(v) → idx.search(v, top_k)
    #   → top-k passages를 "Title: {title}\n{text}" 형식 문자열로 (text는 ~N자 절단)
```
- `make_vector_search_tool(index_path, meta_path, model, top_k)` 헬퍼가 싱글톤 초기화 + 도구 인스턴스 반환.

### 4. few-shot — `AgentBench/src/agents/ReAct/prompt/hotpotqa_vectordb.py`
- 원본 `hotpotqa.py` 복제·수정: 지시문은 "Observations are Wikipedia passages…" 유지, **`lookup[...]` 예시와 "Could not find X. Similar:[...]" 예시 제거**, `search[query] → passages → finish[answer]` 흐름의 예시로 교체.

### 5. `run_react.py::run_single_query` 분기
```python
backend = agent_kwargs.get("retrieval_backend", "wikipedia")
if backend in ("vectordb", "cohere_faiss"):
    from src.tools.hotpotqa_tools.vector_search import make_vector_search_tool
    from src.tools.hotpotqa_tools.wikipedia import FinishTool
    from src.agents.ReAct.prompt.hotpotqa_vectordb import get_system_prompt
    search = make_vector_search_tool(
        index_path=agent_kwargs["index_path"], meta_path=agent_kwargs["meta_path"],
        model=agent_kwargs.get("embed_model", "embed-multilingual-v3.0"),
        top_k=int(agent_kwargs.get("top_k", 5)))
    tools = [search, FinishTool(name="finish")]
    system_prompt = get_system_prompt(fewshots=min(fewshot, 5))
else:
    # 기존 경로(Wikipedia) 그대로
```
(LLM 생성·stream 루프·`parse_answer` 등 나머지는 동일.)

### 6. sweep config — `repro/sweep/configs/react_vectordb.yaml`
```yaml
run_id: react_vectordb
workload: hotpotqa
agent_type: react
defaults:
  fewshot: 5
  iteration_limit: 30
  retrieval_backend: vectordb
  embed_model: embed-multilingual-v3.0
  index_path: retrieval/index/cohere_hotpot.faiss     # cwd=repro 기준
  meta_path: retrieval/index/cohere_hotpot_meta.parquet
  top_k: 5
sweeps:
  iteration_limit: [30]      # cells.py가 정확히 1개 sweep var 요구 → 단일값
samples_per_cell: 50
sample_seed: 42
```

### 7. 실행
```bash
cd /Users/imdonghyeon/agentic_rag/repro && source .venv/bin/activate
export COHERE_API_KEY=...                  # 전제조건
python retrieval/build_cohere_hotpot_index.py     # 1회 (스트리밍, 수 시간)
./setup/start_server.sh &                   # llama-server
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=sk-dummy-local
python -m sweep.sweep_runner --config sweep/configs/react_vectordb.yaml --out results/raw/react_vectordb.jsonl
```

## 검증 (end-to-end)

1. **빌드 검증**: 리포트의 gold-title 커버리지 확인(예: ≥70%). 인덱스/메타 파일 생성 확인.
2. **스모크(1쿼리)**: `retrieval_backend=vectordb`로 ReAct 1건 → `search`가 Cohere API 호출 + 관련 passage 반환 + `finish`로 종료하는지 로그 확인.
3. **소규모 sweep(50샘플)**: `react_vectordb.jsonl` 생성 후 —
   - `n_tool_calls` 평균 > 1 (multi-hop 검색)
   - **decode-RAG 비율 = `tool_total_ms / (e2e_latency_s*1000)`** 산출 (목표 >2%, Cohere API 네트워크로 확보)
   - `correct`(정확도) — 커버리지 범위 내 의미있는 값
4. **baseline 대조**: 기존 fig13 ReAct(≈32% acc / 41s, tool 44%는 라이브 위키)와 비교 — tool 시간 구성/비율·e2e 변화 관찰.

## 리스크 · 주의

- **패키지 wheel**: `faiss-cpu`/`cohere`/`datasets`의 arm64-mac·py3.13 wheel 설치 가능 여부 확인. faiss 불가 시 대안 `hnswlib` 또는 numpy matmul(부분집합이 작아 가능).
- **~90GB 스트리밍**: 빌드 1회에 영어 50샤드 전송(수 시간, 네트워크 필요). 저장은 매칭분(수백 MB~1GB)만.
- **Cohere rate limit**: 빌드엔 doc 임베딩 불필요(사전계산 사용). 런타임 쿼리 임베딩만(50문항×~5 ≈ 250콜) → 트라이얼로 충분. 쿼리→벡터 캐시 추가 가능.
- **title drift (2017↔2023)**: 일부 gold 미매칭 → 해당 질문 풀 수 없음. 커버리지 리포트로 정량화·보고.
- **비로컬**: 쿼리 임베딩이 원격 → 완전 로컬 아님(의도된 선택: 비율 확보). 재현성은 라이브 위키보다 개선되나 네트워크 변동 잔존.
- **L4-b(토큰 분류)**: tool 관찰 토큰은 `SystemMessage`라 `tokens_by_role`에서 'tool' 미분류 — 기존과 동일 유지(행동·비교가능성 보존).
- **emb dtype**: parquet의 `emb`가 fp16일 수 있음 → 적재 시 fp32 변환.
