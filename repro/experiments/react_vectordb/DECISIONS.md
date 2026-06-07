# 설계 결정 기록 (Decision Log)

이 실험을 설계하며 합의한 결정과 그 근거. 각 항목: **결정 / 근거 / 대안·기각 이유**.
조사 중 확인한 수치는 [부록: 확인된 사실]에 정리.

---

## D1. 에이전트는 ReAct만
- **결정**: ReAct만 실행. LATS/Reflexion/LLMCompiler 코드는 **삭제하지 않고 유지**(비실행).
- **근거**: 이번 목표는 4종 비교가 아니라 ReAct 단일 에이전트의 RAG baseline. 코드 삭제는 침습적·위험 → config로 ReAct만 돌리면 충분.
- **부수효과**: ReAct만 쓰면 발표 한계 **L4**(Reflexion/LATS/LLMCompiler가 자체 wrapper로 콜백 우회 → tool 측정 0, 사후복구 필요)와 **latency 중복 attribution**(폴링 동시구간) 문제를 자동 회피. ReAct는 표준 `BaseTool.invoke` 경로라 정상 측정됨(`react.py:97`).

## D2. 검색을 라이브 Wikipedia → 실제 dense 벡터DB
- **결정**: 검색 도구를 FAISS dense 검색으로 교체.
- **근거**: 논문/AgentBench의 ReAct는 **라이브 Wikipedia API**(키워드)만 씀 — 고정 코퍼스·벡터DB·임베딩 전혀 없음. 이게 발표 한계 **L5**(라이브 호출 → 4개월간 내용 변동, 재현 불가)의 원인. 고정 벡터DB로 재현성↑ + "실제 RAG(dense)" 시나리오 확보.
- **대안**: 라이브 위키 유지 → 재현성 문제 지속, 또 dense RAG가 아님 → 기각.

## D3. 문서 임베딩은 Cohere 사전계산 그대로 (재임베딩 X) — 그리고 "문서 vs 쿼리" 구분
- **결정**: Cohere `wikipedia-2023-11-embed-multilingual-v3`(영어)의 사전계산 `emb`(1024d) 사용. 문서는 다시 임베딩하지 않음.
- **핵심 통찰(혼란의 근원이었음)**: 임베딩은 **두 종류**다.
  - **문서 임베딩** = 위키 문단 4,148만 개 → 이미 Cohere가 해놓음(`emb`). 우리가 다시 안 함.
  - **쿼리 임베딩** = ReAct가 런타임에 만드는 검색어 → **데이터셋에 없음**(즉석 생성이라 미리 못 만듦). 벡터 검색은 "쿼리 벡터 ↔ 문서 벡터" 유사도라 쿼리도 벡터화해야 함.
- **제약**: 쿼리는 문서와 **같은 모델**로 임베딩해야 같은 벡터공간에서 비교됨. → D4.

## D4. 쿼리 인코더 = Cohere API (오픈 로컬 모델 대신)
- **결정**: 런타임 쿼리 임베딩을 Cohere API `embed-multilingual-v3.0`(`input_type="search_query"`)로 수행. `COHERE_API_KEY` 필요.
- **근거 1 (필연)**: Cohere `embed-multilingual-v3.0`은 **오픈 웨이트가 없음(API 전용)**. Cohere 사전임베딩을 쓰려면 쿼리도 같은 모델 = Cohere API뿐. (다른 모델로 임베딩하면 벡터공간 불일치 → 검색 무의미.)
- **근거 2 (목적 부합)**: 사용자의 측정 목표가 **decode-RAG 비율**(D9). Cohere API의 **네트워크 지연**(~0.3–0.5초/call)이 retrieval 시간을 키워 이 비율을 의미있게 만든다.
- **대안 (오픈 로컬 모델)**: 키 불필요·완전 로컬이지만 → Cohere 사전임베딩을 버리고 문서·쿼리 둘 다 재임베딩해야 함. 그리고 로컬 검색이 너무 빨라 decode-RAG 비율이 ~2%로 낮아짐(D9) → **목적과 충돌하여 기각**.
  - (단 이 경로는 "완전 로컬·재현"이 최우선일 때의 합리적 대안이었음. 사용자가 decode-RAG 비율을 우선해 Cohere API 선택.)

## D5. 코퍼스 = HotpotQA 타깃 부분집합 (full / random / fullwiki-context 기각)
- **결정**: Cohere 영어 50샤드를 스트리밍하며 **HotpotQA 관련 title**(gold ∪ context)의 passage만 보관해 코퍼스 구성. (저장은 수백 MB~1GB → 내장 디스크 OK.)
- **근거**: decode-RAG 비율은 주로 네트워크가 좌우하므로 코퍼스 "크기"는 비율보다 **정답 커버리지·현실성**에 영향. 타깃 부분집합은 정답 커버리지를 확보하면서 내장 디스크에 들어감.
- **대안·기각**:
  - **fullwiki context 그대로 사용**: 디스크의 `hotpot_dev_fullwiki_v1.json` context는 **정답 2개를 모두 포함하는 질문이 30.3%뿐**(평균 커버리지 57.5%, 직접 측정). ~70% 질문이 풀 수 없음 → 기각.
  - **랜덤 N샤드(~400만)**: 가볍지만 HotpotQA 정답 커버리지가 랜덤이라 정확도 낮음 → 기각(지연/비율 측정엔 무방하나 정확도 의미 약함).
  - **전체 4,148만**: 외장 SSD ~150–200GB + IVF-PQ 압축 인덱스(36GB RAM로 flat 170GB 불가) 필요. 외장 디스크 없으면 불가 → 기각(외장 있으면 재고 가능).
- **주의**: HotpotQA(2017) title ↔ Cohere(2023-11) title **drift**로 일부 gold 미매칭 가능 → 빌드 시 커버리지 리포트로 정량화.

## D6. lookup 도구 제거
- **결정**: ReAct tools = `[search, finish]`. `lookup` 삭제.
- **근거**: 기존 `lookup`은 직전 `search`가 **저장한 단일 위키 페이지** 안에서 문단을 찾음(`wikipedia.py::DocstoreExplorer.lookup`). dense passage 검색엔 "현재 페이지" 개념이 없어 매핑 불가(코드로 확인).
- **대안 (lookup 재정의)**: 직전 검색 결과 내 재랭킹으로 바꿀 수 있으나 의미 모호·코드 증가 → 기각.

## D7. few-shot 프롬프트를 passage 스타일로 신규 작성
- **결정**: `hotpotqa_vectordb.py` 신규(원본 유지). `lookup[...]`·"Could not find X. Similar:[...]" 예시 제거, `search[query] → passages → finish` 흐름 예시로 교체.
- **근거**: 원본 few-shot은 title 기반(`search[엔티티]` → 페이지 요약, `lookup`). dense 검색 동작과 불일치 → 프롬프트도 passage 검색에 맞춰야 함.

## D8. 평가 = HotpotQA dev + 기존 EM
- **결정**: 질문/정답은 `hotpot_dev_fullwiki_v1.json`(정식 fullwiki dev, 7,405문항), 채점은 기존 `measurement/eval.py::hotpotqa_em`.
- **근거**: dev는 정답·근거 공개라 로컬 채점 가능(test는 정답 비공개). 기존 harness 그대로 재사용. 타깃 코퍼스라 커버리지 양호.
- **주의**: 이번 1차 목표는 **정확도가 아니라 지연/비율 측정**. 커버리지 drift로 정확도가 낮아도 측정 자체는 유효.

## D9. decode-RAG "비율"의 함정 (D4·D5를 좌우한 핵심)
- **사실**: decode-RAG 비율 = `tool 시간 ÷ e2e`. 로컬 LLM이 느림(~20초/쿼리)인데 로컬 임베딩+FAISS는 ~수십 ms.
  - 순수 로컬 dense → 비율 **~2%** (병목 사라짐 → decode-RAG가 숨길 게 거의 없음).
  - 기존 라이브 위키 44%는 사실 **네트워크 스크래핑**(~3.6초/call).
  - Cohere API(~0.3–0.5초/call) → 비율 **의미있게 회복**.
- **함의**: "실제 벡터DB(로컬·고속)"는 사용자에겐 좋지만 **연구 동기인 병목을 없애버림**. 그래서 의도적으로 네트워크(Cohere API)를 retrieval에 실어 비율 확보. (참고: 논문 A100은 LLM이 빨라 같은 retrieval이라도 비율이 높게 나옴 — 우리 비율이 낮은 건 느린 로컬 LLM 탓.)
- **검토했으나 미채택**: 로컬 + 리랭커(cross-encoder)로 지연 확보(완전 로컬·재현). 사용자가 Cohere API + 대형 코퍼스를 선택.

## D10. 측정은 기존 harness 재사용 (sweep 코드 변경 불필요)
- **결정**: `sweep_runner/agent_runner/cells/measurement`를 수정하지 않음. config `defaults`에 `retrieval_backend` 등만 추가.
- **근거**: `defaults`의 임의 필드가 `extra_kwargs` → `agent_kwargs`로 전달됨(`cells.py:69-71`, `agent_runner.py:219-225`). `search`가 표준 `BaseTool` 경로라 `TraceCallbackHandler`가 tool span(=Cohere API+FAISS 시간) 자동 캡처 → `tool_total_ms`/`n_tool_calls` 그대로 기록.

---

## 부록: 조사 중 확인된 사실

**Cohere `CohereLabs/wikipedia-2023-11-embed-multilingual-v3` (영어 split)**
- 41,488,110 passage, **1024차원**, 50 parquet 샤드, 총 **~90 GB**(fp16 추정). 스키마: `_id, url, title, text, emb`.
- 쿼리 검색 예시(카드): `co.embed(texts=[q], model='embed-multilingual-v3.0', input_type="search_query")` → **API 전용, 오픈 웨이트 없음**.

**HotpotQA `hotpot_dev_fullwiki_v1.json`** (디스크에 존재, 47 MB)
- 7,405문항. keys: `_id, answer, question, supporting_facts, context, type, level`. context = `[title, [문장들]] × 10`.
- **fullwiki context의 gold 커버리지**: 두 gold 문단을 모두 포함하는 질문 **30.3%**(1000개 표본), 평균 gold-title 커버리지 **57.5%**. → D5의 근거.
- 풀 dev context의 unique title ≈ **66,573**.

**하드웨어 (M3 Pro)**
- RAM 36 GB, 내장 디스크 여유 **~30 GB**(94% 사용). Python 3.13 venv(`repro/.venv`). LLM = 로컬 llama-server(Llama-3.1-8B Q4_K_M, GGUF 준비됨).
- 전체 코퍼스 flat 인덱스: 41.49M × 1024 × 4B ≈ **170 GB**(fp32) → 36 GB RAM 불가 → 전체 시 IVF-PQ 압축 필수.

**기존 코드 한계 (발표 L1–L7) 중 이 실험과 관련**
- **L5**(라이브 위키 재현불가) → 이 실험이 **개선**(고정 벡터DB).
- **L4**(비-ReAct 콜백 우회) → ReAct-only로 **회피**.
- **L4-b**(tool 관찰이 `ToolMessage`가 아닌 `SystemMessage` → tool 토큰 0, Fig8 FAIL) → **잔존**(비교가능성 위해 래핑 유지).
- **L1/L2/L6**(A100 vs M3, Q4 양자화, DCGM 없음) → 공통 잔존(이 실험 무관).
