# 실험: ReAct × 실제 벡터DB (Cohere Wikipedia 2023-11)

**상태**: 📋 계획 단계 — 코드/연구 **미실행** (이 폴더는 계획 문서만 담음)
**작성일**: 2026-06-07
**상위 프로젝트**: `repro/` — arXiv **2506.04301v2** (*The Cost of Dynamic Reasoning*) 로컬 재현

---

## 한 줄 요약

기존 재현에서 ReAct가 쓰던 **라이브 Wikipedia 키워드 검색**을 **실제 dense 벡터DB**(Cohere 사전임베딩 Wikipedia 2023-11)로 교체하고, **ReAct만** 돌려 **decode-RAG baseline**(검색 호출 수 + retrieval 시간 비율)을 측정한다.

## 큰 그림

| | 지난번 (완료) | 이번 (이 실험) |
|---|---|---|
| 대상 | 논문 4종 에이전트 재현 | **ReAct만** |
| 검색 | 라이브 Wikipedia API (키워드) | **dense 벡터DB (FAISS, Cohere 임베딩)** |
| 산출 | fig4/5/7/8/13 등 패턴 재현 | decode-RAG baseline 측정 |

## 목적 (왜 이걸 하나)

사용자의 **decode-RAG 연구**(decode 중 retrieval prefetch로 검색 지연을 숨김)의 baseline 확보. 측정 핵심 두 가지:

1. **tool(검색) 호출 수** — HotpotQA는 multi-hop이라 쿼리당 검색을 여러 번 함.
2. **decode-RAG 비율** = `retrieval 대기 / e2e`. 쿼리 임베딩을 **Cohere API**로 수행해 네트워크 지연을 retrieval에 실어 이 비율을 의미있게(>2%) 확보.

> ⚠️ 핵심 통찰: 로컬 LLM이 느려서(~20초/쿼리) **순수 로컬 dense 검색만으론 이 비율이 ~2%로 낮음**. 기존의 44%는 사실 라이브 위키 **네트워크 대기**였음. 그래서 의도적으로 Cohere API(네트워크)를 써서 비율을 확보한다. → 자세한 근거 `DECISIONS.md` D4·D9.

부수 효과: 발표가 지적한 한계 **L5(라이브 호출 재현불가)** 개선, **ReAct-only**라 L4(비-ReAct 콜백 우회)·latency 중복 attribution 문제를 자동 회피.

## 핵심 결정 (요약)

| 항목 | 결정 |
|---|---|
| 에이전트 | **ReAct만** (다른 에이전트 코드는 유지, 실행 안 함) |
| 검색 백엔드 | 라이브 Wikipedia → **dense 벡터DB(FAISS)** |
| 문서 임베딩 | **Cohere 사전임베딩 그대로** (재임베딩 X) |
| 쿼리 임베딩 | **Cohere API** `embed-multilingual-v3.0` (네트워크 = 비율 확보) |
| 코퍼스 | **HotpotQA 타깃 부분집합** (Cohere 영어 50샤드 스트리밍 → 관련 title만 보관) |
| lookup 도구 | **제거** → ReAct tools = `[search, finish]` |
| 평가 | HotpotQA dev 질문 + 기존 EM (`hotpotqa_em`) |

각 결정의 근거·대안·기각 이유는 **`DECISIONS.md`**.

## 폴더 구성

| 파일 | 내용 |
|---|---|
| `README.md` | (이 문서) 개요·목적·결정 요약 |
| `PLAN.md` | 구현 계획 — 아키텍처, 생성/수정 파일, 단계별 구현(코드 스케치), 실행 명령, 검증, 리스크 |
| `DECISIONS.md` | 합의된 결정과 근거 (대안·기각 이유, 발견한 수치 포함) |

## 전제조건

- **`COHERE_API_KEY`** — cohere.com 무료 트라이얼 키로 테스트 규모는 충분. 런타임 쿼리 임베딩용. (문서는 사전임베딩을 쓰므로 **인덱스 빌드 시 API 불필요**.)
- 기존 `repro/.venv` (Python 3.13), llama-server (Llama-3.1-8B Q4_K_M) — 이미 준비됨.

## 다음 단계 (구현을 시작할 때)

`PLAN.md`의 7단계: ① deps → ② 인덱스 빌드 → ③ retriever/도구 → ④ 프롬프트 → ⑤ `run_react` 분기 → ⑥ sweep config → ⑦ 실행·측정.
**현재는 의도적으로 미실행 상태.**
