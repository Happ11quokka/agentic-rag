# state-sync prefetch — 설계

**작성일**: 2026-08-05
**브랜치**: `wikipedia_hdd`
**상위 문서**: [`PLAN.md`](PLAN.md) (Stage 0–3 기획), [`../wikipedia_hdd/DEVLOG.md`](../wikipedia_hdd/DEVLOG.md) (HDD 처치)

---

## 무엇을 만드는가

Target 모델이 **검색할 때마다** 그 시점의 agent 상태를 drafter에 동기화하고, drafter가
예측한 다음 쿼리를 미리 검색해 캐시에 넣는다. 다음 hop에서 실제 쿼리가 예측과 일치하면
검색 대기가 0이 된다.

`PLAN.md`의 Stage 2를 이 브랜치에 맞춰 구체화한 것이다. 차이는 두 가지다.

| | PLAN.md (원안) | 이 설계 |
|---|---|---|
| 검색 백엔드 | Cohere embed + FAISS (`AgentBench/`) | **Milvus DiskANN on HDD** (`wikipedia/`) |
| 트리거 | `react.py::call_model` 끝 | **검색 완료 시점** (아래 참조) |

`AgentBench/`는 third-party 클론이라 이 저장소에 없고 현재 머신에도 없다. 검색 경로를
Milvus로 옮기는 것이 `wikipedia_hdd` 브랜치의 목적이므로, 이 브랜치가 실제로 가진 검색
seam인 `search_text(database, encoder, ...)`를 감싼다.

---

## 1. 트리거 — 왜 "검색 완료" 시점인가

한 hop의 시간 구성:

```
[decode: Thought/Action] → [retrieval q_n: 블로킹] → [decode: Thought/Action] → [retrieval q_{n+1}] → ...
```

prefetch를 걸 수 있는 시점은 두 곳이다.

**(A) 검색 시작 시점** — `Action: search[q_n]`을 파싱한 직후. 예측 입력에 observation n이
없고, prefetch가 **target 자신의 검색과 같은 디스크를 동시에 때린다.** 이 HDD에서 희소
자원은 랜덤 IOPS(실측 68회/초)다. 두 검색이 서로를 굶긴다.

**(B) 검색 완료 시점** — observation n이 생긴 직후. 채택.

- observation은 다음 쿼리를 예측하는 **가장 강한 입력**이고, 검색이 끝나야 존재한다
- prefetch가 **decode와 겹친다.** 이것이 `PLAN.md` 5.1절의 상한
  `hidden ≤ min(retrieval, decode)`와 정확히 대응한다
- target이 decode하는 동안 디스크는 유휴다 — DEVLOG 5.2절이 관측한 "검색이 대기하는 동안
  CPU/GPU가 논다"의 뒷면이다

> 사용자 요구는 "Retrieval할 때마다 state sync"다. (B)는 이를 만족한다 — 동기화는 검색
> 경계에서 hop당 정확히 한 번 일어난다.

## 2. seam — 호출 하나

```python
results = retriever.retrieve(query, state)
```

내부 동작:

1. `normalize(query)`로 캐시 조회
2. **hit** → 캐시 반환 (아직 실행 중이면 재시작하지 않고 **대기**)
   **miss** → 실제 검색
3. `state + (query, results)`를 drafter에 sync하고 prefetch 스레드 spawn

**sync를 검색의 부수효과로 만드는 것이 설계의 핵심이다.** 별도의 `sync()` 호출을 두면
agent 루프의 어느 분기에서 빠뜨릴 수 있고, 그러면 그 hop만 조용히 baseline으로 돌아간다 —
측정에는 "prefetch가 켜져 있었다"고 남은 채로. 호출을 하나로 묶으면 그 실패 모드가
구조적으로 불가능하다.

## 3. 무결성 — 정확도가 움직일 수 없는 이유

캐시는 `normalize(예측) == normalize(실제)`일 때만 쓴다. 불일치면 기존 경로를 그대로 탄다.

```
hit  → 캐시의 결과 = 그 쿼리로 검색한 결과 (같은 DB, 같은 파라미터)
miss → 지금 검색한 결과
```

두 경우 모두 반환 passage가 baseline과 동일하다. prefetch는 **언제 검색하느냐**만 바꾸고
**무엇을 받느냐**는 바꾸지 않는다. 따라서 EM/F1은 수학적으로 불변이고, 이는 테스트로
강제한다(§6).

`normalize_query()`가 캐시 writer와 lookup의 **단일 진실원**이다. 둘이 갈라지면 hit이
영원히 0이 되거나(무해하지만 실험이 죽음) 다른 쿼리의 결과를 반환한다(무결성 붕괴).

## 4. 구성 요소

`wikipedia/src/wikipedia/prefetch.py` 하나에 둔다. 검색 경로가 이 패키지에 있고,
prefetch는 그 경로의 래퍼이기 때문이다.

| 요소 | 책임 |
|---|---|
| `normalize_query(text)` | 캐시 키. casefold + 공백 정규화 |
| `Hop` / `AgentState` | 동기화되는 불변 스냅샷: question + 지금까지의 (query, 결과 요약) |
| `QueryPredictor` | 프로토콜: `predict(state) -> str \| None` |
| `PrefetchCache` | 에피소드당 새로 생성. Lock 보호 |
| `PrefetchingRetriever` | seam. hit/miss·타이밍 기록 |
| `PrefetchStats` | attempts / hits / hidden_ms / wasted / contended |

**predictor 구현 두 종**

- `ReplayPredictor` — 기록된 쿼리 순서를 재생. 결정적이라 테스트와 기계장치 검증에 쓴다.
  적중률을 인위적으로 조절할 수 있어 hit/miss 경로를 둘 다 밟게 한다
- `HTTPPredictor` — OpenAI 호환 `/v1/chat/completions`. llama-server와 Ollama 양쪽에 붙는다.
  서버가 없거나 죽으면 **무조건 miss로 graceful degrade** — prefetch 실패가 본 파이프라인을
  절대 깨뜨리면 안 된다

`PrefetchCache`를 에피소드당 새로 만드는 것이 교차오염을 구조적으로 막는다. 전역 캐시라면
에피소드 A의 결과가 B에 새는지를 테스트로 증명해야 하지만, 수명이 에피소드에 묶이면
증명할 것이 없다.

## 5. 측정 — 무엇을 정직하게 기록하는가

| 값 | 정의 |
|---|---|
| `prefetch_ms` | prefetch 스레드가 검색에 쓴 시간 |
| `wait_ms` | target이 실제로 기다린 시간 (hit이고 이미 끝났으면 ≈0) |
| `hidden_ms` | `prefetch_ms − wait_ms`. 이번 hop에서 실제로 숨긴 양 |
| `contended` | miss인데 prefetch가 아직 디스크를 쓰고 있었는가 |
| `wasted` | 끝까지 안 쓰인 prefetch 검색 수 |

**hit인데 prefetch가 아직 실행 중이면 재시작하지 않고 기다린다.** 재시작하면 이미 한
디스크 작업을 버리고 두 번 하게 되어, 숨긴 양을 과소평가하는 동시에 HDD를 두 배로 때린다.

`contended`가 이 구성의 contention tax다. `PLAN.md`는 두 모델이 GPU 대역폭을 다투는
비용을 걱정했는데, 여기서는 **빗나간 prefetch가 target의 실제 검색과 같은 HDD를 다툰다.**
적중률이 낮으면 prefetch가 순손실이 될 수 있다. 이 값이 그것을 드러낸다.

> 순절감 = Σ hidden_ms − (빗나간 prefetch가 유발한 지연)

## 6. 테스트

| 테스트 | 무엇을 고정하는가 |
|---|---|
| 정확도 불변 | 어떤 예측(적중·빗나감·예외·타임아웃)에도 `retrieve()` 반환이 prefetch를 끈 것과 동일 |
| 게이트 off | predictor가 없으면 검색 횟수·결과가 baseline과 정확히 일치, 스레드 0개 |
| normalize 단일 진실원 | writer와 lookup이 같은 함수를 쓴다 |
| drafter 장애 | predict가 raise/timeout/None/빈문자열이어도 검색이 정상 반환 |
| 캐시 격리 | 에피소드 A의 캐시가 B에서 안 보인다 |
| 대기 회계 | 실행 중 hit에서 재검색이 아니라 대기가 일어나고 `hidden_ms`가 맞다 |

단위 테스트는 가짜 DB로 돈다 — Milvus도 HDD도 필요 없다. 실측은 드라이버가 실제
컬렉션에 대고 따로 한다.

## 7. 채택하지 않은 안

| 안 | 사유 |
|---|---|
| 검색 시작 시점에 sync (§1 A안) | prefetch가 target 검색과 같은 HDD IOPS를 다툰다. observation도 못 쓴다 |
| fuzzy / semantic 매칭으로 적중률 올리기 | 반환 passage가 baseline과 달라져 정확도 불변이 깨진다. 적중률은 올라가지만 측정 대상이 바뀐다 |
| 전역 캐시 | 교차오염을 테스트로 증명해야 한다. 에피소드 수명에 묶으면 증명할 게 없다 |
| 예측 여러 개를 동시에 prefetch | 이 HDD에서 IOPS가 천장이다. 빗나간 검색 하나가 이미 경합인데 n개면 target을 굶긴다 |
| `search_text()` 자체를 수정 | 검색 경로는 baseline도 쓴다. 래퍼로 두면 prefetch를 끈 경로가 바이트 단위로 종전과 같다 |

## 8. 남은 리스크

- **적중률이 이 설계로 결정되지 않는다.** drafter 품질의 문제이고, `PLAN.md`의 게이트
  (exact hit ≥ ~0.4)가 그대로 유효하다. 이 설계는 적중했을 때 실제로 숨겨지는지를 보장할 뿐이다
- **10M 컬렉션 로드가 아직 안 끝났다.** 실측은 그 뒤다. 기계장치 검증은 로드와 무관하게 가능
- **target 모델이 현재 머신에 없다.** `HTTPPredictor`는 서버가 생기면 붙는다. 그 전까지
  `ReplayPredictor`로 기계장치와 회계를 검증한다 — 적중률은 재지 못하고, 재는 척도 하지 않는다
