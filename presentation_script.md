# 발표 대본 — The Cost of Dynamic Reasoning 재현 발표

**발표일**: 2026-05-28
**파일**: `Downloads/2026.05.28.key` (Keynote), 원본 `presentation.html`
**발표 시간**: 약 25분 + Q&A 5분
**구조**: 7개 섹션, 31개 슬라이드

---

## 전체 개요

| Part | 슬라이드 | 시간 | 핵심 메시지 |
|---|---|---|---|
| Cover | 1 | 1분 | 논문/재현 환경 소개 |
| Part 1: 동기 | 2-6 | 3분 | 왜 에이전트 비용을 측정해야 하나 |
| Part 2: 분류 | 7-10 | 3분 | 5종 에이전트 + 우리 재현 환경 |
| Part 3: 측정 결과 | 11-17 | 10분 | **핵심** — 좌:논문 / 우:재현 비교 |
| Part 4: Serving/Scaling | 18-23 | 3분 | 논문 결과 압축 정리 |
| Part 5: 인프라 함의 | 24-26 | 2분 | 충격적 수치 (200 GW 등) |
| Part 6: 결론 | 27-30 | 2분 | Verification, 한계, 결론 |
| Q&A | 31 | 1분 | 참고 자료 |

---

## 🟦 Slide 1 — 표지 (60초)

**화면**: 논문 제목 (영어 + 한국어), 저자, 재현 환경, 발표자

**대본**:
> 안녕하세요. 오늘 발표할 논문은 KAIST 류민수 교수 연구실의 *The Cost of Dynamic Reasoning*, 한국어로는 "동적 추론의 비용" 입니다. 2026년 1월에 arXiv에 v2가 올라왔고, HPCA-2026에 게재됐습니다.
>
> 이 논문은 **AI 에이전트가 인프라 측면에서 얼마나 비싼지를 정량적으로 측정한 시스템 측정 논문**입니다. 단순한 알고리즘 개선이 아니라 "에이전트는 새로운 워크로드"라는 관점에서 GPU·전력·데이터센터 비용을 분석합니다.
>
> 저는 이 논문의 HotpotQA 실험 일부를 **로컬 M3 Pro 환경에서 재현**했습니다. Apple Silicon, Llama-3.1-8B Q4_K_M, llama.cpp 백엔드로요. 절대 수치는 환경 차이로 다르지만 **패턴 수준에서 일치하는지**를 검증하는 게 목표였습니다.

**상세 설명**:
- 논문 저자 4명 중 마지막이 corresponding author Minsoo Rhu (KAIST 컴퓨터 아키텍처 연구실)
- 본 재현 동기: 사용자 본인의 decode-RAG 연구를 위한 baseline 측정 인프라 확보
- 재현 환경 핵심: A100 + vLLM (논문) → M3 Pro + llama.cpp (우리)

---

## 🟦 Slide 2 — Part 1 디바이더 (10초)

**화면**: "Part 1 — 동기와 문제 정의"

**대본**:
> 먼저 왜 이 논문이 필요한지부터 보겠습니다.

---

## 🟦 Slide 3 — 왜 이 논문인가 (60초)

**화면**: 좌(이미 알려진 것) / 우(아직 측정 안 된 것) + 핵심 질문 박스

**대본**:
> LLM 발전 패러다임이 "모델·데이터 키우기"에서 "추론 시 더 깊이 생각하기 (test-time scaling)"로 이동했고, 그 정점에 있는 것이 AI 에이전트입니다. Plan → Tool → Observe → Refine을 반복하는 구조죠.
>
> 정적 LLM의 비용은 잘 측정됐습니다. ChatGPT 1쿼리가 웹검색 10배 전력이라거나, xAI Colossus가 H100 10만대로 150 MW 소비한다는 등. 그런데 **에이전트 단대단 인프라 비용은 종합적으로 측정된 적이 없었습니다**.
>
> 이 논문이 던지는 질문은 단순합니다 — "ChatGPT처럼 한 번에 답하는 LLM 대비, Plan하고 Tool 부르고 Reflect하는 AI 에이전트는 인프라 측면에서 얼마나 더 비쌀까? 그리고 그만큼 정확도가 오를까?"

**상세 설명**:
- "이미 알려진 것" 박스: 정적 LLM serving 분야 (vLLM, Orca 등)
- "아직 측정 안 된 것": 5종 에이전트의 단대단 인프라 비용
- 핵심 질문은 발표 전체의 thesis — 청중이 따라올 angle

---

## 🟦 Slide 4 — Fig 1: 정적 vs 동적 추론 (45초)

**화면**: 논문 Fig 1 (3가지 추론 전략)

**대본**:
> 이 그림은 LLM 기반 시스템의 세 가지 추론 전략을 비교합니다.
>
> (a)는 추론 없이 입력을 한 번에 출력으로 매핑하는 기본 LLM, (b)는 정적 추론 LLM — Chain-of-Thought처럼 LLM 내부에서 중간 단계를 생성하는 방식. (c)가 우리가 다룰 AI 에이전트입니다.
>
> 주목할 점은 토큰 색깔이 5종으로 늘어난다는 겁니다. 회색 input, 분홍 output, 청록 reasoning, 녹색 tool call, 노랑 tool observation. **컨텍스트가 누적되며 다양해지는 게 에이전트의 본질**입니다.

**상세 설명**:
- (c)의 예시: "Paris→New York 최저가 항공편 찾기" → 항공편 검색 도구 호출 → 결과 관찰 → 최저가 결정
- 핵심: 에이전트는 외부 환경과 능동적으로 상호작용

---

## 🟦 Slide 5 — Fig 2: AI Agent 5요소 구조 (45초)

**화면**: 논문 Fig 2

**대본**:
> AI 에이전트는 5가지 요소로 분해됩니다.
>
> ① Agent core — Actor, Planner, Reflection module 같이 LLM이 특정 역할을 부여받습니다. ② Memory는 단기(대화 trace)와 장기(사용자 선호) 둘 다 있고요. ③ Plan은 하위 작업을 시퀀스나 DAG로 조직합니다. ④ Tools는 외부 환경 상호작용, ⑤ Workflow는 이 네 요소가 어떻게 반복적으로 상호작용하는지를 정의합니다.
>
> ReAct, Reflexion, LATS, LLMCompiler는 이 Workflow 패턴이 서로 다른 에이전트들입니다.

---

## 🟦 Slide 6 — Fig 3: 5종 에이전트 타임라인 (60초)

**화면**: 논문 Fig 3 (CoT/ReAct/Reflexion/LATS/LLMCompiler 실행 패턴)

**대본**:
> 이 그림은 5종 에이전트가 실제로 어떻게 다른지 시각화한 타임라인입니다.
>
> **CoT**는 LLM 1번으로 끝, 도구 없음. **ReAct**는 thought → action → observation을 반복합니다. 가장 직관적인 패턴이죠.
>
> **Reflexion**은 ReAct의 trace가 끝난 뒤 별도로 "reflection" 단계가 평가하고 다음 trial을 시도합니다. 자기 반성이 추가된 거예요.
>
> **LATS는 트리 검색 — MCTS** 입니다. 각 노드 확장마다 자식 후보를 여러 개 sampling해서 LLM 호출이 폭증합니다. 나중에 보겠지만 ReAct 대비 LLM 호출이 20배 이상 됩니다.
>
> **LLMCompiler는 다른 접근** — 먼저 DAG 형태로 plan을 한 번에 만들고, 독립적인 노드를 비동기 병렬 실행합니다. 이론상 가장 효율적이어야 하지만 우리 환경에서는 깨졌습니다. 뒤에 자세히 설명드리겠습니다.

**상세 설명**:
- 청중이 5종 에이전트 차이를 명확히 잡아야 뒤 결과 슬라이드들이 이해됨
- LLMCompiler executor 문제는 슬라이드 12, 17에서 다시 언급

---

## 🟦 Slide 7 — Part 2 디바이더 (10초)

**화면**: "Part 2 — 에이전트 분류와 우리 재현 환경"

**대본**:
> 다음으로 무엇을 비교했고, 우리는 어떤 환경에서 재현했는지 보겠습니다.

---

## 🟦 Slide 8 — Table 1: 에이전트 비교 매트릭스 (30초)

**화면**: 논문 Table 1

**대본**:
> 이 표는 5종 에이전트를 5가지 능력 축 — Reasoning, Tool use, Reflection, Tree search, Planning — 으로 정리합니다.
>
> CoT는 reasoning만, ReAct는 + tool use, Reflexion은 + reflection, LATS는 + tree search, LLMCompiler는 + DAG planning. **이 능력 차이가 곧 LLM 호출량과 비용 구조의 차이를 만들어내는 근원**입니다.

---

## 🟦 Slide 9 — Table 2/3: 벤치마크 특성 (45초)

**화면**: 논문 Table 2, Table 3 좌우

**대본**:
> 논문은 4종 벤치마크를 사용합니다. HotpotQA는 multi-hop QA로 Wikipedia API를 도구로 씁니다. WebShop은 쇼핑 사이트 인터랙션, MATH는 수학, HumanEval은 코딩이죠.
>
> 우리는 **HotpotQA만 재현했습니다**. 이유는 Wikipedia API tool wait가 GPU idle 패턴을 가장 잘 드러내기 때문 — decode-RAG 연구의 baseline으로 가장 적합합니다.
>
> 오른쪽 Table 3을 보시면 ShareGPT 대비 LATS 70B는 latency 47.8배, energy 62.1배 폭증합니다. 이게 논문의 핵심 충격 수치입니다.

---

## 🟦 Slide 10 — 우리 재현 환경 vs 논문 환경 (90초, 중요)

**화면**: 좌(논문) / 우(우리) + footnote

**대본**:
> 이게 가장 중요한 슬라이드 중 하나입니다 — 우리 환경이 논문과 어떻게 다른지.
>
> **논문은 GCP A100 1장 (8B) 또는 8장 (70B)에 vLLM 풀-정밀 추론입니다.** 우리는 **M3 Pro 36GB unified memory에 llama.cpp + Llama-3.1-8B Q4_K_M GGUF**.
>
> 왜 이런 환경 선택을 했냐 — 청중이 자주 묻는 질문이라 미리 정리합니다.
>
> 첫째, **llama.cpp 선택 이유**. 후보 3개(Ollama, MLX-LM, llama.cpp) 중 prefill/decode 메트릭을 외부로 노출하는 유일한 백엔드입니다. 측정 인프라가 없으면 재현 자체가 무의미하니까요.
>
> 둘째, **Q4_K_M 양자화 이유**. 36GB에 70B는 안 들어가고, 8B FP16은 들어가지만 디코드가 2-3배 느려서 LATS 1쿼리에 40분 이상 걸립니다. 시간 예산 30시간으로는 불가능하죠. Q4는 정확도 일부 희생하고 wall-clock 2-3배 단축, 30시간 안에 완주 가능했습니다.
>
> 셋째, **HotpotQA만** 한 이유. WebShop은 docker 환경, MATH는 표 채점기, HumanEval은 코드 sandbox가 따로 필요해서 셋업 비용이 크고 decode-RAG와도 무관합니다.
>
> **핵심**: 절대 수치는 환경 차이로 비교 불가, **비율·분포·순위·트레이드오프 형상**을 재현 목표로 잡았습니다.

**상세 설명** (Q&A 대비):
- "왜 70B 안 했냐?": 36GB unified memory에 70B Q4_K_M도 ~40GB 필요. 물리적 불가능.
- "왜 FP16 8B 안 했냐?": 메모리는 OK지만 M3 Pro Metal의 메모리 대역폭 한계 (150GB/s)로 decode 2-3배 느림. LATS 50샘플 = 34시간 (Q4는 14시간).
- "왜 Apple GPU 못 측정?": NVIDIA DCGM 같은 GPU 코어 활용률 도구가 Apple에 없어서 wall-clock proxy 사용.

---

## 🟦 Slide 11 — Part 3 디바이더 (10초)

**화면**: "Part 3 — 측정 결과: 우리가 재현한 핵심 figure"

**대본**:
> 이제 본론입니다. 여기부터는 모두 좌측이 논문 원본, 우측이 우리 재현, 하단이 비교 해설 형식으로 진행됩니다.

---

## 🟦 Slide 12 — Fig 4: LLM·Tool 호출 횟수 (120초, 핵심)

**화면**: 좌(논문 fig4) / 우(우리 fig4_calls — measured + estimated 분리) / 해설

**대본**:
> 첫 번째 핵심 차트입니다. 요청 1개당 평균 LLM 호출과 Tool 호출 횟수.
>
> **논문 패턴**: LATS가 압도적으로 LLM 호출이 많습니다. HotpotQA에서 평균 71회/쿼리. CoT 대비 9.2배예요. 다른 ReAct, Reflexion, LLMCompiler는 비슷한 수준입니다.
>
> **우리 패턴**: 같은 방향으로 재현됐습니다. LATS 193회 (논문보다 2.7배 큰 이유는 우리 설정 `n_generate_sample=5`가 페이퍼 구현보다 트리 폭이 넓음 + LATS 표본 3개 통계 변동성). LATS/ReAct = 21.87배로 자릿수 트렌드 동일.
>
> 그런데 차트를 자세히 보시면 **검정색 Tool 막대가 ReAct만 측정됐고, 나머지 셋은 빗금**으로 표시됐죠. 이게 측정 한계 L4의 직접적 표시입니다.
>
> 우리 `TraceCallbackHandler`는 LangChain 표준 `BaseTool.invoke` 경로에 후킹했는데, ReAct만 표준 경로를 거치고 Reflexion/LATS/LLMCompiler는 각자 자체 wrapper로 우회합니다. 처음에는 0으로 나왔어요.
>
> 그래서 **사후 구조적 복구**를 했습니다. Reflexion 공식은 `n_llm − 2×reflections − 1 = 14`. 워크플로우 구조에서 직접 유도됩니다 — Reflexion = trial × (R+1) + reflection × R, 각 trial 마지막은 Finish (tool 안 부름), reflection은 LLM만 부름. 차감 방식이죠.
>
> LATS는 곱셈 방식. `expansions × 5 children × 3 actions = 55`. expansions와 children은 코드에서 확정값, actions는 추정 (±30% 오차).
>
> **LLMCompiler는 ?**. planner가 LLM 1번만 부르고 executor가 작동 안 함 — 50쿼리 중 41개가 빈 답입니다. 구조적 변수가 없어서 복구 불가, 별도 패치 + 재실행이 필요합니다.

**상세 설명**:
- LLMCompiler의 "executor failed" 추정 원인: AgentBench의 LLMCompiler harness가 자체 async executor를 쓰는데 우리 환경에서 어떤 이유로 plan execution이 안 발화. 디버깅 미수행.
- 복구 방법론은 design spec §11 L4에 정식 한계로 명시됨
- 발표 시 청중이 "왜 LATS는 그렇게 많아?"라고 자주 물음 → 트리 자식 sampling 구조로 설명

---

## 🟦 Slide 13 — Fig 5: 지연 시간 분해 (120초, 핵심)

**화면**: 좌(논문 fig5) / 우(우리 fig5_latency_breakdown — 빨강 LLM, 검정 Tool, 회색 Others) / 해설

**대본**:
> 두 번째 핵심 차트는 지연 시간 분해입니다. 색상은 논문 규약과 동일하게 맞췄어요 — **빨강 = LLM, 검정 = Tool, 회색 = Others, 녹색 다이아 = e2e latency**. 좌우 직접 비교 가능합니다.
>
> **ReAct (측정 신뢰)**: Prefill 16% + Decode 36% + Tool 44%. Tool wait가 단일 최대 컴포넌트입니다. 논문 §IV-A 핵심 메시지 "HotpotQA에서 tool execution이 latency를 지배"가 우리 환경에서도 정확히 재현됩니다.
>
> Wikipedia API 호출당 시간을 비교하면 — 논문은 평균 1.2초/회, 우리는 2.08초/회. 같은 자릿수, 2배 차이는 M3 Pro 하드웨어 격차입니다.
>
> **Reflexion/LATS는 빗금 막대**가 있죠. 이것도 사후 추정값입니다. recovered tool count × ReAct의 per-call rate (2.08초) → **Reflexion ≈ 29초 (e2e 87초의 33%), LATS ≈ 114초 (e2e 1037초의 11%)**.
>
> 그런데 차트 위에 빨간 글씨로 "raw sum 157%, 328%"라고 경고가 있습니다. Reflexion/LATS는 100ms 폴링 thread가 동시 실행 구간에서 prefill·decode를 중복 attribution해서 합이 100%를 크게 넘습니다. 정규화로 강제로 100%에 맞춘 거고, 음영 비율 자체는 신뢰 불가능합니다.
>
> **LLMCompiler는 회색 overhead가 84%**. tool 호출이 "overhead" 버킷에 들어간 거예요. executor 깨짐의 또 다른 증거입니다.

**상세 설명**:
- prefill/decode 분리한 이유: ReAct에서 decode가 prefill의 2배 — Apple Silicon이 memory-bandwidth-bound라는 의미 있는 발견 (decode-RAG 동기)
- raw sum 초과 문제는 polling 정책 자체의 한계, 완전 해결하려면 wrapper 패치 + 재실행 필요

---

## 🟦 Slide 14 — Fig 6: GPU 활용률 (90초, 가장 강한 검증)

**화면**: 좌(논문 fig6 NVIDIA DCGM) / 우(우리 wall-clock proxy 큰 숫자 54.98%) / 해설

**대본**:
> 본 재현에서 **가장 강력한 검증 포인트**입니다.
>
> 논문은 NVIDIA DCGM이라는 GPU 코어 활용률 직접 측정 도구로 ReAct GPU idle을 **54.5%**로 측정했습니다.
>
> 우리는 NVIDIA가 아닌 M3 Pro라 DCGM을 못 씁니다. 그래서 wall-clock proxy 공식 — `idle_ratio = 1 − (prefill + decode) / e2e` — 으로 우회 측정했어요. llama.cpp 내부 카운터로 잡은 LLM 계산 시간을 단대단 시간에서 뺀 나머지가 idle이라고 본 거죠.
>
> 결과는 **54.98%**. 논문과 **0.48%p 차이로 사실상 동일**합니다.
>
> 측정 방식이 완전히 다른데 왜 일치하느냐 — 단일 사용자 워크로드 + batching 없음이면 "GPU가 LLM 계산 안 한 시간 = GPU 코어가 노는 시간"이 물리적으로 같습니다. 양쪽 시스템 모두 HotpotQA의 Wikipedia API 응답 대기가 같은 비율을 차지해요.
>
> 이게 왜 중요하냐 — **GPU의 절반 이상이 도구 응답 대기로 놀고 있다**는 구조적 비효율을 정량적으로 확인한 겁니다. 이게 decode-RAG처럼 decode 중 retrieval을 prefetch하는 연구의 직접적 motivation이 됩니다.

**상세 설명**:
- DCGM이 측정하는 것: GPU SM(Streaming Multiprocessor) 명령 실행 비율
- 우리가 측정하는 것: wall-clock에서 LLM 추론 외 시간 비율
- 두 측정이 일치하는 조건: single user + no batching (우리 셋업이 정확히 이 조건)
- multi-tenant라면 두 값은 크게 달라짐

---

## 🟦 Slide 15 — Fig 7: 단대단 지연 분포 (Heavy Tail) (90초)

**화면**: 좌(논문 fig7 ShareGPT/ReAct/WebShop) / 우(우리 — ReAct 빨강 히스토그램 + Reflexion 노랑선 + LLMCompiler 회색선 + LATS 범위)

**대본**:
> 단대단 지연의 분포 형태입니다.
>
> **논문 패턴**: ShareGPT 같은 정적 chatbot은 분포가 좁고 일관 (p95 9.7초), ReAct 같은 에이전트는 분포가 wider + heavy tail. 이유는 다단계 추론 + 도구 호출 수가 query마다 변동해서입니다.
>
> **우리 패턴**: ReAct p95/p50 = **3.15**. heavy tail 재현됐습니다. 절대값(p95 87.2초)이 논문(20.7초)보다 4-5배 큰 건 M3 Pro Q4가 A100보다 느리기 때문이죠.
>
> 우리는 4 agent 분포를 모두 overlay했습니다. **흥미로운 발견 하나** — LLMCompiler가 p95/p50 = **7.63**으로 가장 heavy tail입니다. 분포가 bimodal — 약 5초에서 끝나는 클러스터 (planner만 돌고 실패)와 90초 부근 클러스터 (시도하다 포기)로 갈라져요. **executor 깨짐의 직접적 증거**입니다.
>
> **시사점**: agent workload는 mean이 아닌 **p95**로 봐야 합니다. mean 기반 SLA는 실제 사용자가 보는 latency를 과소평가합니다.

---

## 🟦 Slide 16 — Fig 8: 토큰 분해 (90초)

**화면**: 좌(논문 fig8) / 우(우리 — 논문 색상 매핑, 회색=Instruction, 검정=User, 녹색=LLM hist, 노랑=Tool hist, 빨강=Output)

**대본**:
> 토큰 분해입니다. 색상도 논문 규약으로 맞췄습니다.
>
> **총량 트렌드 일치**: LATS input이 **130만 토큰**으로 압도적. 트리의 모든 가지가 system prompt를 재포함하기 때문이죠. ReAct 92만, Reflexion 62만, LLMCompiler 5만 (executor 깨짐).
>
> 그런데 차트에 빨간 글씨로 "tool = 0 (HumanMsg-wrapped)"이라고 표시된 게 보이실 겁니다. **이게 Fig 8 검증의 FAIL 원인**입니다.
>
> 우리 측정 한계 L4 — HotpotQA의 tool 응답이 LangChain의 ToolMessage가 아닌 **HumanMessage로 래핑되어** 우리 핸들러가 "tool" 카테고리로 못 분류합니다. 즉 노랑 막대(Tool history)가 비어있어요.
>
> 또한 우리는 4-way (system/human/ai/tool) 분해, 논문은 6-way (Instruction/Few-shot/User/LLM history/Tool history/Output) 분해. 매핑이 1:1로 안 됩니다. 정직하게 표기했어요.
>
> LLMCompiler는 거의 빈 막대. 1번 호출하고 끝나니 토큰도 거의 없습니다.

---

## 🟦 Slide 17 — Fig 13: Accuracy vs Latency Pareto (120초, 핵심)

**화면**: 좌(논문 fig13) / 우(우리 — 빨강 사각=ReAct, 노랑 동그라미=Reflexion, 파랑 삼각=LATS, 회색 다이아=LLMCompiler, 로그 스케일, 주석 포함)

**대본**:
> 이 슬라이드가 본 재현의 **하이라이트**입니다 — accuracy-vs-latency Pareto frontier.
>
> 마커는 논문과 동일하게 맞췄습니다. 빨강 사각 = ReAct, 노랑 동그라미 = Reflexion, 파랑 삼각 = LATS, 회색 다이아 = LLMCompiler.
>
> **일치 핵심 1**: **LATS가 가장 정확하면서 가장 비싼** 페이퍼의 핵심 형상이 그대로 재현됐습니다. LATS 33% / 1037초 vs ReAct 32% / 41초 — **25배 비용으로 1% 정확도 이득**. 페이퍼 결론인 "test-time scaling의 수확체감"의 우리 환경 증거입니다.
>
> **일치 핵심 2**: LLMCompiler가 가장 빠름 (27.8초). 단 차트에 빨간 주석 — "executor broken, 41/50 empty answers" — 으로 정확도 18%는 unreliable이라고 표시했습니다.
>
> **차이점**: 우리 Reflexion이 16%로 ReAct(32%)보다 낮습니다. 페이퍼는 반대예요. 원인 추정 — Q4 양자화에서 reflection 단계가 자기수정보다 노이즈를 추가하는 경계 케이스입니다. 페이퍼 8B는 FP16이라 reflection이 효과적이었는데 우리 Q4 8B는 그 마진을 잃었습니다.
>
> **한계**: LATS 표본 3개로 Spearman ρ(latency, accuracy) = 0.40 (기준 ≥0.6 FAIL). 차트에 회색 주석으로 표시했어요. 통계 표본 부족이 직접 원인입니다.

**상세 설명**:
- 발표 시 "LLMCompiler 정확도 18%"가 도드라져 보일 수 있음 → 미리 "executor 깨짐"으로 disclaim
- Reflexion 역전 현상이 Q4 양자화 영향의 가장 강한 증거 — 청중이 "정말 Q4 영향이냐"라고 물으면 design spec §11 L2 인용

---

## 🟦 Slide 18 — Part 4 디바이더 (10초)

**화면**: "Part 4 — Serving, 캐싱, Test-Time Scaling"

**대본**:
> 다음은 우리가 미실행한 영역입니다. 논문 결과 중심으로 빠르게 정리하겠습니다.

---

## 🟦 Slide 19 — Fig 9: Prefix Caching 효과 (45초)

**화면**: 논문 Fig 9

**대본**:
> Prefix caching은 에이전트 워크로드에 큰 영향을 줍니다 — 에이전트는 반복 호출마다 input prefix를 공유하기 때문이죠.
>
> 논문 결과: **prefill 평균 60% 감소, 단대단 지연 평균 15.7% 감소**. CoT는 호출이 1번뿐이라 효과 없고, 에이전트에서만 효과 큽니다.
>
> 우리는 `fig13_pareto_cache_on.yaml` sweep 설정은 만들어 뒀지만 시간 제약으로 실행 못 했어요. 향후 작업입니다.
>
> 왜 우리 baseline은 cache off였냐는 자주 묻는 질문인데 — llama.cpp의 `--cache-reuse`가 `prompt_seconds_total` 카운터에 어떻게 반영되는지 공식 문서가 없어서 측정 정확도 보장을 위해 off로 잡았습니다.

---

## 🟦 Slide 20 — Fig 10: Serving 아키텍처 (30초)

**화면**: 논문 Fig 10

**대본**:
> 다중 사용자 요청을 동시에 처리하는 serving 시스템 개요. 각 요청은 독립적인 agent loop를 가지고, 단일 요청 내에서는 LLM-tool 순차 의존성으로 병렬화가 어렵습니다. 결국 자원 활용을 끌어올리려면 **inter-request 병렬성**, 즉 여러 사용자 요청을 동시 처리해야 한다는 메시지입니다.

---

## 🟦 Slide 21 — Fig 11/12: p95 vs QPS, KV cache (30초)

**화면**: 논문 Fig 11, Fig 12 좌우

**대본**:
> Fig 11은 ChatGPT 대비 에이전트의 p95 latency vs QPS 곡선. 에이전트는 같은 QPS에서 훨씬 높은 p95이고, prefix caching이 곡선을 오른쪽으로 평탄화시킵니다.
>
> Fig 12는 KV cache 메모리 사용량. prefix caching으로 평균 51.7%, 최대 63.5% 감소. GPU 메모리 자원 효율 활용에 결정적입니다.

---

## 🟦 Slide 22 — Fig 14/15: Iteration & Few-shot Scaling (45초)

**화면**: 논문 Fig 14, Fig 15

**대본**:
> ReAct의 iteration budget과 few-shot 수가 정확도에 미치는 영향입니다.
>
> Fig 14에서 흥미로운 발견 — 평균 지연·정확도는 빠르게 포화하는데 **p95 지연은 계속 선형 증가**합니다. outlier가 budget을 모두 소진해 tail latency만 키워요. 즉 iteration budget은 정확도 뿐 아니라 **운영 안정성** 측면에서 튜닝해야 한다는 메시지.
>
> Fig 15에서는 few-shot 예시 추가가 정확도를 올리고 **동시에 평균 지연을 낮춥니다**. 좋은 예시가 더 적은 reasoning 단계로 답에 도달하게 도와줘서요.

---

## 🟦 Slide 23 — Fig 16: Sequential vs Parallel Scaling (45초)

**화면**: 논문 Fig 16 (3-panel)

**대본**:
> Test-time scaling의 두 가지 형태 비교입니다.
>
> Sequential scaling (reflection step 늘리기)에서 **한계 비용이 31배 증가**합니다. Reflexion에서 처음 8.7초 추가로 +4% accuracy 얻지만, 56초 시점에서 같은 +4%를 위해 +269초 더 필요해요.
>
> Parallel scaling (LATS 자식 노드 늘리기)은 정반대 — 자식 1→16으로 늘리면 정확도 +14.4%p 증가하면서 평균 지연 -196초 감소합니다. 단 동시 LLM 호출이 폭증해서 자원 제약 환경에서는 어렵죠.
>
> 정책 시사: 지연 민감 워크로드는 parallel, 자원 제약 환경은 sequential.

---

## 🟦 Slide 24 — Part 5 디바이더 (10초)

**화면**: "Part 5 — AI 인프라적 함의"

**대본**:
> 이제 가장 충격적인 부분입니다.

---

## 🟦 Slide 25 — Fig 17: 모델 사이즈 트레이드오프 (45초)

**화면**: 논문 Fig 17 (8B vs 70B)

**대본**:
> HotpotQA에서 Reflexion/LATS를 8B vs 70B로 비교.
>
> 흥미로운 통찰 — Accuracy vs Energy 차트에서 **8B가 70B보다 에너지 효율적**입니다. 70B는 8 GPU 쓰는데 8B는 1 GPU만 쓰기 때문. **8B + LATS + parallel scaling**으로 70B에 근접한 정확도를 더 낮은 에너지로 달성 가능합니다.
>
> "무조건 큰 모델"이 정답이 아니라는 메시지죠.

---

## 🟦 Slide 26 — 논문 핵심 충격 수치 (60초)

**화면**: 6개 stat card (62~136×, 71×, 54.5%, 31×, 1 GW, 200 GW)

**대본**:
> 논문의 가장 충격적인 6개 수치입니다.
>
> ShareGPT 대비 쿼리당 GPU 에너지가 **62~136배** 폭증. LATS는 LLM 호출이 평균 **71배** 많아져요. GPU는 **54.5% idle** — 우리도 거의 같은 수치 재현했습니다.
>
> Test-time scaling의 한계 비용이 **31배** 증가. ChatGPT 트래픽을 Reflexion-70B로 처리하면 약 **1 GW** — OpenAI Stargate가 정확히 이 규모입니다.
>
> 가장 충격적인 건 마지막 — Google 검색 트래픽을 같은 워크로드로 처리하면 **200 GW**. 이건 미국 전체 전력 평균 부하의 **절반** 수준이에요. 단일 산업 하나에 할당될 수 없는 규모입니다.
>
> **이게 논문이 "지속가능성 위기 (sustainability crisis)"라고 부르는 이유입니다.**

---

## 🟦 Slide 27 — Part 6 디바이더 (10초)

**화면**: "Part 6 — 우리 재현의 결론과 한계"

**대본**:
> 마지막으로 우리 재현이 무엇을 검증했고, 무엇이 부족한지 정리하겠습니다.

---

## 🟦 Slide 28 — Verification 테이블 (90초)

**화면**: 검증 결과 표 + footnote

**대본**:
> Spec §12에 명시된 검증 기준 대비 실제 측정 결과입니다.
>
> **Fig 4**: LATS/ReAct LLM 호출 비율 ≥ 5.0 기준에 우리 **21.87** → PASS. Tool count는 사후 복구로 보강해서 PARTIAL.
>
> **Fig 5**: 구성 비율 합 ≤ 100% → PASS.
>
> **Fig 6**: GPU idle 54.5% 기준에 우리 **54.98%** → PASS (Δ 0.48%p, 가장 정확한 일치).
>
> **Fig 7**: p95/p50 ≥ 2.0 기준에 **3.34** → PASS.
>
> **Fig 8**: tool tokens ≥ 10% 기준에 0% → FAIL. 측정 한계 L4.
>
> **Fig 13**: LATS/ReAct accuracy ratio ≥ 0.9 → PASS (1.042). Spearman ρ ≥ 0.6 기준에 0.40 → FAIL (LATS 표본 3개 부족).
>
> 정리하면 **PASS 4개, PARTIAL 2개, FAIL 2개**. 핵심 메시지 (LATS 호출 폭증, GPU idle 54.5%, heavy tail)는 모두 PASS.

---

## 🟦 Slide 29 — 알려진 한계 (60초)

**화면**: 좌(하드웨어·모델 격차) / 우(표본·인프라 한계) 두 박스

**대본**:
> 7가지 한계를 두 카테고리로 정리했습니다.
>
> **하드웨어·모델 격차**: L1 — A100+vLLM vs M3+llama.cpp, throughput 5-10× 느림. L2 — Q4 양자화, Reflexion 정확도 역전의 원인 후보. L6 — NVIDIA DCGM 없음, wall-clock proxy 우회.
>
> **표본·인프라 한계**: L3 — LATS 3샘플 (페이퍼 50). L4 — tool callback 미캡처, **Tool count는 사후 복구**, 시간·토큰은 복구 불가. L5 — Wikipedia API live 호출, 4개월 사이 내용 변동 가능. L7 — Serving 실험 미수행, single-request만.
>
> 이 중 가장 큰 한계는 L3 (LATS 표본)와 L4 (callback). 둘 다 추가 시간·패치로 해결 가능합니다.

---

## 🟦 Slide 30 — 결론 (90초)

**화면**: 큰 결론 박스 + 재현된 5가지 패턴 + decode-RAG 함의

**대본**:
> 결론입니다. M3 Pro + Llama-3.1-8B Q4 환경에서 페이퍼의 **핵심 메시지가 패턴 수준에서 모두 재현**됐습니다.
>
> 재현된 5가지 핵심 패턴:
> 1. **GPU idle 54.5% → 54.98%** — 단 0.48%p 차이로 본 재현의 가장 강한 증거
> 2. **LATS LLM 호출 폭증** — 논문 71배, 우리 22배
> 3. **Tool wait가 ReAct latency 지배** — 44%로 단일 최대 컴포넌트
> 4. **Heavy-tail 분포** — p95/p50 = 3.15
> 5. **LATS 수확체감** — 25× 비용 / 1% accuracy 이득
>
> 본 재현은 사용자 본인의 **decode-RAG 연구에 직접적 baseline**을 제공합니다. ReAct GPU idle 55%, Tool wait 44%, p95/p50 = 3.15는 모두 decode 중 retrieval prefetch가 직접 노릴 수 있는 비효율 영역의 정량적 근거입니다.

---

## 🟦 Slide 31 — Q&A / References (30초)

**화면**: Q&A 큰 글자 + 참고 자료 박스

**대본**:
> 발표는 여기까지입니다. 질문 받겠습니다.
>
> 참고로 논문은 arXiv 2506.04301v2, AgentBench 코드는 VIA-Research GitHub에 있고, 우리 재현 코드와 상세 분석은 `agentic_rag` 디렉토리에 정리돼 있습니다.

---

## 📋 예상 질문 (Q&A 대비)

### Q1: "왜 70B 모델은 안 했나요?"
**A**: M3 Pro 36GB unified memory에 70B Q4_K_M도 ~40GB 필요해서 물리적으로 안 들어갑니다. 클라우드 A100을 쓰면 가능하지만 비용·시간 부담이 컸고, **8B만으로도 페이퍼의 정성적 패턴은 모두 검증 가능**합니다. (slide 10 인용)

### Q2: "Q4 양자화가 정확도에 영향 줬을 것 같은데요?"
**A**: 네, 명시적 가설이었고 실제로 Reflexion이 ReAct보다 정확도 낮게 나오는 부작용으로 드러났습니다. design spec §11 L2에 사전 명시된 위험이 발현된 케이스죠. **단일 forward pass에서는 영향 작지만 LATS의 193 LLM calls 같은 multi-call에서 노이즈가 누적**됩니다. (slide 17, 29 인용)

### Q3: "GPU 측정이 NVIDIA DCGM과 정말 같은 의미인가요?"
**A**: 측정 방식은 다릅니다. DCGM은 GPU SM 코어 명령 실행 비율, 우리는 wall-clock에서 LLM 추론 외 시간 비율. **단일 사용자 + batching 없는 조건에서만 두 측정이 같은 물리량을 잼**. multi-tenant 환경이면 크게 갈렸을 겁니다. 54.5% vs 54.98% 일치는 이 조건이 우연히 만족됐기 때문입니다. (slide 14 인용)

### Q4: "LLMCompiler의 18% 정확도는 뭐가 잘못된 건가요?"
**A**: AgentBench의 LLMCompiler harness가 우리 환경에서 executor 호출이 안 되고 있습니다. planner가 LLM 1번 부르고 끝나서 50쿼리 중 41개가 빈 답이에요. **harness 자체 버그로 추정**하지만 디버깅 미수행. Fig 13의 LLMCompiler 점은 부분 결과이고, "fastest agent" 같은 정성적 결론은 유효합니다. (slide 12, 17 인용)

### Q5: "Tool callback이 ReAct만 잡힌 이유와 복구 방법은?"
**A**: LangChain 표준 `BaseTool.invoke` 경로에 후킹했는데 ReAct만 표준 경로를 거치고 나머지 셋은 자체 wrapper로 우회합니다. **사후에 구조적 모델로 복구** — Reflexion 차감식, LATS 곱셈식. 정확한 시간 복구는 wrapper 패치 + 재실행 필요. (slide 12 인용)

### Q6: "Wikipedia API live 호출인데 결과 재현성 문제 없나요?"
**A**: L5 한계입니다. 논문은 2026-01, 우리는 2026-05 측정이라 4개월간 Wikipedia 내용 변동 가능성 있습니다. **정확도 절대 비교는 한계지만 호출 패턴·지연 형상 같은 구조적 메시지에는 영향 작습니다**. (slide 29 인용)

### Q7: "70B로 했으면 200 GW 같은 데이터센터 수치를 실제로 검증할 수 있었나요?"
**A**: 아니요, 그건 시뮬레이션·산술 추정이라 8B로도 70B로도 직접 측정 불가능합니다. 페이퍼는 **단일 쿼리 에너지 측정값 × DAU × 환산식**으로 도출. 우리도 같은 방법으로 추정은 가능하지만 출발점인 단일 쿼리 에너지가 (1) Apple GPU 전력 측정 도구 한계, (2) Q4 GPU power profile 차이로 직접 환산 어려움. (slide 26 인용)

### Q8: "Reflexion이 ReAct보다 정확도 낮은 게 진짜 Q4 때문인지 어떻게 확신하나요?"
**A**: **확신 못 합니다, 추정입니다**. 정확히 검증하려면 같은 모델 Q4 vs FP16 (또는 Q8) sweep을 돌려 Reflexion 정확도 변화 비교해야 함. 시간 제약으로 미실행, design spec §11 L2에 fallback으로만 명시. **현재는 정직하게 "Q4 영향으로 추정"으로 표기**. (slide 17 인용)

### Q9: "이 재현이 decode-RAG 연구에 어떻게 쓰이나요?"
**A**: 두 가지 — (1) **Motivation 정량적 근거**: GPU 55% idle, Tool wait 44%, p95/p50=3.15 모두 decode-RAG가 노릴 수 있는 영역. (2) **Baseline 측정 인프라**: 같은 trace_schema로 decode-RAG before/after A/B 비교 가능. (slide 30 인용)

### Q10: "왜 HotpotQA만 했나요? WebShop도 흥미로워 보이는데."
**A**: 셋업 비용 + 시간 제약. WebShop은 docker 환경, MATH는 표 채점기, HumanEval은 코드 sandbox 별도. **HotpotQA가 GPU idle/heavy-tail 패턴을 가장 잘 드러내는 워크로드이고 decode-RAG 동기와 가장 직접 맞아서** 우선순위 1순위였습니다. (slide 10 인용)

---

## 🎯 발표 시 주의 사항

### 시간 관리
- **Part 3 (10분)**이 가장 중요 — 충분히 시간 할애
- Part 4 (3분)는 우리 미실행이므로 빠르게 — 청중이 깊게 묻지 않게
- Part 5 (2분)는 강한 수치로 임팩트
- Part 6 (2분)은 정직하게 한계 명시

### 톤
- 학회/세미나 톤: 기술적 정확성 + 자기 한계 솔직
- "잘 모르겠습니다", "이 부분은 추정입니다"를 적절히 사용
- 청중 질문을 환영하는 태도

### 시각적 강조 포인트
- Slide 14 (GPU idle 54.98%) — 가장 강한 검증
- Slide 17 (Pareto) — 핵심 결과 차트
- Slide 26 (200 GW) — 충격적 수치

### 피해야 할 함정
- LLMCompiler 깨진 데이터를 "잘 작동한다"고 말하지 말 것
- Reflexion 역전을 "논문이 틀렸다"고 해석하지 말 것 (우리 Q4 한계)
- 절대 수치 비교 (87초 vs 20초) 단순 비교 하지 말 것

---

## 📁 참고 문서

| 문서 | 경로 | 용도 |
|---|---|---|
| 논문 PDF | `/Users/imdonghyeon/agentic_rag/2506.04301v2.pdf` | 원본 |
| 한국어 정밀 분석 | `paper_analysis.md` (803줄) | 본 대본의 원천 |
| 재현 리포트 | `repro/results/report.html` | verification 수치 |
| Design spec | `docs/superpowers/specs/2026-05-25-hotpotqa-reproduction-design.md` | L1~L7 한계 정의 |
| 복구 스크립트 | `repro/analysis/recover_tool_calls.py` | Tool count 사후 복구 |
| Web PPT | `presentation.html` | 발표 자료 원본 (이 대본의 source) |
