# 실험 방법론 정리 — *The Cost of Dynamic Reasoning* (arXiv 2506.04301v2)

> **본 문서의 목적**: 페이퍼가 "어떻게 실험을 설계하고 어떻게 진행했는지"를 한 곳에 모으고, 공개된 GitHub 리포지토리(`VIA-Research/AgentBench`)와의 일치/불일치를 §6에서 별도로 검증한다.
> 페이퍼 전체 풀이는 `paper_analysis.md` 참조 — 본 문서는 §III–§VI(방법론·실험·인프라 환산)에만 집중한다.
>
> **⚠️ 주의**: §1–§5는 페이퍼 본문 기준 기술이다. 실제 공개 코드와 다른 부분이 있으니 §6를 반드시 함께 읽을 것.

---

## §0. 한 페이지 요약

저자들은 **세 가지 변수 축**과 **두 가지 실험 시나리오**를 조합해 측정 매트릭스를 구성했다.

**변수 축**
1. **AI Agent (5종)** — CoT, ReAct, Reflexion, LATS, LLMCompiler
2. **Benchmark (4종 + ShareGPT baseline)** — HotpotQA, WebShop, MATH, HumanEval (+ ShareGPT 정적 chatbot 트래픽)
3. **Model size (2종)** — Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct

**측정 범주**
- 요청별 LLM/tool 호출 수, 입력·출력 토큰 분해
- 단대단 지연(평균·95%-ile) 및 Prefill/Decode/Tool/Idle 분해
- GPU 활용률 (DCGM 기반) 및 KV cache 메모리
- 요청당 GPU 에너지 (Wh/query) → 일별 GWh, 데이터센터 단위 MW/GW 환산
- Serving 시 처리량(QPS) 한계와 95%-ile latency

**실험 시나리오**
- (A) **Single-request** — 한 번에 한 쿼리, 워크플로우 분해 (§IV-A·B = 페이퍼 §4)
- (B) **Serving** — 다중 요청 동시, Poisson 트래픽 + vLLM continuous batching (§IV-C)
- (C) **Test-time scaling sweeps** — iteration budget, few-shot 수, reflection depth, child node 수, 모델 사이즈 (§V)
- (D) **Infrastructure projection** — 측정된 Wh/query를 ChatGPT/Google 규모 트래픽에 곱해 데이터센터 전력 환산 (§VI)

샘플 사이즈: test-time scaling sweep에서 **설계 포인트당 50 질문** 명시 (§V).

---

## §1. 실험 변수 상세

### 1.1 AI Agent 5종

| Agent | 핵심 메커니즘 | 본 페이퍼 기준 베이스 구현 | 저자 수정 |
|---|---|---|---|
| **CoT** [84] | Reasoning만, 도구 없음 | 표준 CoT prompting | — |
| **ReAct** [96] | Reason → Act → Observe 루프 | 공식 구현 [97] | — |
| **Reflexion** [72] | ReAct + verbal RL 형태의 reflection | 공식 구현 [71] | — |
| **LATS** [102] | MCTS 기반 트리 검색 | 공식 구현 [103] | **concurrent LLM inference + parallel tool invocation** 추가 (원본은 직렬) |
| **LLMCompiler** [31] | DAG 계획 + 비동기 tool streaming | 공식 구현 [30] | — |

(인용 번호는 페이퍼 references 기준.)

### 1.2 Benchmark 4종 + ShareGPT baseline

| Benchmark | 작업 유형 | 사용 도구 | 적용 에이전트 | 정확도 메트릭 |
|---|---|---|---|---|
| **HotpotQA** [92] | Multi-hop QA | Wikipedia search/lookup API [85] | 5종 전부 | Exact match |
| **WebShop** [94] | 온라인 쇼핑 | 인터랙티브 웹 navigation (search/click) | CoT 제외 4종 | Task score |
| **MATH** [25] | 수학 문제 | Wolfram Alpha API [86] + Python 계산기 | LLMCompiler 제외 4종 | Exact match (등가식 허용) |
| **HumanEval** [10] | 프로그래밍 | 자기생성 unit test 실행 | LLMCompiler 제외 4종 | All unit tests pass |
| **ShareGPT** [70] | 정적 chatbot | (도구 없음) | 단일 턴 baseline | — |

벤치마크별 에이전트 제외 사유:
- WebShop에서 **CoT 제외**: 도구 없이 웹과 상호작용 불가
- MATH/HumanEval에서 **LLMCompiler 제외**: DAG 기반 계획이 단계적 수학·코딩 reasoning에 부적합

### 1.3 하드웨어·소프트웨어 스택

| 항목 | 8B 실험 | 70B 실험 |
|---|---|---|
| GCP instance | `a2-highgpu-1g` | `a2-highgpu-8g` |
| GPU | NVIDIA A100 40GB × 1 | NVIDIA A100 40GB × 8 |
| vCPU / RAM | 12 / 85 GB | 96 / 680 GB |
| LLM serving | vLLM 0.6.6 (OpenAI 호환) | 동일 |
| Framework | PyTorch 2.6 + CUDA 12.8 | 동일 |
| Prefix caching | 별도 명시 없으면 **ON** | 동일 |

저자는 **TPU 등 다른 가속기에도 적용 가능한 architecture-agnostic 결론**임을 §III에서 명시.

---

## §2. 측정 인프라 (페이퍼 기준)

### 2.1 GPU 활용률
- **도구**: NVIDIA DCGM (Data Center GPU Manager) [51]
- **정의**: 실제 사용된 GPU 코어 비율
- **세분화**: Prefill(빨강) / Decoding(분홍) / Idle(검정) 3-way (Figure 6의 색상)

### 2.2 지연 분해
- **세분화**: 4 카테고리 — LLM(빨강), LLM+Tool overlap(분홍, LLMCompiler 전용), Tool(검정), Others(회색)
- **단위**: 각 단계별 timestamp 측정 후 누적 비율 계산
- **참조 figure**: Fig 5 (지연 분해 + 단대단 다이아몬드), Fig 6 (GPU 시간 분해)

### 2.3 KV cache 메모리
- **측정**: 평균과 최대값 (Figure 12)
- **비교**: prefix caching ON/OFF 두 조건

### 2.4 에너지
- **단위**: Wh/query (요청 1건당 GPU 에너지)
- **측정 방법**: 페이퍼 본문에 정확한 도구명(NVML, nvidia-smi 등)을 명시하지 않음. GPU 전력 × 실행 시간 적분으로 추정. → **§6에서 더 다룸**
- **Baseline 비교 단위**: ShareGPT 단일 턴 추론 대비 배수 (Table III)

### 2.5 Prefix caching
- 동일 워크로드를 **ON/OFF로 두 번 측정**해 차이 비교 (Fig 9, 11, 12)
- vLLM 자체의 prefix caching 기능 [32] 토글

### 2.6 Tool latency 실측치 (페이퍼 보고)
| 도구 | 평균 호출 지연 |
|---|---|
| Wikipedia API (HotpotQA) | ~1.2초 |
| WebShop 로컬 web | ~20 ms |
| Python interpreter (HumanEval) | 가변, GPU도 사용함 |
| Wolfram Alpha (MATH) | 외부 API |

---

## §3. 실험 시나리오 A — Single-Request (§IV-A·B)

### 3.1 셋업
- **트래픽**: 한 번에 한 요청, 다른 요청과 겹치지 않음
- **목적**: 워크플로우 자체의 비용 구조 분해

### 3.2 측정 지표 → 결과 figure 매핑

| 측정 | Figure | 핵심 발견 |
|---|---|---|
| 요청당 평균 LLM·Tool 호출 수 | Fig 4 | CoT 대비 9.2배(평균), LATS 71배(최대) |
| 지연 breakdown | Fig 5 | LLM 69.4% / Tool 30.2%, LLMCompiler overlap 18.2% |
| GPU 활용률 + runtime breakdown | Fig 6 | Tool 대기로 GPU **최대 54.5% idle** |
| 단대단 95%-ile 지연 | Fig 7 | ShareGPT 9.7s / HotpotQA ReAct 20.7s / WebShop ReAct 50.8s |
| 입력·출력 토큰 분해 | Fig 8 | 반복마다 input 누적, LATS는 root→node 경로만 |
| Prefix caching 효과 (지연) | Fig 9 | Prefill **60.1%↓**, 단대단 **15.7%↓** |

### 3.3 Prefix caching 메모리 효과
- 도구 보조 에이전트는 CoT 대비 평균 3.0배(최악 5.4배) GPU 메모리
- LATS는 **prefix caching으로 평균 64.8% 메모리 절약** (트리 확장의 KV cache 중복 제거)

---

## §4. 실험 시나리오 B — Serving (§IV-C)

### 4.1 시스템 아키텍처 (Figure 10)
```
Users
  │
  ▼
Server entrypoint
  │
  ├── LLM agent worker × N (각자 워크플로우 실행)
  │     │
  │     ├──→ vLLM backend (Scheduler + Engine)
  │     │      • Continuous batching [32, 98]
  │     │      • FCFS scheduler
  │     │
  │     └──→ Tools (Wikipedia, Wolfram, Python 등)
  │           • 비동기 처리
```

### 4.2 트래픽 모델
- **분포**: **Poisson** [47] — 사용자 도착이 독립 무기억 사건
- **이유**: 실서비스 트래픽의 표준 근사

### 4.3 스케줄러
- **vLLM 기본 FCFS** (first-come-first-served)
- **Batching**: continuous batching (token-level)

### 4.4 측정 → 결과 figure 매핑

| 측정 | Figure | 핵심 발견 |
|---|---|---|
| QPS 증가 시 95%-ile latency | Fig 11 | ShareGPT 6.4 / HotpotQA ReAct 2.6 / WebShop ReAct 1.2 (peak QPS) |
| Prefix caching의 처리량 효과 | Fig 11 | ShareGPT 1.03× / **ReAct 5.62×** 향상 |
| KV cache 메모리 (평균·최대) | Fig 12 | 평균 51.7%↓, 최대 63.5%↓ |
| 순차 vs 동시 처리량 비교 | (본문 수치) | HotpotQA 0.10→2.6 QPS (25×), WebShop 0.19→1.2 (6.2×) |

### 4.5 Peak QPS 정의
**95%-ile latency 곡선의 무릎점**(knee) — QPS를 더 올리면 꼬리 지연이 폭발적으로 증가하기 시작하는 지점.

---

## §5. 실험 시나리오 C — Test-Time Scaling Sweeps (§V)

설계 포인트 **하나당 50 sample question**으로 평균.

### 5.1 Sweep 매트릭스

| Sweep | 변수 | 값 | Figure |
|---|---|---|---|
| Sequential (Reflexion) | max reflection steps | 2, 4, 8, 16, 32 | Fig 16(a) |
| Sequential (LATS) | max reflection steps | 4, 8, 16, 32, 64, 128, 256, 512 | Fig 16(b) |
| Parallel (LATS) | child nodes per expansion | 1, 2, 4, 8, 16 | Fig 16(c) |
| Iteration budget (ReAct) | per-query budget cap | 페이퍼 X축 참조 | Fig 14 |
| Few-shot count (ReAct) | in-context examples | 0–5 | Fig 15 |
| Model size 비교 | params | 8B vs 70B | Fig 17 |

### 5.2 Sweep에서 관찰된 핵심 트레이드오프

**Sequential scaling의 수확체감 (Fig 16)**
- Reflexion: latency 16.9s → 25.6s (+8.7s)로 **+4% accuracy**
- 그러나 56.0s 시점부터 **같은 +4%를 얻으려면 +269.5s 필요** → **31배 더 비싼 한계 비용**

**Parallel scaling의 역전 (Fig 16c)**
- LATS child node 1→16: **+14.4%p accuracy, 동시에 평균 지연 −196.3s**
- 다만 **메모리·동시 LLM 호출 폭증** 비용 발생

**Iteration budget의 tail 문제 (Fig 14)**
- Mean latency는 포화하지만 **95%-ile은 선형 증가** (outlier task가 budget 다 소진)

**Few-shot의 비단조성 (Fig 15)**
- 적당량은 accuracy↑이면서 평균 latency↓ (문제를 더 빨리 풂)
- 과도하면 **accuracy까지 하락** (prompt 길이 초과)

**8B vs 70B (Fig 17)**
- 70B가 적은 단계로 같은 accuracy 도달
- 단 **8B는 단일 GPU라 에너지가 더 효율적** → 효과적 sweep으로 모델 사이즈 격차 일부 메움

### 5.3 정확도 측정 프로토콜
- **HotpotQA / MATH**: exact match (MATH는 등가 수식 변형 허용)
- **WebShop**: 벤치마크 정의 task score
- **HumanEval**: 모든 unit test 통과한 작업의 비율

---

## §6. 검증 결과 — Paper vs GitHub `VIA-Research/AgentBench`

> **본 섹션의 evidence source**: 2026-05-25 세션에서 `gh api`/curl로 GitHub `VIA-Research/AgentBench` 리포지토리(main 브랜치)를 직접 fetch해 확인. 모든 행에 구체적 출처(파일 경로/검색 결과 카운트) 동반.

### 6.1 검증 매트릭스

| # | 항목 | 페이퍼 § | 페이퍼 주장 | GitHub 실제 (확인된 사실) | 평가 |
|---|---|---|---|---|---|
| 1 | 에이전트 종류 수 | §III-A, Table I | 5종 (CoT 포함) | `src/agents/` 하위 **4 폴더**: `ReAct/`, `Reflexion/`, `LATS/`, `LLMCompiler/`. README도 supported types를 `["react", "reflexion", "lats", "llmcompiler"]`로만 명시 — **CoT 별도 구현 없음** | ⚠️ 불일치, 마이너 |
| 2 | 모델 백본 | §III-C | Llama-3.1-8B/70B-Instruct | `config.yaml` global.model: **`"Qwen/Qwen3-32B"`** | ⚠️ 불일치, **메이저** — 재현 시 override 필수 |
| 3 | LLM 서버 백엔드 | §III-C | vLLM 0.6.6 backend | README 명시: *"OpenAI-compatible LLM server. **We used vLLM**"* — 그러나 **vLLM 부팅 스크립트는 리포에 없음**, `host: localhost, port: 8000`만 가정 | ⚠️ 불일치, 메이저 — 별도 띄워야 함 |
| 4 | GPU util 측정 (DCGM) | §IV-A.2, Fig 6 | NVIDIA DCGM 사용 | GitHub code search `q='DCGM repo:VIA-Research/AgentBench'` → **0 hits** | 🔧 외부 instrumentation |
| 5 | 에너지 측정 (Wh/query) | §VI, Table III | 41–348 Wh 보고 | `NVML`=0, `nvidia-smi`=0, `pynvml`=0 hits | 🔧 외부 측정, 코드 비공개 |
| 6 | Poisson 트래픽 시뮬레이션 | §IV-C, Fig 10 | Poisson 분포 | `poisson` code search → **0 hits**. `run_*.py`는 sequential 샘플 iteration | 🔧 외부 driver 필요 |
| 7 | Continuous batching 설정 | §IV-C | vLLM continuous batching | 코드 내 batching 설정 없음 — vLLM 서버측 기본값에 암묵 의존 | ⚠️ 암묵적 의존 |
| 8 | Prefix caching 토글 | §IV-B, Fig 9·11·12 | ON/OFF 비교 | 코드 토글 없음 — vLLM 서버측 `--enable-prefix-caching` 플래그로 추정 | 🔧 vLLM 서버 옵션 |
| 9 | LATS concurrent 수정 | §III-A 명시 | concurrent LLM + parallel tool | `src/agents/LATS/model_client.py:1` `import asyncio`, `:114` `asyncio.create_task(llm.ainvoke(...)) for _ in range(n)`, `:115` `await asyncio.gather(*tasks)`, `:122` `achat_batch` | ✅ 일치 (확인됨) |
| 10 | 벤치마크 도구 통합 | §III-B | Wikipedia/Wolfram/Python/WebShop | `src/tools/` 하위에 각각 별도 폴더 존재 (별도 검증된 Explore 보고) | ✅ 일치 |
| 11 | Sweep 샘플 사이즈 | §V | 설계 포인트당 **50** | `config.yaml` global.**samples: 5** (실제 키 이름 `samples`, 기본값 5) | ⚠️ 불일치, override 필요 |
| 12 | 분석 노트북 | Fig 4–17 | figure 생성 코드 | 톱레벨 13개 항목 중 `*.ipynb` 없음. `trace.txt`만 출력 | 🔧 분석 별도 처리 |

**범례**
- ✅ 일치 (직접 확인)
- ⚠️ 불일치 — 코드와 페이퍼가 다름, 재현 시 명시적 조정 필요
- 🔧 외부 추정 — 페이퍼는 사용했다고 명시하지만 공개 코드에 없음

**Top-level 리포 구조** (확인됨):
```
.env_tmp   .gitignore   LICENSE   README.md
agent_bench.py   config.yaml   requirements.txt
run_lats.py   run_llmcompiler.py   run_react.py   run_reflexion.py
dataset/   src/   trace.txt
```

### 6.2 종합 해석

공개된 AgentBench 리포지토리는 **"에이전트 로직 + 벤치마크 통합" 평가 하네스**다. LangChain·LangGraph 위에 4개 에이전트를 깔끔하게 구현했고, 도구 통합도 페이퍼와 일치한다.

그러나 **페이퍼 본문의 핵심 인프라 측정값** — Wh/query, GPU util%, Prefill/Decode/Tool/Idle 분해, peak QPS — 은 공개 코드만으로 재현 불가능하다. GitHub code search로 확인된 부재 항목:

1. vLLM **서버 부팅·관리** 스크립트 — README는 "we used vLLM"만 언급, 사용자가 알아서 띄워야 함
2. **DCGM / NVML / nvidia-smi / pynvml** — 모두 0 hits (GPU 텔레메트리 수집 코드 부재)
3. **Poisson** — 0 hits (트래픽 시뮬레이터 부재)
4. Prefill/Decode/Tool/Idle **timestamp 분리 instrumentation** — 부재
5. **분석/플로팅 코드** (Fig 4–17 재생성) — `*.ipynb` 없음

이들은 페이퍼 작성 시점의 **사내 측정 하네스**로 존재했을 것이며, AgentBench 공개판에는 빠져있다.

### 6.3 검증에 사용한 커맨드 (재현용)

```bash
# 1. config.yaml의 모델·samples 설정 확인
curl -fsSL https://raw.githubusercontent.com/VIA-Research/AgentBench/main/config.yaml
#   → global.model: "Qwen/Qwen3-32B", global.samples: 5

# 2. 톱레벨 디렉터리·src/agents 하위 폴더
gh api repos/VIA-Research/AgentBench/contents | jq -r '.[].name'
gh api repos/VIA-Research/AgentBench/contents/src/agents | jq -r '.[].name'
#   → LATS, LLMCompiler, ReAct, Reflexion (CoT 없음)

# 3. LATS concurrent 증거
curl -fsSL https://raw.githubusercontent.com/VIA-Research/AgentBench/main/src/agents/LATS/model_client.py \
  | grep -nE "async|asyncio|achat|gather|create_task"
#   → line 1, 104, 114-115, 122, 131-132, 142, 152-153

# 4. 측정 인프라 부재 확인 (모두 0 hits)
for kw in DCGM NVML nvidia-smi pynvml poisson vllm.entrypoints; do
  echo -n "$kw: "
  gh api -X GET search/code -f q="$kw repo:VIA-Research/AgentBench" --jq '.total_count'
done
```

---

## §7. 사용자의 decode-RAG 연구 관점에서의 시사점

### 7.1 motivation으로 직접 인용 가능한 결과

`paper_analysis.md` §5.1.2 (Figure 6)에서 보고된 **GPU idle 최대 54.5%** 는 본 페이퍼가 측정한 "intra-request 직렬 의존성의 비용"이다. 이는 decode 중 다음 retrieval을 미리 예측해 overlap시키는 접근의 가장 강력한 motivation 근거가 된다.

LLMCompiler가 비동기 tool execution을 시도해도 **overlap이 전체 latency의 18.2%에만 그치는** 한계(Fig 5)도 함께 인용 가능하다 — 즉 plan-then-execute 방식의 천장이 어디에 있는지를 측정한 셈.

### 7.2 baseline 재현 시 주의사항

페이퍼와 같은 측정 결과를 자기 baseline으로 쓰려면, **§6의 🔧로 표시된 다섯 항목**을 직접 구축해야 한다:

1. vLLM 서버 + prefix caching 토글 가능한 부팅 스크립트
2. DCGM 폴링 기반 GPU util 로거
3. NVML 기반 GPU 에너지 적분기 (Wh/query)
4. Poisson arrival driver (lambda 가변)
5. Prefill/Decode/Tool/Idle timestamp 분리 instrumentation

이 중 (3)·(4)·(5)가 가장 부담이 크다. (3)은 `pynvml` + 주기 폴링으로 비교적 빠르게 가능, (5)는 vLLM/agent 양쪽에 hook이 필요하다.

### 7.3 GitHub 코드를 활용할 수 있는 범위

페이퍼와 정확히 같은 측정을 못해도, AgentBench는 다음에는 그대로 쓸 수 있다:
- **에이전트 4종의 동작 검증** — 코드가 페이퍼 알고리즘과 일치
- **벤치마크 통합** — Wikipedia·Wolfram·Python·WebShop tool wrappers 재사용
- **prompt template** — Reflexion `fewshots.py`(33KB), LATS task별 prompt 등

decode RAG 새 방법을 implement할 때 ReAct/Reflexion/LATS를 baseline 에이전트로 쓰고, **자기 측정 하네스**로 wrap하는 방식이 현실적이다.

---

## §8. 참고 자료

- 원본 페이퍼: `/Users/imdonghyeon/agentic_rag/2506.04301v2.pdf` (§III–§VI)
- 한국어 풀이: `/Users/imdonghyeon/agentic_rag/paper_analysis.md` (lines 225–665)
- GitHub: <https://github.com/VIA-Research/AgentBench> (main 브랜치, 2026-05-25 확인)
- 본 문서 §1–§5 evidence: paper_analysis.md + 원본 PDF deep-read
- 본 문서 §6 evidence: `gh api` + `curl` 직접 호출로 GitHub 리포 cross-check (검증 커맨드는 §6.3에 첨부)
- 작성일: 2026-05-25
