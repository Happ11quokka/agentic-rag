# 관련 문헌 & 비-Cohere 대체 벡터DB 조사

**작성일**: 2026-07-08  
**목적**: 2026.07.02 미팅 Keynote의 9편 논문에 대해 (1) 우리 baseline과의 유사도 및 벤치마크 구성, (2) Cohere 쿼터 회피 용 비-Cohere 사전임베딩 대형 코퍼스 조사  
**출처**: deep-research 워크플로우 (25개 claim, primary source 3-0 만장일치 검증)

---

## 임무 1: 9편 논문 × 벤치마크 & 검색 인프라 분석

### 핵심 맥락

우리 baseline:
- **에이전트**: ReAct (반복 multi-hop 검색)
- **검색 코퍼스**: Cohere `wikipedia-2023-11-embed-multilingual-v3` 
  - 전체: ~41.5M passages (1024-d embeddings)
  - 현재 범위: HotpotQA 개발셋 타이틀 subset
- **인덱스**: FAISS `IndexFlatIP`
- **쿼리 임베딩**: Cohere API `embed-multilingual-v3.0` ← **여기서 quota 차단**
- **평가 지표**: HotpotQA dev, EM / retrieval-time fraction (목표: decode 중 retrieval 지연 은닉)

**"Decode-RAG" 핵심**: 검색 요청이 도착하는 시점을 **사전에 예측**하거나 **speculative call** 해서, LLM 생성 중에 검색 지연을 숨기는 것. 따라서 우리와 유사한 시스템은 **(1) 대형 dense vector DB**, **(2) multi-hop retrieval** (여러 번 호출), **(3) retrieval latency를 target하는 최적화** 를 모두 가져야 합니다.

---

### 9편 논문 상세 정리

#### 1️⃣ **Predictive Prefetching for Retrieval-Augmented Generation**
- **저자**: Zhang et al. (2026)
- **논문ID**: arXiv:2605.17989
- **우리와 관련도**: ⭐⭐⭐ **가장 직접적 (우리와 동일 아이디어)**

**아이디어**:  
hidden state를 모니터링해서 다음 retrieval 필요 시점을 예측(retrieval predictor + context monitor) → 그 전에 미리 쿼리 임베딩/ANN 검색을 시작(self-speculation). LLM이 생성하는 동안 검색이 백그라운드에서 완료되는 구조.

| 항목 | 내용 |
|-----|------|
| **평가 벤치마크** | HotpotQA, 2WikiMultiHopQA (multi-hop QA)<br/>Natural Questions, TriviaQA (single-hop factual)<br/>RepoBench-P (repo-level code completion)<br/>QMSum (query-focused summarization)<br/>**총 6개 / 4가지 유형** |
| **검색 코퍼스** | Wikipedia (크기 미명시, 전체 위키라고 추정) |
| **임베딩 모델** | **Contriever** (768-d, 오픈소스) |
| **인덱스 타입** | **FAISS IVF** (large-scale ANN) |
| **검색 지연** | 125ms median |
| **Cohere 의존도** | ❌ 없음 (완전 Cohere-free) |

**우리 baseline과의 비교**:
- ✅ Dense Wikipedia DB, multi-hop QA → 구조 동일
- ✅ Prefetching으로 latency 은닉 → 우리 decode-RAG 개념과 일치
- ✅ Contriever 사용 → 비-Cohere 임베딩의 좋은 예시
- ⚠️ 모든 baseline이 동일 인프라 사용 → 비교 공정 (우리도 따라할 가치)

---

#### 2️⃣ **TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval**
- **저자**: Lin et al. (2025)
- **학회**: MLSys'26
- **논문ID**: arXiv:2502.20969
- **우리와 관련도**: ⭐⭐⭐ **구조 최근접**

**아이디어**:  
IVF 클러스터를 미리 CPU에서 GPU로 올려두고(lookahead clustering), 실제 쿼리 인코딩 전에 top cluster들을 미리 GPU 메모리에 prefetch → 사실상 retrieval latency의 cluster-load 부분을 숨김. LLM과 retrieval 병렬 실행.

| 항목 | 내용 |
|-----|------|
| **평가 벤치마크** | Natural Questions, HotpotQA, TriviaQA<br/>**각 512쿼리 샘플** (표준적 QA 벤치) |
| **검색 코퍼스** | **wiki_dpr** (Wikipedia)<br/>**21,015,300 passages**<br/>100-word chunk, Dec-2018 dump<br/>**2.1B tokens, 61GB index** |
| **임베딩 모델** | **Contriever** (768-d) |
| **인덱스 타입** | **FAISS IVF-Flat**<br/>4096 clusters<br/>Inner-product distance |
| **Cohere 의존도** | ❌ 없음 (완전 Cohere-free) |

**우리 baseline과의 비교**:
- ✅ **가장 직접 비교 가능** (dense Wikipedia, 정확한 passage count/size 제시)
- ✅ wiki_dpr은 표준 HuggingFace dataset (재현 용이)
- ✅ Contriever = 오픈소스 임베딩 (우리가 즉시 교체 가능)
- ✅ IVF-Flat 4096 = 대규모 ANN 인덱스 (our IndexFlatIP와 유사 개념)
- 📊 **이 논문의 setup을 우리 baseline의 검증 대상으로 삼을 것을 강력히 권장**

---

#### 3️⃣ **PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design**
- **저자**: Jiang et al. (2024)
- **학회**: KDD 2025
- **논문ID**: arXiv:2403.05676
- **우리와 관련도**: ⭐⭐ (아이디어는 겹치나, 대상 task가 다름)

**아이디어**:  
retrieval과 generation을 병렬화 (RETRO 스타일) → 토큰 디코딩 중에 다음 토큰들의 관련 문서를 미리 retrieval. 초대규모 문서DB(~3B chunks)에서 효율적 prefetching.

| 항목 | 내용 |
|-----|------|
| **평가 벤치마크** | **QA가 아님** — Language Modeling perplexity<br/>- Wikipedia: 13.47<br/>- RealNews (C4 subset): 14.87<br/>- C4 English: 19.36<br/>(표 5에 supplementary QA 실험 있으나 부수적) |
| **검색 코퍼스** | C4 corpus (deduplicated English)<br/>64-token chunks → **~3 billion chunks**<br/>~192 billion tokens<br/>(LM 학습용, 매우 큼) |
| **임베딩 모델** | **all-MiniLM-L6-v2** (384-d, 오픈소스) |
| **인덱스 타입** | **FAISS IVF-PQ** (Inverted-File + Product Quantization)<br/>nlist=16384<br/>384GB CPU 서버 호스팅 |
| **Cohere 의존도** | ❌ 없음 (completely open) |

**우리 baseline과의 비교**:
- ✅ 대규모 ANN 인덱스 (가장 큰 규모)
- ✅ Prefetch 개념 (우리와 공통)
- ❌ **Task가 다름** (LM perplexity vs. QA)
  - 학습 데이터에서 긴 context dependency 학습 vs. 일회성 질문 응답
  - Retrieval 패턴이 다름 (학습 중 implicit vs. inference 중 explicit)
- ⚠️ **모델링 복잡도**: 384GB 서버 호스팅 필요 (우리 로컬 환경과 맞지 않을 수 있음)

**우리 프로젝트에서의 위치**: 아이디어 (prefetch) 는 유사하나, 실제 구현/eval은 TeleRAG나 Predictive Prefetching을 따라가는 것이 낫습니다.

---

#### 4️⃣ **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**
- **저자**: Jin et al.
- **출판**: ACM Transactions on Computer Systems (ToCS) 2025
- **논문ID**: arXiv:2404.12457
- **우리와 관련도**: ⭐⭐ (시스템 최적화이나, 소규모 코퍼스)

**아이디어**:  
retrieved document들의 KV-cache를 tiered하게 관리 (GPU → CPU → disk) → 자주 재사용되는 doc들은 GPU에 유지. 일종의 retrieval-aware 캐싱.

| 항목 | 내용 |
|-----|------|
| **평가 벤치마크** | MMLU (4지선다 multiple-choice)<br/>Natural Questions (일반 팩트 QA)<br/>**2가지** |
| **검색 코퍼스** | Wikipedia **인기 페이지만** (most popular)<br/>**~300,000 documents** ← **매우 소규모** |
| **임베딩 모델** | **OpenAI text-embedding-3-small**<br/>(유료 API, 비-Cohere이지만 여전히 외부 의존) |
| **인덱스 타입** | FAISS IVF (1024 clusters)<br/>또는 HNSW<br/>Top-k=2 (매우 적음) |
| **Cohere 의존도** | ❌ Cohere 아님<br/>⚠️ 하지만 OpenAI API 의존 |

**우리 baseline과의 비교**:
- ❌ **코퍼스가 너무 작음** (0.3M vs. 우리 41.5M)
  - Small-world retrieval → retrieval latency가 매우 짧음
  - retrieval-time fraction 실험에 부적합
- ❌ **QA 벤치마크 수가 적음** (2개만)
- ✅ 시스템 최적화 아이디어는 참고 가능
- ⚠️ OpenAI API 의존 (또 다른 외부 의존성)

**우리 프로젝트에서의 위치**: 비슷한 크기의 작은 코퍼스가 필요한 경우만 참고.

---

#### 5️⃣ **FLARE (Active Retrieval Augmented Generation)**
- **저자**: Jiang et al.
- **출판**: EMNLP 2023
- **논문ID**: arXiv:2305.06983
- **우리와 관련도**: ⭐⭐ (개념적 조상, 하지만 retriever가 다름)

**아이디어**:  
생성 중 각 토큰 직후 **"다음 문장이 retrieval이 필요한지 여부"를 예측** (active retrieval). 필요하면 생성 멈추고 retrieval 수행. 순환적 multi-hop 구조.

| 항목 | 내용 |
|-----|------|
| **평가 벤치마크** | 2WikiMultiHopQA (multi-hop QA)<br/>StrategyQA (commonsense reasoning)<br/>ASQA (long-form QA)<br/>WikiAsp (open-domain summarization)<br/>**4가지** |
| **검색 코퍼스** | DPR Wikipedia corpus (Karpukhin et al. 2020)<br/>~21M passages |
| **Retriever 타입** | **BM25** (lexical/sparse, NOT dense)<br/>+ Bing Search API (WikiAsp) |
| **임베딩 인덱스** | ❌ **없음** (dense vector DB 미사용) |
| **Cohere 의존도** | ❌ Cohere 없음<br/>⚠️ Elasticsearch (BM25 서버 필요) |

**우리 baseline과의 비교**:
- ✅ **Active retrieval 개념** (우리의 decode-RAG 영감)
- ✅ Multi-hop QA 벤치마크
- ✅ 21M Wikipedia corpus (크기 적당)
- ❌ **Dense retrieval 아님** (BM25)
  - Embedding 모델 불필요 → 비교 대상으로 부적합
  - Retrieval 지연 특성이 완전히 다름
- ⚠️ Bing API 의존 (또 다른 외부 서비스)

**우리 프로젝트에서의 위치**: 
- "Active retrieval" 개념은 학습 (우리도 언제 retrieval할지 예측)
- 실제 구현은 dense DB를 사용해야 함

---

#### 6️⃣ **Speculative RAG: Enhancing RAG through Drafting**
- **저자**: Wang et al.
- **출판**: ICLR 2025
- **논문ID**: arXiv:2407.08223
- **우리와 관련도**: ⭐ (개념적 영감, 구조는 다름)

**아이디어**:  
작은 "drafter" 모델이 주어진 문서 subset에서 빠르게 초안을 작성 → 큰 "verifier"가 검증하고 재생성. Speculative decoding의 RAG 버전.

| 항목 | 내용 |
|-----|------|
| **평가 벤치마크** | TriviaQA, PopQA (단일 문서 QA)<br/>MuSiQue (multi-hop QA)<br/>PubHealth (biomedical claim verification)<br/>ARC-Challenge (science multiple-choice)<br/>**5가지** |
| **검색 코퍼스** | Self-RAG (Asai et al. 2023)로부터 상속<br/>(자체 명시 없음) |
| **Retriever** | Self-RAG의 retriever 사용<br/>Top-10 문서 (MuSiQue에서 top-15)<br/>(매우 적음 — full ranking 아님) |
| **자체 인덱스** | ❌ 없음<br/>InBedder_Roberta는 **이미 retrieval된 doc들을 클러스터링**용<br/>(새로운 query에 대한 ANN 인덱스 아님) |
| **Cohere 의존도** | ❌ Cohere 없음 |

**우리 baseline과의 비교**:
- ✅ Speculative decoding 개념 (우리의 decode-RAG도 speculative)
- ❌ **자체 대형 dense vector DB 없음** (Self-RAG 상속)
- ❌ Top-k retrieval만 사용 (ANN ranking 없음)
- ❌ Retrieval latency를 최적화 대상으로 하지 않음

**우리 프로젝트에서의 위치**: 
- Speculative 개념 참고
- 실제로 우리가 비교할 대상 아님

---

#### 7️⃣ **Speculative Actions: A Lossless Framework for Faster Agentic Systems**
- **저자**: Ye et al.
- **논문ID**: arXiv:2510.04371
- **우리와 관련도**: ⭐⭐ (agentic이나, retrieval DB 없음)

**아이디어**:  
agentic loop에서 LLM의 다음 action(tool call)을 **"추정"해서 미리 실행** → 실제 LLM output과 비교 (lossless: 잘못 예측해도 무조건 검증). 평균 -48.5% latency.

| 항목 | 내용 |
|-----|------|
| **평가 환경** | Chess (TextArena, turn-based gameplay)<br/>E-commerce retail (tau-bench)<br/>**HotpotQA (live Wikipedia API call, ReAct Search/Lookup/Finish)**<br/>OS hyperparameter tuning (sysbench)<br/>**4가지** |
| **HotpotQA 설정** | **Live Wikipedia API 호출**<br/>(classical ReAct-style, no embedding corpus) |
| **Vector DB / 인덱스** | ❌ **없음** |
| **Cohere 의존도** | ❌ Cohere 없음<br/>⚠️ Wikipedia API 의존 |

**우리 baseline과의 비교**:
- ✅ HotpotQA multi-hop QA
- ✅ Agentic loop의 latency 최적화 (우리도 목표)
- ❌ **Live Wikipedia API** (우리는 dense vector DB)
  - Retrieval 특성 완전히 다름 (API latency vs. local ANN latency)
  - Network round-trip time이 dominates (우리는 reduction 가능, 여긴 불가)
- ❌ Dense retrieval latency 실험 가치 없음

**우리 프로젝트에서의 위치**: 
- Speculative execution 아이디어 (우리도 같은 방향)
- 하지만 retrieval infrastructure가 안 맞아서 직접 비교 불가

---

#### 8️⃣ **PASTE (Pattern-Aware Speculative Tool Execution) / "Act While Thinking"**
- **저자**: Sui et al.
- **기관**: Shanghai Jiao Tong University + Microsoft Research
- **논문ID**: arXiv:2603.18897
- **공개일**: 2026-03
- **우리와 관련도**: ⭐⭐ (agent tool prefetch, 아이디어만)

**아이디어**:  
agent trace에서 도구(tool) 호출 패턴을 학습 → 다음 tool call을 예측해서 **미리 실행 시작**. LLM thinking 중에 tool 실행이 백그라운드에서 진행. Adaptive termination으로 overspeculation 회피.

| 항목 | 내용 |
|-----|------|
| **평가 에이전트 시스템** | VirtualLab (science-focused)<br/>Qwen-DeepResearch (deep research)<br/>gemini-cli (code/general-purpose)<br/>**3가지** |
| **평가 벤치마크** | DeepResearchBench<br/>SWE-bench<br/>ScholarQA<br/>**3가지 / research/coding 태스크** |
| **도구 종류** | Web search, file edit, code execution<br/>(구체적 corpus 미명시) |
| **Vector DB** | ❌ 없음 (search tool이 있으나, retrieval DB를 자체 호스팅하지 않음) |
| **Cohere 의존도** | ❌ 없음 |

**우리 baseline과의 비교**:
- ✅ Tool/action prefetch 개념 (우리의 retrieval prefetch와 유사)
- ✅ Adaptive termination (overthinking 회피, 우리도 고려)
- ❌ **QA 벤치마크 없음** (research/coding task)
- ❌ Dense vector DB 평가 없음 (retrieval 특성이 일반화되지 않음)

**우리 프로젝트에서의 위치**: 
- Tool prefetch의 일반적 패턴 참고 (우리는 retrieval 특화)
- 평가 메트릭 설계 참고 (기존 agent framework의 측정 방식)

---

#### 9️⃣ **Fast Inference from Transformers via Speculative Decoding**
- **저자**: Leviathan et al.
- **출판**: ICML 2023 (PMLR)
- **우리와 관련도**: ☆ (배경 개념, RAG와 무관)

**아이디어**:  
작은 "drafter" 모델이 빠르게 여러 토큰 예측 → 큰 "verifier"가 한 번에 검증. 디코딩 시간 30~40% 단축. (Speculative RAG의 원점)

| 항목 | 내용 |
|-----|------|
| **평가 태스크** | WMT English-German translation (T5-XXL)<br/>CNN/DailyMail summarization<br/>Language modeling (LM1B)<br/>**LLM 디코딩 성능** |
| **Retrieval 관련** | ❌ **없음** (pure LLM decoding) |
| **Cohere 의존도** | N/A |

**우리 프로젝트에서의 위치**: 
- 개념적 토대만 제공 (우리의 speculative retrieval은 이것의 확장)
- 직접 비교 대상 아님

---

### 1번 임무 요약 표

| # | 논문 | 관련도 | 벤치마크 수 | Dense VectorDB? | QA 중심? | 우리와 비교가능? | 추천 |
|---|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Predictive Prefetching (2026) | ⭐⭐⭐ | 6개 | ✅ | ✅ | ✅✅ | **#1 비교대상** |
| 2 | TeleRAG (2025) | ⭐⭐⭐ | 3개 | ✅ | ✅ | ✅✅ | **#1 비교대상** |
| 3 | PipeRAG (2024) | ⭐⭐ | 1개 (LM) | ✅ | ❌ | △ | 아이디어만 |
| 4 | RAGCache (2025) | ⭐⭐ | 2개 | ✅ | ✅ | △ (소규모) | 참고용 |
| 5 | FLARE (2023) | ⭐⭐ | 4개 | ❌ (BM25) | ✅ | ❌ | 아이디어만 |
| 6 | Speculative RAG (2025) | ⭐ | 5개 | ❌ (inherited) | ✅ | ❌ | 아이디어만 |
| 7 | Speculative Actions (2025) | ⭐⭐ | 4env | ❌ (live API) | △ | ❌ | 아이디어만 |
| 8 | PASTE (2026) | ⭐⭐ | 3env | ❌ | ❌ | ❌ | 아이디어만 |
| 9 | Speculative Decoding (2023) | ☆ | - | - | ❌ | ❌ | 배경만 |

**직접 비교 대상**: **1번 (Predictive Prefetching)** + **2번 (TeleRAG)**  
**아이디어 참고**: 3, 5, 7, 8

---

---

## 임무 2: Wikipedia 비-Cohere / 대형 벡터DB 대체 옵션

### 핵심 맥락

현재 우리의 quota 차단 지점:
- **문서 임베딩** (인덱스 빌드): Cohere Wikipedia 2023-11의 사전계산 벡터 사용 → **빌드 시점에 API 불필요**
- **쿼리 임베딩** (런타임): 각 쿼리마다 Cohere API 호출 → **trial key 1k/month 쿼터 부족** (현재 139/1405에서 블락)

따라서 해결책은 **"비-Cohere 쿼리 인코더"를 쓰는 것** 이며, 크게 두 가지 전략:

**전략 A (권장)**: 로컬/오픈소스 임베딩 모델 사용  
- 쿼리 인코딩이 로컬에서 즉시 실행 (API 호출 0)
- Network latency 제거 → retrieval-time fraction이 ~2%로 떨어짐 ⚠️ 
- Trade-off: fraction 메트릭을 다시 정의해야 함 (로컬 ANN 검색 시간 + embedding 시간)

**전략 B (보조)**: 다른 임베딩 API 사용 (OpenAI, Jina, etc.)  
- Network latency 유지 → fraction 성질 유지 ✅
- 다시 API/과금 의존 ⚠️

**이 문서의 포커스**: 전략 A (비-Cohere, 로컬/오픈)

---

### A. 즉시 사용 가능한 대형 사전임베딩 Wikipedia DB

#### ① `facebook/wiki_dpr` ✅ **#1 권장**

| 항목 | 값 |
|-----|-----|
| **HuggingFace ID** | `facebook/wiki_dpr` |
| **Passages 수** | 21,015,300 |
| **Passage 규격** | 100-word chunks<br/>December 2018 Wikipedia dump |
| **Corpus 크기** | ~2.1 billion tokens<br/>FAISS index: 61GB |
| **임베딩 모델** | DPR (Dense Passage Retrieval) 768-d<br/>두 가지 variant:<br/>- `nq` (Natural Questions 학습)<br/>- `multiset` (multi-dataset) |
| **사전계산 여부** | ✅ **YES** (모든 passages에 embedding 포함) |
| **인덱스 형태** | 선택가능:<br/>- DPR 사전임베딩 + 메타데이터<br/>- FAISS index 옵션 존재 |
| **오픈소스** | ✅ YES (Meta, Apache-2.0) |
| **로컬 인코딩** | ❌ 쿼리는 DPR 모델로 로컬 인코딩 필요<br/>(facebook/dpr-question_encoder-*) |
| **Download 링크** | https://huggingface.co/datasets/facebook/wiki_dpr |

**왜 #1 권장인가**:
1. **정확히 TeleRAG 논문과 동일 코퍼스** (재현 최고)
2. **21M passages = 우리 41.5M과 비슷한 규모** (비교가능)
3. **표준 공개 데이터셋** (장기적 안정성)
4. **명확한 임베딩 스펙** (768-d DPR)
5. **HuggingFace에서 한 줄로 다운로드**

**사용법 스케치**:
```python
from datasets import load_dataset
wiki_dpr = load_dataset("facebook/wiki_dpr", "multiset")  
# or "nq" depending on your task
# Returns: dataset with columns ["id", "title", "text", "embeddings"]

# 쿼리 인코딩 (로컬):
from transformers import AutoTokenizer, AutoModel
qencoder = AutoModel.from_pretrained("facebook/dpr-question_encoder-multiset")
qtokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset")
query_emb = qencoder(**qtokenizer("your query", return_tensors="pt"))
# 768-d vector

# FAISS 인덱스 빌드 (이미 사전계산된 벡터 사용):
import faiss
index = faiss.IndexFlatIP(768)  # Inner product
all_embeddings = [ex["embeddings"] for ex in wiki_dpr]
index.add(np.array(all_embeddings))
```

---

#### ② Contriever / mContriever (over wiki_dpr) ✅ **#1 짝**

| 항목 | 값 |
|-----|-----|
| **모델명** | `facebook/contriever` (English)<br/>`facebook/contriever-msmarco` (MS MARCO 튜닝)<br/>`facebook/mcontriever` (multilingual) |
| **모델 크기** | 768-d embeddings |
| **라이선스** | ✅ CC-BY-NC (메인), MIT (contriever-msmarco) |
| **특징** | - Contrastive learning (in-batch negatives)<br/>- Wikipedia 관련 문서쌍으로 학습<br/>- Dense retrieval의 SOTA |
| **사용 대상 코퍼스** | facebook/wiki_dpr의 passages<br/>(또는 임의의 passage corpus) |
| **임베딩 차원** | 768-d |
| **FAISS 인덱스** | TeleRAG: IVF-Flat 4096 clusters<br/>내적 거리 (inner-product) |
| **Download 링크** | https://huggingface.co/facebook/contriever |

**왜 Contriever인가**:
1. **정확히 TeleRAG 논문이 사용** (우리와 비교 가능)
2. **메타의 오픈소스** (신뢰도)
3. **Cohere와 다른 vendor** (quota 회피)
4. **768-d = DPR과 동일 차원** (기존 인프라 호환)
5. **Self-RAG, RAGCache 등도 사용** (좋은 신호)

**사용법 스케치**:
```python
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 모델 로드
model = AutoModel.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

# 문서 임베딩 (배치)
docs = ["doc1 text", "doc2 text", ...]
doc_embs = model(**tokenizer(docs, padding=True, truncation=True, return_tensors="pt"))
# Shape: (len(docs), 768)

# 쿼리 임베딩
query_emb = model(**tokenizer("query text", return_tensors="pt"))
# Shape: (1, 768)

# FAISS 인덱스
import faiss
d = 768
nlist = 4096  # TeleRAG 기준
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(np.array(doc_embs, dtype=np.float32))
index.add(np.array(doc_embs, dtype=np.float32))

# 검색
scores, indices = index.search(np.array([query_emb], dtype=np.float32), k=10)
```

---

#### ③ `Upstash/wikipedia-2024-06-bge-m3`

| 항목 | 값 |
|-----|------|
| **HuggingFace ID** | `Upstash/wikipedia-2024-06-bge-m3` |
| **Corpus** | June 2024 Wikipedia dump (multilingual) |
| **총 벡터 수** | ~144 million (11개 언어) |
| **영어 only** | ~47,018,430 vectors |
| **임베딩 모델** | BGE-M3 (multilingual, 1024-d) |
| **사전계산** | ✅ YES |
| **형태** | HuggingFace dataset (columnar) |
| **다운로드** | https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3 |

**평가**:
- ✅ 매우 큼 (47M 영어)
- ✅ BGE-M3 = 좋은 다국어 임베딩
- ⚠️ 1024-d (DPR/Contriever 768-d와 다름, 호환성 고려)
- ⚠️ HuggingFace 카드 기반 정보 (내용 실제 확인 권장)

---

#### ④ `maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2` ⚠️

| 항목 | 값 |
|-----|-----|
| **HuggingFace ID** | `maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2` |
| **Corpus** | English Wikipedia 2022-12 (우리가 원래 쓰던 코퍼스 버전과 같음) |
| **Rows** | 35,167,920 passages |
| **임베딩 모델** | all-MiniLM-L6-v2 (384-d)<br/>(PipeRAG도 이 모델 사용) |
| **사전계산** | ✅ YES |
| **형태** | Parquet |
| **다운로드** | https://huggingface.co/datasets/maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2 |

**평가**:
- ✅ Passage count 우리와 비슷 (35M vs 41.5M)
- ✅ 384-d MiniLM (가벼움, PipeRAG와 호환)
- ✅ Wikipedia-22-12 = 우리가 쓰던 버전
- ⚠️ 384-d는 768-d보다 정보 손실 가능 (quality trade-off)
- ⚠️ 임베딩 성능이 DPR/Contriever보다 낮을 수 있음

**특별 주의**: 이 코퍼스는 우리가 원래 쓰던 **wikipedia-22-12과 동일**하므로, Cohere 임베딩만 대체하면 **완벽한 비교 가능**. 단, 품질이 DPR/Contriever보다 떨어질 가능성.

---

#### ⑤ `NeuML/txtai-wikipedia`

| 항목 | 값 |
|-----|-----|
| **설명** | Self-contained txtai ANN index over English Wikipedia |
| **임베딩 모델** | e5-base (768-d) |
| **형태** | txtai index (binary, 즉시 검색 가능) |
| **다운로드** | https://huggingface.co/NeuML/txtai-wikipedia |

**평가**:
- ✅ E5 모델 (좋은 성능)
- ✅ 즉시 사용 가능 (인덱스 빌드 불필요)
- ⚠️ txtai 라이브러리 의존 (FAISS와 다른 API)
- ⚠️ 메타데이터 (title, offset) 확인 필요

---

#### ⑥ `Supabase/wikipedia-en-embeddings`

| 항목 | 값 |
|-----|-----|
| **설명** | Wikipedia embeddings with pgvector (SQL-ready) |
| **임베딩 모델들** | all-MiniLM-L6-v2 (384-d)<br/>GTE-small (384-d)<br/>OpenAI text-embedding-3-small |
| **형태** | pgvector 포맷 (PostgreSQL) |
| **다운로드** | https://huggingface.co/datasets/Supabase/wikipedia-en-embeddings |

**평가**:
- ✅ 여러 임베딩 모델 (선택지)
- ✅ SQL 호환 (프로덕션용)
- ⚠️ OpenAI 임베딩 포함 (우리는 로컬/오픈을 원함)
- ⚠️ pgvector 설정 필요 (FAISS보다 복잡)

---

### B. 다른 대형 코퍼스 (임무1 벤치마크에 등장)

#### ⑦ **MS MARCO passage corpus**

| 항목 | 값 |
|-----|-----|
| **소스** | BeIR benchmark collection |
| **HuggingFace ID** | `BeIR/msmarco` |
| **Passages** | 8,841,823 passages |
| **Corpus** | Real search engine queries & clicked passages (Microsoft) |
| **특징** | QA task가 아님 (passage ranking / IR task) |
| **임베딩 모델** | 독립적 선택 (Contriever, E5, DPR 등 가능) |
| **참고논문** | TeleRAG는 안 씀 (NQ/HotpotQA/TriviaQA만) |

**평가**:
- ✅ 크기 괜찮음 (8.8M)
- ❌ QA task가 아님 (우리 목표와 맞지 않음)

---

#### ⑧ **C4 / RETRO-Pile 스타일 (PipeRAG)**

| 항목 | 값 |
|-----|-----|
| **소스** | amazon-science/piperag GitHub |
| **Corpus** | C4 + RealNews (MassiveText-style) |
| **Chunks** | ~3 billion (64-token chunks) |
| **임베딩** | all-MiniLM-L6-v2 384-d |
| **인덱스** | FAISS IVF-PQ, 384GB 호스팅 |

**평가**:
- ✅ 초대형 (우리 실험 규모의 한계)
- ❌ LM 작업용 (QA와 다름)
- ❌ 384GB 서버 필요 (우리 로컬 환경 초과)

---

#### ⑨ **Pyserini Prebuilt Indexes** (한 줄 다운로드)

| 항목 | 값 |
|-----|-----|
| **소스** | Castorini의 pyserini |
| **제공 인덱스** | Wikipedia (BM25 + dense:<br/>- `dpr` (DPR)<br/>- `ance`<br/>- `bge`)<br/>MS MARCO<br/>BEIR 전체 |
| **한 줄 사용** | `python -m pyserini.download --corpus wikipedia-dpr` |
| **형태** | 직바로 검색 가능 (wrapper 제공) |
| **다운로드** | https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md |

**평가**:
- ✅ 매우 편함 (한 줄 다운로드)
- ✅ 다양한 임베딩 옵션 (DPR, BGE, ANCE)
- ✅ 빠른 baseline 개발에 최적
- ⚠️ Dense index의 크기/세부사항 확인 필요

---

### 2번 임무 권장순서

| 우선도 | 옵션 | 코퍼스 크기 | 임베딩 모델 | 특징 | 추천 용도 |
|:---:|------|:---:|:---:|---|---|
| 🥇 | `facebook/wiki_dpr`<br/>+ DPR query encoder | 21M | DPR 768-d | **TeleRAG 정확 재현** | Primary baseline |
| 🥇 | `facebook/wiki_dpr`<br/>+ Contriever | 21M | Contriever 768-d | **TeleRAG 및 Predictive Prefetching과 호환** | Embedding 교체 실험 |
| 🥈 | `maloyan/wikipedia-22-12-all-MiniLM` | 35M | MiniLM 384-d | **우리 원래 코퍼스 버전 대체** | Cohere-only 교체 |
| 🥈 | `Upstash/wikipedia-2024-06-bge-m3` | 47M (영어) | BGE-M3 1024-d | **최신, 대형** | Modern alternative |
| 🥉 | Pyserini prebuilt | 21M~8.8M | 다양 | **빠른 프로토타이핑** | Quick baseline |
| ⚠️ | OpenAI API | 가변 | text-embedding-3-small | **Network latency 유지** | Fraction 메트릭 보존 (보조) |

---

### ★ 우리 프로젝트의 함정: Retrieval-Time Fraction

**원래 상황**:
- Cohere API 쿼리 인코딩 → network round-trip (~50-200ms 추정)
- 이 네트워크 지연을 **의도적으로** retrieval에 포함시켜 fraction을 >2%로 부풀림
- Baseline decode-RAG를 의미있게 비교하기 위한 설정

**비-Cohere 로컬 임베딩으로 교체하면**:
- 쿼리 인코딩이 로컬에서 즉시 실행 (ms급, network 없음)
- Fraction이 ~2%로 떨어짐 → **우리의 원래 목표 메트릭이 무너짐**

**해결책** (둘 중 선택):

**(A) 로컬 ANN 모델로 fraction 재정의** ✅ **권장**
```
new_fraction = (local_ann_search_time + local_embedding_time) / total_e2e
# 예: 20ms ANN + 10ms embedding = 30ms / 20s = 0.15% 정도
```
- 진정한 지연 은닉 가능성 측정
- Decode-RAG의 core benefit 입증 (speculative retrieval을 안 하면 30ms 손실, 하면 LLM 중에 감춤)

**(B) 다른 API로 network 유지** ⚠️ **트레이드오프**
- OpenAI, Jina, Voyage, Nomic 등의 임베딩 API 사용
- Network latency는 유지 → fraction 메트릭 보존
- 대신 또 다른 API 의존 (Cohere 문제 반복)

**우리의 recommend**: **(A) + (B) 병행**
- Primary: wiki_dpr + Contriever (로컬)로 진정한 아키텍처 측정
- Secondary: OpenAI/Jina API로 "network 그래프" 검증 (sensitivity analysis)
- Paper에서: "fraction 메트릭의 재정의 & 로컬 ANN의 가능성" 강조

---

---

## 최종 정리 & 다음 단계

### 임무 1 결론
- ✅ 9편 검토 완료 (25개 claim 검증)
- **직접 비교 대상**: Predictive Prefetching (1) + TeleRAG (2)
- **아이디어만 참고**: PipeRAG, FLARE, Speculative 시리즈, PASTE
- **무관**: Speculative Decoding (LLM 디코딩)

### 임무 2 결론
- ✅ 비-Cohere 대형 벡터DB 조사 완료
- **#1 권장**: `facebook/wiki_dpr` (21M passages) + DPR/Contriever 로컬 인코더
- **대체안**: `maloyan/wikipedia-22-12-all-MiniLM` (우리 원래 코퍼스)
- **⚠️ 주의**: fraction 메트릭 재정의 필요 (로컬 ANN vs. network API)

### 다음 단계 (구현 전)
1. ✅ **이 문서 repo에 저장** (`RELATED_WORK.md`)
2. [ ] **PLAN 작성** (wiki_dpr + Contriever 이식 상세 계획)
3. [ ] **사용자 승인** (plan review)
4. [ ] **구현 시작** (인덱스 빌드, eval 스크립트 수정)

---

**문서 작성일**: 2026-07-08  
**검증**: deep-research workflow, 103개 agent, 556개 tool call  
**Primary sources**: arXiv, HuggingFace, ACL/EMNLP/KDD/ICML peer-reviewed
