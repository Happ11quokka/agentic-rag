# 개발로그 — HDD 벡터DB 검색 지연 진단 및 대응

**작성일**: 2026-08-02 (2026-08-03 갱신 — 결정을 Milvus DiskANN으로 교체)
**브랜치**: `wikipedia_hdd`
**대상**: 랩 미팅 보고용
**관련 문서**: [`STATUS.md`](STATUS.md) (적재 기록), [`HNSW_HDD_MISMATCH.md`](HNSW_HDD_MISMATCH.md) (원인 상세 진단), [`../decode_rag_prefetch/PLAN.md`](../decode_rag_prefetch/PLAN.md) (prefetch 기획안)

---

## 요약

외장 HDD에 Wikipedia 47M 벡터DB 적재는 완료했으나(13시간), **검색 1회가 48분에도 끝나지 않아 실험이 불가능한 상태**다.

가설 A(Docker VM 램 부족)를 검증한 결과 **기각**됐다. 램을 8.2 GB → 27.4 GB로 3.3배 늘렸음에도 Qdrant는 2.96 GB만 사용했고 검색 시간은 그대로였다. 원인은 캐시 용량이 아니라 **디스크 랜덤 IOPS(실측 68회/초)** 이며, 더 근본적으로는 **HNSW가 랜덤 접근이 싼 매체를 전제로 설계된 자료구조**라는 데 있다.

대응은 **백엔드를 Milvus로 바꾸고 DiskANN 인덱스를 쓰는 것**이다. 종전 초안은 Qdrant에서 binary 양자화 + `rescore`로 "DiskANN과 같은 구조"를 흉내내는 안을 권장했으나, 그 안은 *디스크에 벡터를 두는 실제 시스템이 채택하는 구성*이 아니라는 문제가 남는다. 자세한 판단 근거는 [6절](#6-결정--milvus-diskann-이관)에 있다.

2026-08-03 기준 **1M 부분집합에서 검색이 완주한다.** Qdrant HNSW로는 48분에도 끝나지 않던 것이 DiskANN에서는 콜드 17.7초 / 웜 71–80 ms다(로드 25.5분 별도). 결과 품질도 정상이다.

다만 이 수치는 **"되느냐"의 답이지 "얼마나 빠르냐"의 답이 아니다.** 규모가 47배 작고, 1M은 램에 통째로 들어가며, 벤치마크가 같은 질의를 반복해 캐시를 재고 있다. 상세와 한계는 [6절 결과](#결과--1m-부분집합-검색-성공-2026-08-03)에 있다.

---

## 1. 지금까지 겪은 문제 일람

시간순. 각 항목의 상세는 링크된 절과 문서에 있다.

| # | 시점 | 문제 | 판정 |
|---|---|---|---|
| 1 | 07-31 | hf-xet 전송이 두 번 모두 ~270 MB에서 바이트 증가 0으로 스톨 | **해결** — `--disable-xet --download-timeout 30`으로 resumable HTTP 전환. 재개할 때마다 다시 붙여야 한다 |
| 2 | 07-31 | Docker Desktop이 5일째 기동 불가 | **해결** — 고아 VM(PID 83067, fd 1,200개+ 점유) kill 후 125초 만에 정상화. [STATUS.md §4.2](STATUS.md) |
| 3 | 07-31~08-01 | 다운로드 에러 9건 (timeout 1 + peer closed 8) | **해결** — resumable HTTP가 9건 전부 자동 복구, 미복구 0건 |
| 4 | 08-01 | 초반 20분간 `records this run`이 0에 머물러 정체로 오독 | **측정 함정** — 워커 4개가 각자 첫 샤드를 받는 구간. `.incomplete` 파일 합계로 속도를 재면 샤드 완료 시 음수가 된다 |
| 5 | 08-01 | **검색 1회가 48분에도 미완료** | **원인 규명 완료, 대응 진행 중** — HNSW×HDD 불일치. [4절](#4-진단--seek-횟수를-결정하는-것은-세그먼트-개수), [HNSW_HDD_MISMATCH.md](HNSW_HDD_MISMATCH.md) |
| 6 | 08-02 | 가설 A: Docker VM 램 8.2 → 27.4 GB 증설 | **기각** — Qdrant가 2.96 GB만 사용, 검색 시간 불변. [3절](#3-가설-a-검증--docker-vm-램-증설) |
| 7 | 08-02 | `hnsw_ef` 128 → 16 (8배 축소) | **효과 없음** — 세그먼트 96개를 도는 고정 비용이 지배적 |
| 8 | 08-03 | 결정 B(Qdrant binary 양자화)가 "실제로 쓰이지 않는 구성을 측정한다"는 지적 | **결정 교체** — Milvus DiskANN으로 전환. [6절](#6-결정--milvus-diskann-이관) |
| 9 | 08-03 | arm64 Knowhere에 DiskANN이 없을 가능성 (과거 x86 전용) | **해소** — 실측으로 DISKANN 빌드·로드·검색 성공 확인 |
| 10 | 08-03 | `describe_index`가 인덱스를 하나도 안 만든 상태에서도 `state=Finished`를 반환 | **해결** — `pending_index_rows == 0`으로 판정하도록 `wait_for_index` 구현 |
| 11 | 08-03 | 200k건(800 MB)은 세그먼트 봉인 임계에 못 미쳐 brute-force로 검색됨 | **해결** — 600k로 늘려 `indexed_rows=66,000` 확인. 부분집합 규모를 1M으로 정한 근거 |
| 12 | 08-03 | 단위 테스트가 `ensure_milvus`를 패치하지 않아 실제 스택을 pytest tmp 디렉터리로 재생성 | **해결** — 테스트 패치 + `ensure_milvus`가 실행 중 컨테이너의 마운트 경로를 검증하도록 수정 |
| 13 | 08-03 | 적재 중 다운로드가 6 MB/s → 87 KB/s로 70배 하락 | 아래 [6절 실측](#실측--부분집합-적재-2026-08-03) 참조 |
| 13a | 08-03 | Milvus `upsert`가 신규 컬렉션에서 레코드마다 delete tombstone 생성 → 압축이 HDD 포화 | **해결** — 신규 적재는 `insert`로 전환. `--milvus-upsert`로 종전 동작 선택 가능. 저장소 3배 감소, 샤드당 적재 55초 → 35초 |
| 13b | 08-03 | "인덱스 빌드가 HDD 대역을 독점해 다운로드가 굶는다"고 판단 | **판단 철회** — 근거로 쓴 비교가 다른 파일을 받은 것이라 오염돼 있었다. 스테이징을 SSD로 옮긴 뒤에도 같은 샤드가 느렸다. 남는 설명은 **회선 변동**(같은 시각 `/dev/null` 기준 1.7–2.9 MB/s) |
| 14 | 08-03 | `dataset`를 SSD로 심볼릭 링크하니 `ingest.py:125` 컨테인먼트 가드가 거부 | **정상 동작** — manifest 경로 탈출 방지 장치. 링크 대신 `--bundle-dir`(SSD) + `--milvus-storage-dir`(HDD) 분리로 해결 |
| 15 | 08-03 | 4번째 샤드에서 텍스트 73,931바이트 레코드(`18886_2`)가 Milvus VARCHAR 상한(65,535) 초과 → 적재 중단 | **해결** — Milvus에 더 큰 문자열 타입이 없어 자르는 수밖에 없다. `--milvus-truncate-text`로 명시적 선택(UTF-8 경계 보존, 건수 보고). 기본값은 여전히 실패. **임베딩은 영향 없고 payload만 짧아진다** |
| 16 | 08-03 | 15번으로 중단된 시점에 shard 003의 60,000행이 이미 들어가 있어, `insert` 재개 시 중복 발생 | **회피** — 볼륨과 체크포인트를 지우고 처음부터 다시 적재. `insert`는 `upsert`와 달리 멱등하지 않으므로 부분 적재 후 재개는 안전하지 않다 |
| 17 | 08-03 | 인덱스 빌드 완료 후에도 `pending_index_rows`가 총계(1,000,000)를 반환 → 대기 루프가 영원히 안 끝남 | **해결** — 10번에서 `pending==0`으로 고쳤던 판정이 이번엔 반대로 걸렸다. `state`도 `pending`도 단독으로는 못 쓴다. **`indexed_rows >= total_rows`를 2회 연속 확인**하는 방식으로 재수정 |
| 18 | 08-03 | 첫 검색 20초에 컬렉션 로드 시간이 섞여 검색 지연으로 보고됨 | **해결** — `inspect`가 벤치마크 전에 명시적으로 로드하고 로드 시간을 따로 출력. 분리 결과 **로드 25.5분 + 첫 검색 17.7초**로 드러남 |
| 19 | 08-03 | `benchmark_query`가 같은 질의를 10회 반복 → 2회차부터 페이지 캐시를 측정 | **해결 (08-04)** — 실행마다 서로 다른 질의를 쓰도록 교체. 주제가 겹치지 않는 내장 질의 20개에서 `--runs` 개를 뽑고, `--query`를 반복해 직접 줄 수도 있다. 질의 수보다 `--runs`가 크면 조용히 재사용하지 않고 오류를 낸다. `summary:`에 `cold`(1회차)를 별도 출력. **이 변경 이전 수치(웜 71–80 ms 포함)는 비교 대상이 아니다** |

| 20 | 08-04 | 기존 Qdrant 47M 컬렉션에서 `scroll`로 벡터를 뽑아 재다운로드를 대체할 수 있는가 | **기각** — 실측 930 rec/s. 9M에 2.7시간이고, 그동안 Milvus 쓰기가 **같은 HDD**를 두고 경합한다(네트워크 경로는 소스가 회선이라 경합이 없다). 새 코드도 필요하다. [아래 실측](#실측--scroll-대체-경로-판정-2026-08-04) |
| 21 | 08-04 | 회선이 6 MB/s(08-01)에서 2.1 MB/s로 하락 | 1스트림 2.21 MB/s, 4스트림 합계 2.06 MB/s — **병렬로 안 늘어나므로 회선 포화**. 90샤드 39 GB에 약 5.4시간. 13b에서 관측한 회선 변동이 계속되고 있다 |
| 22 | 08-04 | minio 컨테이너가 계속 `unhealthy` | **해결** — 헬스체크가 `curl`을 쓰는데 minio 이미지에 `curl`이 없다(`executable file not found`). 프로브가 영원히 실패하므로 적재 내내 빨간 컨테이너가 남아 진짜 고장을 가린다. 이미지에 있는 `mc ready local`로 교체 |
| 23 | 08-04 | **10M 적재 중 Milvus가 OOM으로 강제 종료**(exit 137), 샤드 51/90 중간에서 사망 | **해결** — 적재 내내 컬렉션이 **로드된 상태**였다. 로드된 컬렉션에 insert하면 새 행이 인덱싱될 때까지 query node 메모리에도 상주한다. 적재 전 `release()`를 호출하도록 수정. [아래 상세](#실측--적재-중-oom-2026-08-04) |
| 24 | 08-04 | 23번 때문에 샤드 060이 89,450/100,000만 들어간 채 체크포인트 안 됨 | **해결** — `insert`가 멱등하지 않아 재개하면 중복된다. 샤드가 선언한 id 100,000개를 delete(없는 id는 무영향)한 뒤 재적재. Milvus delete는 언로드 상태에서도 동작하며, 삭제 타임스탬프가 재삽입보다 앞서므로 새 행이 보인다(upsert와 같은 원리) |

| 25 | 08-04 | 10M 로드가 `OOM if load`로 거부 (`predict 28,998 MB > limit 28,020 MB`) | **부분 해결** — 필드 전체 mmap(`QUERYNODE_MMAP_*`)으로 예측치 하락. 그러나 다른 항이 남아 재발 |
| 26 | 08-05 | mmap 적용 후에도 601초에 다시 거부 (`memUsage 26,192 > threshold 25,218`) | **해결** — `memUsage`의 90%가 실제 메모리가 아니라 **진행 중 세그먼트 로드의 예약치**였다. `queryCoord.taskExecutionCap` 256 → 4. [아래 상세](#실측--10m-로드-거부의-정체-2026-08-05) |
| 27 | 08-05 | "601초"를 OOM 소요 시간으로 오독 | **해결** — 601초는 `queryCoord.loadTimeoutSeconds`(기본 600)의 취소 시점이다. OOM 문자열은 `FailedLoadCache`가 재생한 **직전 에러의 잔상**이었다. 36000으로 상향 |
| 28 | 08-05 | 가설: Milvus가 페이지 캐시를 사용 메모리로 세므로 캐시를 주기적으로 드롭하면 로드가 통과한다 | **기각** — `GetUsedMemoryCount()`는 `/proc/<pid>/statm`의 `RSS − Shared`, 즉 익명 RSS다. file-backed 페이지가 **구조적으로** 빠진다. 실측: 6.5 GB 캐시를 비워도 익명 RSS 변화 0 kB. [아래](#기각--페이지-캐시-드롭) |
| 29 | 08-05 | 적재/로드 중 컨테이너가 `exit=1`로 사망 (`OOMKilled=false`) | **해결** — 메모리가 아니라 **etcd 세션 상실**. 전 role이 `connection lost detected, shuting down` 후 종료. 7절이 예고한 "etcd가 HDD 위에 있다"가 현실화. etcd 볼륨만 SSD로 |

**아직 열려 있는 것**: 5번(부분집합에서 지연 재측정 중), SSD 대조군 미구축, AgentBench 미연동. [7절](#7-남은-리스크) 참조.

---

## 2. 문제 정의

적재 결과는 정상이다.

```
points 47,018,430 | indexed 47,018,430 | status green | segments 96
스토리지 209 GB @ /Volumes/agentic_rag/wikipedia/qdrant (외장 USB HDD, APFS)
```

그러나 검색이 완료되지 않는다.

| 조건 | 결과 |
|---|---|
| 기본 `ef` | 900초 타임아웃에도 미완료 |
| `hnsw_ef=16`, `limit=1` (8배 축소) | 240초 타임아웃에도 미완료 |

`ef`를 8배 줄여도 빨라지지 않았다는 점이 첫 단서였다. 그래프 탐색량이 병목이라면 대략 비례해서 줄어야 하는데 그렇지 않았다.

---

## 3. 가설 A 검증 — Docker VM 램 증설

### 설계

벡터 본체 183 GB는 어차피 캐시에 안 들어가지만, HNSW 그래프(2.8 GB, 전체의 1.3%)만이라도 상주시키고 벡터 일부를 캐싱하면 나아질 것이라는 가설.

램 증설은 **재색인이 필요 없고 되돌리기 쉬우며, 벡터를 여전히 HDD에서 읽으므로 처치(treatment)를 보존**한다. 그래서 가장 먼저 시도했다.

```jsonc
// ~/Library/Group Containers/group.com.docker/settings-store.json
{ "MemoryMiB": 28672 }   // 신규 추가. 종전에는 키 자체가 없어 기본값(전체 램의 25%) 사용
```

```
변경 전: CPUs 12  Memory  8.2 GB
변경 후: CPUs 12  Memory 27.4 GB
```

### 측정 조건

- 질의 벡터는 종전 측정과 동일한 `[0.01]*1024` (비교 가능성 유지)
- 서버측 타임아웃 `?timeout=3600`, curl `--max-time 3700` — 종전의 900초/240초 타임아웃 때문에 "완료 시간"을 못 잰 문제를 제거
- 검색 중 `docker stats`를 20초 간격 샘플링
- 컬렉션 green 확인 후 콜드 상태에서 시작

### 결과 — 기각

```
19:25:56  CPU 2.10%  MEM 2.754 GiB / 27.36 GiB   ← 검색 시작
19:33:51  CPU 0.22%  MEM 2.854 GiB / 27.36 GiB
19:39:59  CPU 0.58%  MEM 2.903 GiB / 27.36 GiB
19:53:59  CPU 0.32%  MEM 2.961 GiB / 27.36 GiB
20:14:13                                          ← 48분 17초, 미완료 상태로 수동 중단
```

**콜드 쿼리는 끝내 완료되지 않았다.** 서버 타임아웃을 3600초로 늘려둔 상태였으므로 타임아웃 때문이 아니다.

두 가지가 동시에 관측된다.

1. **CPU가 계속 0.2~0.6%** — 계산이 아니라 I/O 대기다.
2. **27.4 GB 중 2.9 GB만 사용** — 캐시에 여유가 넘치는데 채우질 못한다.

14분 동안 메모리 증가량이 **149 MiB (≈ 0.18 MB/s)** 다. 4 KB 페이지 기준 **초당 약 45회의 랜덤 읽기** — 5400rpm USB HDD의 랜덤 IOPS 한계와 일치한다.

> **결론: 병목은 캐시 용량이 아니라 랜덤 IOPS다.**
> 캐시는 *이미 읽은* 데이터를 재사용하게 해줄 뿐, *처음 읽어야 할* 데이터의 seek 횟수를 줄이지 못한다.
> 콜드 쿼리(= 실험에서 보고할 값)는 램을 아무리 늘려도 개선되지 않는다.

---

## 4. 진단 — seek 횟수를 결정하는 것은 세그먼트 개수

Qdrant는 **모든 세그먼트를 각각 탐색**한다. 세그먼트마다 독립적인 HNSW 그래프가 있으므로, 질의 1회 = HNSW 탐색 96회다.

세그먼트 1개 구성:

```
vector_storage   1.9 GB
payload_storage  291 MB
vector_index      29 MB   ← HNSW 그래프
```

96개 합산:

| 구성요소 | 크기 | 캐시 적재 가능? |
|---|---|---|
| 벡터 본체 | **182.9 GB** | ❌ |
| payload + id_tracker | ~23 GB | ❌ |
| HNSW 그래프 | 2.8 GB | ✅ (27 GB 중) |

HNSW 탐색은 방문하는 노드마다 **해당 벡터를 읽어 거리를 계산**해야 한다. 그래프가 램에 있어도 벡터 읽기는 디스크로 간다. 세그먼트당 수백~수천 노드를 방문하고 그게 96번 반복되므로, 질의 1회에 수만 회의 랜덤 읽기가 발생한다.

`iostat`로 디스크를 직접 재서 산수를 닫았다.

```
disk6 실측     : 68 IOPS (64/59/81 tps), 16.00 KB/t, ~1.09 MB/s
28분간 읽기 횟수: 68 × 1,680초 ≈ 114,000회
세그먼트당      : 114,000 ÷ 96 ≈ 1,190개 노드 방문   ← ef=128의 5~10배, 정상 범위
소요            : 114,000 ÷ 68 ≈ 28분                ← 실측과 일치
```

`ef`를 8배 줄여도 안 빨라진 것도 이걸로 설명된다 — 세그먼트 96개를 도는 고정 비용이 지배적이기 때문이다.

> **버그 가설은 별도로 검증해 배제했다.** 무한루프·미종료 여부를 포함한 상세 진단은
> [`HNSW_HDD_MISMATCH.md`](HNSW_HDD_MISMATCH.md) 참조.

---

## 5. 이 문제가 연구 질문과 직결되는 지점

본 연구의 질문은 **"작은 drafter 모델로 RAG 도중 다음 검색을 미리 예측·실행하는 것이 가능한가"** 이다. HDD는 검색을 느리게 만들어 숨길 여지를 키우기 위한 처치다.

### 5.1 "느릴수록 좋다"가 아니다

prefetch가 hop당 숨길 수 있는 최대치는 `min(retrieval, decode)` 다.

| retrieval | decode | 숨김 상한 | e2e 대비 |
|---|---|---|---|
| 900초 | 5초 | 5초 | **0.5%** |
| 1초 | 1초 | 1초 | **50%** |

검색이 decode보다 압도적으로 느리면 prefetch로 회수할 수 있는 비율은 오히려 **0에 수렴**한다. 의미 있는 구간은 `retrieval ≈ decode`, 즉 **초 단위**다.

> 따라서 현재의 15분은 "강한 처치"가 아니라 **고장**이다. 이건 성능 최적화 이슈가 아니라 실험이 성립하기 위한 전제 조건이다.

### 5.2 반면 이 구성에는 drafter를 돌릴 자리가 비어 있다

prefetch 설계의 핵심 위험은 **drafter와 본 파이프라인의 자원 경합**이다. 그런데 이 구성에서는 둘이 서로 다른 자원을 쓴다.

| 단계 | 점유 자원 | 실측 |
|---|---|---|
| 검색 (Qdrant on HDD) | **디스크 I/O** | 컨테이너 CPU 0.2~0.6%, disk6 68 IOPS |
| drafter 추론 | **MPS GPU** | — |

**검색이 디스크에서 대기하는 28분 동안 CPU도 GPU도 사실상 유휴 상태다.** drafter를 돌릴 자원이 실제로 비어 있다는 뜻이다. 검색이 연산 자원을 쓰는 구성이었다면 drafter가 본 파이프라인을 느리게 만들었겠지만, 여기서는 그 위험이 구조적으로 작다.

단, 쿼리 인코딩(로컬 BGE-M3, `encoder.py`)은 GPU를 쓰므로 경합 지점이 완전히 0은 아니다. 인코딩은 쿼리당 1회 수백 ms 수준이라 검색 대기에 비해 작지만, Stage 3에서 실측해 보고한다.

---

## 6. 결정 — Milvus DiskANN 이관

**Qdrant HNSW를 버리고 Milvus의 `DISKANN` 인덱스로 간다.**

### 왜 종전 결정(Qdrant binary 양자화)을 뒤집었나

종전 초안은 Qdrant에서 세그먼트 96→4 + `hnsw.on_disk: false` + binary 양자화(`always_ram`) + `rescore`를 한 번의 재색인에 묶는 안을 권장했다. 이 안은 **동작 원리로는 DiskANN과 같다** — 램의 압축본으로 탐색하고 원본으로 재순위한다. 백엔드 교체가 없어 비용도 쌌다.

뒤집은 이유는 성능이 아니라 **무엇을 측정하고 있는가**이다.

> HNSW는 램 상주를 전제로 설계된 자료구조다. 디스크에 벡터를 두는 실제 시스템은 DiskANN 계열을 쓴다.
> HNSW를 HDD에 올린 구성은 "느린 저장매체의 비용"이 아니라 **"잘못 고른 자료구조의 비용"** 을 재고 있다.

이 반론에 대해 "Qdrant 설정으로 DiskANN과 같은 구조를 만들었다"는 답은 충분하지 않다. 그 구성을 실제로 운용하는 시스템이 없기 때문에, 결과를 놓고 "그건 아무도 안 쓰는 설정 아니냐"는 질문이 그대로 남는다.

DiskANN으로 가면 성립하는 문장이 달라진다.

> *"디스크 기반 ANN의 정석 구성에서 저장매체를 SSD → HDD로 바꾸면
> 검색 지연이 얼마나 늘고, 그것이 decode 뒤에 얼마나 숨겨지는가"*

### 왜 Milvus인가

| | HNSW (종전) | DiskANN |
|---|---|---|
| 그래프 | 계층형, 근거리 이웃 위주 | **Vamana** — α-가지치기로 장거리 간선 확보 → 홉 수 감소 |
| 벡터 저장 | 그래프와 분리 → 노드마다 별도 읽기 | **벡터 + 이웃목록을 같은 섹터에** → 읽기 1회 |
| 탐색 중 거리계산 | 원본 벡터 필요 → 매번 디스크 | **램의 PQ 압축본** → 탐색 중 디스크 접근 없음 |
| 최종 순위 | — | 후보만 원본 벡터로 재계산 |

Qdrant는 DISKANN을 지원하지 않는다. 이 브랜치에 이미 `wikipedia/src/wikipedia/milvus.py`가 있고 Milvus는 `DISKANN`을 지원하므로 경로가 열려 있었다.

### 구성

Milvus standalone은 etcd + MinIO + milvus 세 컨테이너다. **볼륨 세 개를 모두 외장 HDD에 둔다.**

```
HDD  /Volumes/agentic_rag/wikipedia_diskann/milvus/volumes/
     ├── etcd/     메타데이터
     ├── minio/    원본 binlog (오브젝트 스토리지)
     └── milvus/   ← DiskANN 인덱스 (/var/lib/milvus)

SSD  ~/.cache/wikipedia_diskann/
     ├── dataset/  임시 parquet (적재 직후 삭제)
     ├── models/   BGE-M3 (쿼리 인코딩용)
     └── state/    적재 체크포인트
```

검색 중 디스크를 읽는 지점은 `volumes/milvus` 하나뿐이다. MinIO는 적재와 로드 시점에만 쓰이고 쿼리 경로에는 관여하지 않는다. **저장매체 차이가 정확히 ANN 인덱스 읽기 하나에만 반영**되므로, 세그먼트 96개에 그래프 탐색·링크 읽기·벡터 읽기가 뒤섞여 있던 Qdrant 구성보다 인과 설명이 깨끗하다.

**번들이 SSD에 있는 것은 처치를 약화시키지 않는다.** `dataset/`의 parquet은 적재 직후 삭제되는 임시 파일이고(manifest `retention: ephemeral`), 모델과 체크포인트는 검색 경로에 없다. 벡터DB 저장소 세 개는 전부 HDD에 있다. 다만 이 분리는 **필요성이 입증돼서 한 것이 아니다** — 경위와 정정은 아래 실측에 있다.

`queryNode.enableDisk`는 기본값이 `false`이며 이게 켜져 있지 않으면 DISKANN 인덱스를 로드할 수 없다. `QUERYNODE_ENABLEDISK=true` 환경변수로 켠다.

### 실측 — 실행 가능성 판정 (2026-08-03)

먼저 **합성 벡터로 arm64에서 DiskANN이 되는지부터** 확인했다. 4.3 GB를 내려받기 전에 판정하기 위해서다.

| 확인 항목 | 결과 |
|---|---|
| `create_collection(index_type="DISKANN")` | 수용됨 |
| 인덱스 실제 빌드 | `indexed_rows=66,000` — 빌드됨 |
| `load_collection` (enableDisk 필요) | 성공 |
| 검색 (600k건, `search_list=100`) | 콜드 3,561 ms → 웜 96 ms / 166 ms |

**arm64 Knowhere에 DiskANN이 없을 것이라는 최대 리스크가 여기서 해소됐다.**

이 과정에서 두 가지 함정을 발견했다.

1. **`describe_index`의 `state`는 완료 신호로 못 쓴다.** 인덱스가 하나도 안 만들어진 상태(`indexed_rows=0, pending=200000`)에서도 `Finished`를 반환한다. 이걸 믿고 진행하면 brute-force 스캔을 인덱스 검색으로 착각해 측정한다. → `pending_index_rows == 0`으로 판정한다.
2. **200k건(800 MB)으로는 세그먼트가 봉인되지 않는다.** Milvus는 sealed 세그먼트에만 인덱스를 만들므로, 데이터가 적으면 DiskANN이 아예 안 걸린다. 600k로 늘려서야 `indexed_rows > 0`이 됐다. → 부분집합 규모를 **10샤드(≈1M건)** 로 잡은 근거다.

### 실측 — 부분집합 적재 (2026-08-03)

첫 적재 시도에서 다운로드가 6 MB/s → **87 KB/s로 70배 떨어졌다.** 원인이 두 개였고, 둘 다 "HDD의 쓰기 대역이 유한하다"는 같은 뿌리에서 나왔다.

**원인 1 — `upsert`가 만든 delete tombstone.** Milvus의 `upsert`는 delete + insert로 구현된다. 신규 컬렉션에 200k건을 넣었을 뿐인데 Milvus 로그에 `"delete entries counts"=111056`, `77099`가 찍혔다. **덮어쓸 것이 없는 적재에서 레코드마다 삭제 마커를 만들고 그걸 다시 압축**하고 있었다.

```
200k 레코드 적재 후 디스크    : 18 GB   (예상 ~1.2 GB)
적재가 멈춘 상태에서의 증가율 : 680 MB/분   ← 순수 압축 부하
```

→ `insert`로 전환했다. 되돌릴 수 있게 `--milvus-upsert` 플래그를 남겼다(재개 시 중복 방지용).

**원인 2 — 회선 변동. (디스크 경합이라는 초기 결론은 철회한다.)**

tombstone을 없앤 뒤에도 다운로드가 270 KB/s에 머물렀고, 같은 시각 `iostat -d disk6`이 55–77 MB/s / 84–110 tps로 HDD 순차 쓰기 한계(84 MB/s) 근처였다. 여기서 **"인덱스 빌드가 디스크를 독점해 다운로드가 굶는다"고 결론지었으나, 그 판단은 틀렸다.**

근거로 삼았던 비교가 오염돼 있었다. "HF → SSD 5.3 MB/s vs HF → HDD 270 KB/s"에서 SSD 쪽 테스트는 **다른 파일(`470.parquet`)** 을 받은 것이었다. 같은 대상을 두 매체에 받아 비교한 것이 아니므로 디스크를 분리해 내지 못한다.

번들을 SSD로 옮긴 뒤 다시 재면 결론이 뒤집힌다.

| 측정 | 값 |
|---|---|
| 샤드 000, 001 → SSD | 8.7 MB/s |
| 샤드 002 → **SSD**, 이때 HDD는 유휴(2 MB/s) | **333 KB/s** ← 디스크로 설명 불가 |
| `002`/`003`/`470` → `/dev/null` (디스크 무관) | 2.26 / 1.67 / 2.91 MB/s |

`/dev/null`로 받아도 1.7–2.9 MB/s이고, 앞서 잰 5.3 MB/s와도 다르다. **회선 자체가 시점에 따라 크게 흔들린다**는 것이 남는 설명이다. 샤드 002가 세 번의 실행에서 모두 느렸던 것은 그때마다 회선이 나빴던 것으로 보이며, 파일별 고정 특성인지는 확인하지 못했다.

**따라서 스테이징을 SSD로 옮긴 것은 필요성이 입증된 조치가 아니다.** 다만 되돌릴 이유도 없다 — parquet은 적재 직후 삭제되는 임시 파일이고 검색 경로 밖이며, 인덱스 빌드가 실제로 HDD를 포화시킨다는 측정(55–77 MB/s)은 사실이므로 경합 여지를 없애 두는 편이 측정에 유리하다. 구성은 유지하되 **근거는 "예방"이지 "실측된 병목"이 아니다.**

처음엔 `dataset/`를 SSD로 심볼릭 링크했는데 `ingest.py:125`의 컨테인먼트 가드가 거부했다. manifest에 적힌 상대 경로가 번들 밖을 가리키지 못하게 하는 장치이므로 **가드가 옳고 링크가 틀렸다.** 디렉터리를 통째로 옮기는 방식으로 바꿨다.

> **남는 사실 하나**: 적재 중 HDD는 인덱스 빌드로 쓰기 대역이 포화된다(55–77 MB/s, 한계 84 MB/s). SSD 대조군에는 이 제약이 없으므로 **적재 소요 시간은 두 조건 간 공정한 비교 지표가 아니다.** 비교는 검색 지연으로 해야 한다.

### 결과 — 1M 부분집합 검색 성공 (2026-08-03)

**Qdrant HNSW로는 48분에도 끝나지 않던 검색이 끝난다.**

```
컬렉션 1,000,000행 | index=DISKANN | indexed=1,000,000 | 중복·누락 0
질의: "What causes auroras?"  search_list=100  limit=5
```

| 단계 | 값 |
|---|---|
| 컬렉션 로드 (컨테이너 재시작 직후, HDD에서) | **1,529 s (25.5분)** |
| 1번째 검색 | **17,693 ms** |
| 2번째 | 174 ms |
| 3–10번째 | **71–80 ms** |

결과 품질도 정상이다. 상위 5건이 전부 Aurora 문서이고 점수 0.70–0.73으로, 질의와 의미가 맞는다.

절단된 청크는 1,000,000건 중 **2건**(0.0002%)이다.

#### 이 수치를 그대로 쓰면 안 되는 이유 셋

1. **규모가 47배 작다.** 47M vs 1M이므로 Qdrant 구성과 직접 비교가 아니다. "되느냐"의 답이지 "얼마나 빠르냐"의 답이 아니다.
2. **1M(약 4 GB)은 Docker VM 램 27 GB에 통째로 들어간다.** 웜 71–80 ms에는 캐시 효과가 섞여 있고, 그만큼 HDD 처치가 약해진 상태다.
3. **`benchmark_query`가 같은 질의를 10회 반복했다.** 같은 질의는 그래프에서 같은 경로를 밟으므로 2회차부터는 페이지 캐시를 재는 것이지 검색을 재는 것이 아니다. 디스크 기반 인덱스에서는 이 측정 방식 자체가 부적절하다. **08-04에 실행마다 서로 다른 질의를 쓰도록 고쳤다**(19번). 위 71–80 ms는 그 이전 수치이므로 이후 측정과 비교하면 안 된다.

따라서 신뢰할 수 있는 것은 **콜드 값(로드 25.5분, 첫 검색 17.7초)** 뿐이다. 웜 값은 폐기하고 10M 적재 후 새 측정 방식으로 다시 잰다.

#### 로드 25.5분이 말해 주는 것

`load_collection`이 반환한 뒤에도 첫 검색이 17.7초 걸린다. DiskANN은 설계상 그래프를 디스크에 두므로 로드가 끝나도 질의 경로의 디스크 읽기가 남는다. **HDD에서는 이 "로드 후에도 남는 디스크 접근"이 초 단위로 드러난다.** 이것이 decode-RAG prefetch가 숨겨야 할 대상이며, 규모를 키우면 웜 상태에서도 사라지지 않을 값이다.

### 실측 — scroll 대체 경로 판정 (2026-08-04)

47M 전체 재적재를 안 하기로 하고 **10M(100샤드)** 로 규모를 정했다. 그러면 90샤드
39 GB를 더 받아야 하는데, 벡터가 이미 HDD의 Qdrant 컬렉션(208 GB)에 있으므로
`scroll`로 뽑으면 회선을 안 써도 된다. 그 값을 재봤다.

```
150,000건 scroll (with_vectors + with_payload, 페이지 1,000)
→ 161.3초, 930 rec/s, 4.0 MB/s (4,543 B/record)
동시 iostat -d disk6: 54~77 tps, 280~567 KB/t, 15~37 MB/s
```

**scroll은 순차 스캔이 아니다.** 디스크는 15~37 MB/s를 읽는데 쓸모 있는 데이터는
4.0 MB/s뿐이다(약 6배 증폭). tps가 65 근처로 붙어 있는데, 이는 앞서 잰 랜덤 68 IOPS
상한과 같다. readahead가 전송 단위를 16 KB → 400 KB로 키워 준 덕에 완전 랜덤 접근보다는
낫지만, 여전히 IOPS가 천장이다.

| 경로 | 소요 (9M 기준) | 병목 | 추가 비용 |
|---|---|---|---|
| 재다운로드 | **5.4시간** (현 회선 2.06 MB/s) | 회선 | 없음 — 기존 파이프라인 |
| Qdrant scroll | 2.7시간 (이론) | HDD IOPS | **새 코드 + 같은 HDD에서 Milvus 쓰기와 경합** |

scroll이 숫자상 빨라 보이지만 두 경로의 성격이 다르다. 네트워크 경로는 소스가 회선,
싱크가 HDD라 경합이 없다. scroll 경로는 **소스와 싱크가 같은 디스크**다 — 적재 중
인덱스 빌드가 이미 HDD 쓰기 대역을 55~77 MB/s로 포화시킨다는 측정이 있으므로(위
[6절](#실측--부분집합-적재-2026-08-03)) 2.7시간은 달성되지 않는다. 여기에 새 코드
비용까지 얹으면 이득이 남지 않는다. **채택하지 않는다.**

### 실측 — 적재 중 OOM (2026-08-04)

10M 적재를 시작해 3시간 20분, 샤드 51/90(약 5M건)에서 **Milvus가 커널에 강제 종료됐다**
(`exit=137`, `OOMKilled=true`). 적재 프로세스도 gRPC 연결이 끊기며 함께 죽었다.

죽기 39초 전 Milvus 자신의 경고가 남아 있었다.

```
08:25:53 [WARN] QuotaCenter: QueryNode memory to low water level, limit writing rate
         UsedMem=25,003,339,776 (25.0 GB)  TotalMem=29,381,267,456 (29.4 GB)  curWatermark=0.851
08:26:30 [WARN] ...  UsedMem=25,854,160,896 (25.9 GB)  curWatermark=0.880
08:26:32 OOM kill
```

원인은 컨테이너 기동 로그에 있었다.

```
05:10:13 [INFO] partition load status updated ... partitionLoadPercentage=33
05:31:53 [INFO] partition load status updated ... partitionLoadPercentage=100  loadSegmentCount=5
```

**적재 내내 컬렉션이 로드된 상태였다.** 08-03 벤치마크 때 건 로드가 Milvus에 persist돼
있었고, 오늘 스택을 올리자 QueryCoord가 자동으로 복구한 것이다.

로드된 컬렉션에 insert하면 새 행은 sealed → 인덱싱을 마칠 때까지 **query node 메모리에도
상주한다.** 산수가 맞는다.

```
새로 넣은 5M × 4,543 B/record ≈ 22.7 GB
+ 기존 1M 및 오버헤드
= 25.9 GB   ← 실측 UsedMem과 일치
```

1M 런에서 안 터진 이유는 **한 번도 로드된 적 없는 빈 컬렉션에서 시작**했기 때문이다.
규모가 아니라 로드 상태가 변수였다.

#### 처치

`MilvusVectorDB.release()`를 추가하고 `cli.py`가 적재 직전에 호출한다. 로드 상태는
재기동을 넘어 살아남으므로 **적재가 "언로드되어 있겠지"를 가정할 수 없다.**

#### 파생 피해와 복구

OOM이 샤드 060 중간에 떨어져 **89,450/100,000행만 들어간 채 체크포인트되지 않았다.**
`insert`는 멱등하지 않으므로 그대로 재개하면 그만큼 중복된다(13a에서 upsert를 버린 대가가
여기서 청구된 셈이다). 샤드가 선언한 id 100,000개를 delete한 뒤 재적재해 해결했다.

- Milvus delete는 **언로드 상태에서도 동작한다**(빈 컬렉션 프로브로 확인). 6M을 다시
  로드했다가 또 OOM 날 위험을 피할 수 있었다
- delete는 tombstone이라 `get_collection_stats`의 `row_count`는 압축 전까지 줄지 않는다.
  **최종 검증을 stats로 하면 안 되고**, 압축을 기다리거나 질의 기준으로 세야 한다
- 삭제 타임스탬프가 재삽입보다 앞서므로 새 행이 정상적으로 보인다(upsert의 동작 원리와 같다)

#### 재발 방지

메모리 감시를 붙였다 — Milvus 컨테이너 메모리가 55%를 넘거나 컨테이너가 죽으면 즉시
알린다. 워터마크 0.85에서 죽기까지 39초뿐이었으므로 0.85는 경보 지점으로 너무 늦다.

### 실측 — 10M 로드 거부의 정체 (2026-08-05)

적재는 끝났다. `rows=10,000,000 | index=DISKANN | indexed=10,000,000 | state=Finished`.
그런데 **로드가 안 된다.** 에러는 이렇게 나온다.

```
load segment failed, OOM if load, maxSegmentSize = 128.81 MB,
memUsage = 26192.19 MB, predictMemUsage = 26321.00 MB,
totalMem = 28020.16 MB thresholdFactor = 0.900000
```

읽는 그대로 받으면 "메모리가 부족하다"이다. **셋 다 틀렸다** — 메모리도, OOM도, 601초도.

#### 1. `memUsage`의 90%는 메모리가 아니다

`segment_loader.go:1446`이 계산하는 값은 이렇다.

```
memUsage = hardware.GetUsedMemoryCount() + loader.committedResource.MemorySize
```

Milvus 자신의 INFO 로그가 두 항을 다 찍어 준다.

```
[committedMemSize(MB)=23641.75] [memUsage(MB)=26192.19] [mmapFieldCount=6]
→ GetUsedMemoryCount() = 26192.19 − 23641.75 = 2,550 MB
```

**실제 프로세스 메모리는 2.5 GB였다.** 나머지 23.6 GB는 *진행 중인 세그먼트 로드가 잡아 둔
예약치*다. 세그먼트당 ~128.7 MB인데, 이 값도 DiskANN의 실제 메모리가 아니라
`IndexFactory.cpp`가 `max(knowhere_estimate, 128 MiB)`로 깔아 둔 **다운로드 버퍼 하한**이다.

예약은 로드가 *끝나야* 반납된다(`defer loader.freeRequest`). 그런데 로그를 시계열로 파 보면

```
16:14:04.463  committed=0
16:14:04.774  committed=23,641.75      ← 311 ms 만에 176개 승인
...601초 동안 감소 0회, 완료 0건
```

**176개를 한꺼번에 승인해 놓고 하나도 못 끝낸 것**이다. 느려서 쌓인 게 아니라 처음부터
쏟아부었다. 승인 게이트 `queryCoord.taskExecutionCap`의 기본값이 256이다.

→ **`QUERYCOORD_TASKEXECUTIONCAP=4`.** 적용 후 `committedMemSize`는 515 MB
(= 4 × 128.75)에 고정됐고 `memUsage`는 1.4 GB로 떨어졌다. 천장까지 25 GB가 남는다.

#### 2. 601초는 OOM이 아니라 취소다

```
[observers/collection_observer.go:178] ["load collection timeout, cancel it"] [loadTime=10m0.45s]
```

`queryCoord.loadTimeoutSeconds` 기본값이 600이다. 601초는 **이 타임아웃**이고, 화면에 뜬
OOM 문자열은 `FailedLoadCache`가 들고 있던 **직전 실패의 잔상**을 `ShowCollections`가
되읽은 것이다. 증상을 원인으로 읽고 있었다.

→ **`QUERYCOORD_LOADTIMEOUTSECONDS=36000`.**

#### 3. 그다음은 세그먼트 하나짜리 데드라인이었다

1·2를 고치자 세그먼트가 실제로 완료되기 시작했다(직전 601초 동안 1건 → 4분에 20건).
그러나 5분 뒤 로드가 통째로 풀렸다.

```
At LoadBloomFilter: failed to read files/stats_log/...: context canceled
release new segment created due to load failure
release collection due to ref count to 0
```

`context canceled`는 저장소 에러가 아니라 **태스크 데드라인**이다.
`queryCoord.segmentTaskTimeout` 기본값이 **120,000 ms(2분)**, `channelTaskTimeout`이 60,000 ms다.
이 디스크에서 세그먼트 하나 로드가 14분 26초 걸린 적이 있다. 2분 안에 끝날 리가 없다.

→ **`QUERYCOORD_SEGMENTTASKTIMEOUT=3600000`, `QUERYCOORD_CHANNELTASKTIMEOUT=600000`.**

#### 기각 — 페이지 캐시 드롭

한동안 "Milvus가 페이지 캐시를 사용 메모리로 세니 로드 중에 주기적으로 캐시를 비우면
통과한다"는 가설로 감시 스레드까지 붙였다. **기각한다.**

`GetUsedMemoryCount()`는 `/proc/<pid>/statm`을 읽어 `RSS − Shared`를 돌려준다.
`Shared`가 곧 `RssFile`이므로 이 값은 **익명 RSS**이고, file-backed 페이지는 구조적으로
빠져 있다. 실측으로도 확인된다.

| 조작 | buff/cache | 프로세스 익명 RSS |
|---|---|---|
| 6.5 GB 캐시 채움 → 드롭 | 7,109 → 595 MiB | **변화 0 kB** |

`free`의 `used` 자체가 이미 buff/cache를 제외한 값이라, 애초에 캐시를 세고 있지도 않았다.
로드 직전에 캐시를 비운 것은 **아무 효과가 없는 의식**이었다. 코드는 되돌렸고,
`drop_page_cache()`/`vm_memory_mb()`만 **콜드 측정 준비용**으로 남겼다(그건 실제로 필요하다).

> 교훈 하나: 에러 메시지의 숫자가 무엇으로 구성됐는지 보지 않고 이름만 읽으면
> (`memUsage`니까 메모리겠지) 며칠을 엉뚱한 데 쓴다. 이번엔 Milvus가 두 항을 각각
> 로그로 찍어 주고 있었는데도 그 줄을 안 봤다.

#### 부수 발견 — DiskANN은 mmap되지 않는다

`QUERYNODE_MMAP_VECTORINDEX=true`인데도 인덱스 로드 로그는 `[field=104][enable_mmap=false]`다.
Milvus 메인테이너 확인: DiskANN은 **PQ 인덱스를 항상 램에 올리고 원본 파일을 항상 로컬
디스크로 내려받으며 mmap을 적용하지 않는다.** mmap 플래그는 스칼라 필드(payload)에만
효과가 있었고, 그것만으로도 종전 `predict 28,998 MB` 실패를 없앤 값어치는 했다.

결과적으로 로드는 MinIO(HDD) → 로컬(같은 HDD)로 **75 GB 이상을 복사**하는 작업이다.
한 스핀들에서 읽기와 쓰기를 동시에 하므로, 이 단계는 원래 오래 걸리는 게 맞다.

#### 부수 발견 — etcd가 스택을 죽이고 있었다

컨테이너가 `exit=1`, `OOMKilled=false`로 죽은 적이 있다. 메모리가 아니었다.

```
[querynodev2/server.go:187] ["Query Node disconnected from etcd, process will exit"]
[sessionutil/session_util.go:922] ["connection lost detected, shuting down"]
```

etcd는 쓰기마다 fsync한다. 로드가 HDD를 포화시키면 리스 갱신이 밀리고, 세션을 잃은
전 role이 스스로 종료한다. **7절이 "규모를 키웠을 때 나타나면"이라고 적어 둔 리스크가
그대로 나타났다.** etcd 볼륨(66 MB)만 내장 SSD로 옮겼다. 인덱스는 HDD에 그대로 있으므로
처치는 보존된다.

원본은 HDD에 남겨 뒀고(`volumes/etcd`), 경로는 manifest의 `milvus.etcd_dir`에 기록한다 —
`ensure_milvus`가 compose 파일을 재생성하므로, 이 값을 안 들고 다니면 다음 실행이 조용히
etcd를 HDD로 되돌리고 스택을 재시작한다.

### 부분집합으로 먼저 가는 이유

47M 전체 재적재는 원본 parquet이 이미 삭제돼 회선 6 MB/s로 13시간이 든다. 실행 가능성이 확인되지 않은 구성에 그 시간을 쓸 이유가 없다. 10샤드(≈1M건, 4.3 GB, 다운로드 ~12분)면 sealed 세그먼트가 여러 개 생겨 DiskANN이 실제로 걸리므로, **"되는지"와 "얼마나 걸리는지"를 한 번에 판정**할 수 있다.

### 채택하지 않은 안

| 안 | 사유 |
|---|---|
| Qdrant binary 양자화 + `rescore` | 동작 원리는 DiskANN과 같지만 **실제로 운용되는 구성이 아니다.** 이 논점이 이번 결정의 핵심 |
| `ef` 낮추기 | recall이 바뀌어 대조군과 비교 불가. [4절](#4-진단--seek-횟수를-결정하는-것은-세그먼트-개수)에서 보듯 효과도 없었다 |
| 코퍼스 영구 축소 | 워킹셋이 캐시에 들어가면 HDD 처치가 무의미해진다. 지금의 1M은 **실행 확인용 임시 규모**이지 최종 구성이 아니다 |
| float16 전환 | 벡터 크기가 절반이 되어 처치 강도가 바뀐다. 재적재 13시간 |

---

## 7. 남은 리스크

- **Milvus 공식 문서는 DISKANN에 NVMe SSD를 요구한다.** USB HDD는 지원 범위 밖이다. 동작은 확인했으나 문서가 제시하는 수치대로 나오지 않을 수 있다. 이 사실은 결과 보고에 명시해야 하며, "지원 범위 밖에서 재는 것"은 이 실험이 의도한 바이기도 하다.
- **1M 부분집합은 최종 구성이 아니다.** 워킹셋이 작으면 캐시에 들어가 HDD 처치가 약해진다. 실행 가능성 판정용 규모이며, 지연 수치가 유의미하려면 결국 규모를 키워야 한다. 얼마나 키울지는 이번 측정 결과로 정한다.
- **etcd가 HDD 위에 있다.** etcd는 쓰기마다 fsync하므로 HDD에서 스택이 불안정해질 수 있다. 현재까지는 문제가 없었으나, 규모를 키웠을 때 나타나면 etcd 볼륨만 내장 SSD로 옮긴다(데이터가 작아 부담 없음).
- **MinIO가 인덱스와 별도로 원본을 또 저장한다.** 47M 전체로 가면 MinIO에 ~190 GB가 추가로 필요하다. HDD 1.6 TB 여유로 감당은 되지만 용량 계산에 넣어야 한다.
- **SSD 대조군 미구축.** 현재 treatment만 있다. baseline은 **동일하게 Milvus DISKANN**, 동일한 `search_list`, 동일한 샤드 수, 동일한 Docker VM 램으로 구축해야 비교가 성립한다.
- **AgentBench 미연동.** `AgentBench/`·`repro/` 전체에서 이 벡터DB를 참조하는 코드가 0건이다. 에이전트는 아직 Cohere+FAISS 경로를 쓴다. 검색 도구 연동이 별도 과제로 남아 있다.
- **Milvus 쪽 텍스트가 Qdrant 쪽보다 짧다.** Milvus VARCHAR 상한(65,535 B)을 넘는 청크는 잘려 저장된다. 임베딩은 원본 그대로이므로 **검색 결과 순위는 영향받지 않지만**, 두 백엔드가 돌려주는 payload는 일부 레코드에서 다르다. 에이전트에 물릴 때 문제가 되면 그때 다룬다.
- **`insert` 경로는 멱등하지 않다.** 부분 적재 후 재개하면 중복 행이 생긴다. 중간에 죽으면 해당 컬렉션을 비우고 다시 적재하거나 `--milvus-upsert`로 재개해야 한다.
- **Qdrant 47M 컬렉션(209 GB)은 그대로 남겨 뒀다.** 별도 번들(`wikipedia_diskann`)을 쓰므로 13시간짜리 적재 기록이 훼손되지 않는다. HNSW 대조가 필요하면 그대로 되살릴 수 있다.

---

## 부록 — 재현 명령

```bash
# 외장 HDD 마운트 (연결돼 있어도 자동 마운트가 안 될 수 있음)
diskutil list external          # agentic_rag 볼륨의 디바이스 확인
diskutil mount /dev/disk7s2

# Docker VM 램 확인
docker info --format 'CPUs {{.NCPU}}  Memory {{.MemTotal}}'

# 컬렉션 상태
curl -s http://localhost:6333/collections/wikipedia_2024_06_bge_m3_en_v1 | python3 -m json.tool

# 세그먼트별 구성 크기
du -sh /Volumes/agentic_rag/wikipedia/qdrant/collections/*/0/segments/*/vector_* | sort -h | tail

# 검색 (서버 기본 타임아웃 60초 → 쿼리 파라미터로 연장. body의 timeout은 무시됨)
curl -s -X POST "http://localhost:6333/collections/wikipedia_2024_06_bge_m3_en_v1/points/search?timeout=3600" \
  -H 'Content-Type: application/json' \
  -d "{\"vector\": $(python3 -c 'print([0.01]*1024)'), \"limit\": 3}"
```

### Milvus DiskANN

```bash
# 10샤드(≈1M건) 적재 — 스택 기동·flush·인덱스 빌드 대기까지 한 명령에 포함
nohup caffeinate -ims uv run wikipedia-ingest milvus \
  --bundle-dir /Volumes/agentic_rag/wikipedia_diskann \
  --max-shards 10 --disable-xet --download-timeout 30 \
  > wikipedia_diskann_ingest.log 2>&1 &

# 지연 측정
uv run wikipedia-inspect --backend milvus \
  --bundle-dir /Volumes/agentic_rag/wikipedia_diskann --runs 10

# 스택 상태
docker compose -f /Volumes/agentic_rag/wikipedia_diskann/milvus/docker-compose.yml \
  -p wikipedia-milvus ps
curl -s http://localhost:9091/healthz

# 인덱스 상태 — state가 아니라 pending_index_rows를 봐야 한다
uv run python -c "
from pymilvus import MilvusClient
print(MilvusClient(uri='http://localhost:19530').describe_index(
    'wikipedia_2024_06_bge_m3_en_v1', 'embedding'))"

# 검색 중 HDD 랜덤 읽기 활동
iostat -d disk6 1 4          # tps = IOPS, KB/t = 전송 단위
```
