# Milvus DiskANN 이관 — 설계 스펙

**작성일**: 2026-08-03
**상태**: 설계 승인됨 (구현 계획 대기)
**브랜치**: `wikipedia_hdd`
**관련 문서**: [`repro/experiments/wikipedia_hdd/DEVLOG.md`](../../../repro/experiments/wikipedia_hdd/DEVLOG.md), [`STATUS.md`](../../../repro/experiments/wikipedia_hdd/STATUS.md), [`HNSW_HDD_MISMATCH.md`](../../../repro/experiments/wikipedia_hdd/HNSW_HDD_MISMATCH.md)

---

## 1. 배경

외장 USB HDD에 Wikipedia 47M 벡터(209 GB)를 Qdrant로 적재하는 데는 성공했으나(13시간),
**검색 1회가 48분에도 완료되지 않아** 실험이 진행 불가 상태다.

진단 결과 원인은 버그가 아니라 자료구조와 매체의 불일치였다.

```
HNSW가 요구하는 것 : 랜덤 4 KB 접근 수만 회를 밀리초 안에
HDD가 제공하는 것   : 랜덤 68 IOPS, 건당 16 KB 강제 (실측)
```

Docker VM 램을 8.2 → 27.4 GB로 늘리는 가설 A는 **기각**됐다. Qdrant는 2.96 GB만 사용했고
검색 시간은 변하지 않았다. 단일 쿼리 안에서 HNSW는 같은 노드를 두 번 방문하지 않으므로
캐시할 가치가 있는 데이터가 없었기 때문이다.

이전 문서(`HNSW_HDD_MISMATCH.md` §5-B)는 Qdrant에서 binary 양자화 + `rescore`로
DiskANN과 **같은 구조**를 만드는 안을 권장했다. 그러나 같은 문서 §5-C가 지적한
실험 타당성 논점이 더 무겁다고 판단해 이 스펙에서 결정을 뒤집는다.

> HNSW는 램 상주를 전제로 설계됐다. 디스크에 벡터를 두는 실제 시스템은 DiskANN 계열을 쓴다.
> 현 구성은 "느린 저장매체"가 아니라 "잘못 고른 자료구조"를 측정하고 있다.

DiskANN으로 가면 성립하는 문장이 달라진다.

> *"디스크 기반 ANN의 정석 구성에서 저장매체를 SSD → HDD로 바꾸면
> 검색 지연이 얼마나 늘고, 그것이 decode 뒤에 얼마나 숨겨지는가"*

---

## 2. 목표

**이 스펙의 목표는 성능이 아니라 실행 가능성 판정이다.**

Milvus DISKANN 인덱스를 이 하드웨어(Apple M3 Pro / arm64 / 외장 USB HDD)에서
**끝까지 한 번 돌려보고**, 검색이 완료되어 말이 되는 결과가 나오는지 확인한다.

성공 기준:

1. Milvus 스택이 HDD 위에서 기동한다
2. DISKANN 인덱스가 실제로 빌드된다 (`describe_index` → `index_type=DISKANN`, `state=Finished`)
3. 약 1M건에 대해 검색이 완료되고 top-5 제목이 질의와 관련 있다
4. `wikipedia-inspect`로 지연을 10회 반복 측정할 수 있다

**범위 밖**:

- 47M 전체 재적재 (부분집합으로 실행 가능성부터 판정)
- SSD 대조군 구축 (다음 단계)
- recall 정량 비교 (다음 단계)
- 지연 수치의 좋고 나쁨에 대한 판단 — **느린 것이 처치**이므로 이 단계에서 최적화하지 않는다
- AgentBench 연동

---

## 3. 제약

| 항목 | 값 | 출처 |
|---|---|---|
| CPU | Apple M3 Pro, **arm64** | `uname -m` |
| 맥 전체 램 | 36 GB | |
| Docker VM | CPUs 12, Memory 27.4 GB | 가설 A 검증 때 설정 |
| 내장 SSD 여유 | 56 GB | `df -h` |
| 외장 HDD | `/Volumes/agentic_rag`, APFS(Case-sensitive), 1.6 TB 여유 | `df -h` |
| HDD 순차 쓰기 | 84 MB/s | `dd` 1 GB |
| HDD 랜덤 읽기 | 68 IOPS, 16 KB/t | `iostat -d disk6` |
| 회선 | 6 MB/s (포화) | 적재 실측 |
| 원본 parquet | **잔여 0개** — 재다운로드 필요 | 스토리지 상한 정상 동작 |
| Milvus arm64 이미지 | `milvusdb/milvus:v2.5.27` 존재 | Docker Hub 태그 조회 |

---

## 4. 아키텍처

### 4.1 스택 구성

Milvus standalone은 세 컨테이너다. 사용자 결정에 따라 **볼륨 세 개 모두 HDD**에 둔다.

```
/Volumes/agentic_rag/wikipedia_diskann/
├── dataset/            # parquet 샤드 (적재 후 삭제됨)
├── models -> /Volumes/agentic_rag/wikipedia/models   # 심볼릭 링크
├── state/              # 적재 체크포인트
├── manifest.json
└── milvus/
    ├── docker-compose.yml
    ├── milvus.yaml     # queryNode.enableDisk: true
    └── volumes/
        ├── etcd/       # 메타데이터
        ├── minio/      # 원본 binlog (오브젝트 스토리지)
        └── milvus/     # ← DiskANN 인덱스가 사는 곳 (/var/lib/milvus)
```

검색 중 디스크를 읽는 지점은 **`volumes/milvus` 하나뿐**이다. MinIO는 적재와 로드 시점에만
쓰이고, 쿼리 경로에는 관여하지 않는다. 저장매체 차이가 정확히 ANN 인덱스 읽기 하나에만
반영되므로 인과 설명이 Qdrant 구성보다 깨끗하다.

### 4.2 왜 번들 디렉터리를 분리하는가

`download.py:246-270`의 `prepare_bundle`은 manifest를 **매 실행마다 새로 쓴다**.
`--max-shards 10`으로 실행하면:

```jsonc
{ "status": "incomplete", "partial": true,
  "dataset": { "shards": [ /* 10개만 */ ], "selected_shard_count": 10 } }
```

기존 47M 번들(`/Volumes/agentic_rag/wikipedia`)에 그대로 쓰면 13시간짜리 적재 기록이
덮여 사라진다. `qdrant` 섹션만 `previous`에서 보존되고 나머지는 전부 재작성된다.

→ 별도 번들 `/Volumes/agentic_rag/wikipedia_diskann`을 쓴다.
2.1 GB BGE-M3 모델은 기존 번들에서 `models` 디렉터리를 심볼릭 링크해 재다운로드를 피한다.
`model_is_downloaded()`는 파일 존재만 확인하므로 링크로 충족된다.

### 4.3 코드 변경

#### `milvus_runtime.py` (신규)

`qdrant_runtime.py`와 대칭 구조. 공개 함수는 `ensure_milvus()` 하나.

```python
def ensure_milvus(
    uri: str,
    *,
    storage_dir: Path | None,
    project: str = "wikipedia-milvus",
    image: str = DEFAULT_MILVUS_IMAGE,
) -> str:      # "started" | "already running"
```

책임:

1. `storage_dir`의 파일시스템이 Qdrant와 같은 비호환 목록(exFAT/NTFS/NFS…)에 없는지 확인
2. `docker-compose.yml`과 `milvus.yaml`을 `storage_dir`에 렌더링 (내용이 같으면 다시 안 씀)
3. Docker 데몬 확인 — 없으면 macOS에서 Docker Desktop 기동 (`qdrant_runtime._ensure_docker` 재사용)
4. `docker compose -p <project> up -d`
5. `http://localhost:9091/healthz`가 `OK`를 반환할 때까지 대기 (기본 300초 — 3개 컨테이너 기동은 Qdrant보다 오래 걸린다)

`milvus.yaml`의 핵심 한 줄:

```yaml
queryNode:
  enableDisk: true   # DISKANN 필수. 기본값 false면 인덱스 로드가 거부된다
```

#### `milvus.py` (수정)

| 추가 | 이유 |
|---|---|
| `flush()` | Milvus는 **sealed 세그먼트에만 인덱스를 빌드**한다. 적재 후 flush하지 않으면 growing 세그먼트로 남아 brute-force로 검색된다 — DiskANN이 아예 안 걸린다 |
| `wait_for_index(timeout)` | `describe_index`를 폴링해 `state == "Finished"`까지 대기. HDD 위 빌드는 오래 걸리므로 진행 상황을 출력한다 |
| `index_state()` | 진단용. `index_type`, `state`, `indexed_rows` 반환 |
| `load_timeout` 분리 | DISKANN 로드는 인덱스를 디스크에서 읽어야 해 `timeout`(60초) 안에 못 끝난다. `MilvusConfig.load_timeout` 신설 (기본 1800초) |

`MilvusConfig`는 이미 `index_type` / `index_params` / `search_params`를 갖고 있어 그대로 쓴다.
`search()`가 만드는 `{"metric_type": ..., "params": {...}}` 형태도 DISKANN의 `search_list`에 맞다.

#### `cli.py` (수정)

milvus 분기를 qdrant 분기와 같은 수준으로 올린다.

| 플래그 | 기본값 | 비고 |
|---|---|---|
| `--index-type` | `DISKANN` | 종전 `AUTOINDEX`에서 변경 |
| `--search-list` | `100` | DISKANN 검색 후보 풀 크기 |
| `--milvus-storage-dir` | `<bundle>/milvus` | |

동작:

1. `ensure_milvus()` 호출 후 manifest에 `milvus` 섹션 저장 (qdrant 섹션과 같은 형태)
2. milvus 기본 `batch_size`를 **1000**으로 (256이면 1M건에 왕복 3,900회)
3. 적재 완료 후 `flush()` → `wait_for_index()` → 인덱스 상태 출력

#### `inspect.py` (수정)

현재 Qdrant 하드코딩이라 DiskANN 지연을 잴 수 없다. `--backend {qdrant,milvus}`를 추가하고
컬렉션 상태 출력·`ensure_*` 호출·config 생성을 백엔드별로 분기한다.
`benchmark_query()`와 `Encoder`는 백엔드 무관이라 그대로 쓴다.
`benchmark_query`의 오류 문구 `"Qdrant returned no chunks"`는 백엔드 중립으로 바꾼다.

---

## 5. 검증 순서

**싼 것부터 배치해 실패를 앞으로 당긴다.**

| 단계 | 내용 | 소요 | 실패 시 |
|---|---|---|---|
| **0** | 스택 기동 → **빈 컬렉션에 DISKANN 인덱스만 생성** | ~10분 | arm64 Knowhere에 DiskANN이 없으면 여기서 즉시 드러난다. 4.3 GB 받기 전에 판정 |
| **1** | 10샤드(≈1M건) 다운로드 + 적재 | 다운로드 ~12분 + 적재 | |
| **2** | `flush()` → 인덱스 빌드 대기 → `describe_index` 확인 | 수분~ | sealed 세그먼트가 안 생기면 `dataCoord.segment.maxSize` 축소 |
| **3** | `load_collection` → 검색 1회, top-5 제목 확인 | | |
| **4** | `wikipedia-inspect --backend milvus`로 10회 지연 측정 | | |

0단계를 독립 단계로 둔 이유는 §6의 첫 번째 리스크 때문이다.

### 규모 근거

10샤드 ≈ 998,270건 (47,018,430 ÷ 471 × 10). float32 1024차원이면 원본 4.1 GB.
Milvus 기본 세그먼트 상한은 1024 MB ≈ 26만 벡터이므로 sealed 세그먼트가 **약 4개** 생긴다.
1샤드(10만건)로는 세그먼트 하나도 못 채워 growing 상태로 남고, 그러면 DiskANN이
실제로 걸리지 않아 "실행 확인"이 성립하지 않는다. 10샤드가 목적을 만족하는 최소 규모다.

---

## 6. 리스크

| # | 리스크 | 판정 시점 | 대응 |
|---|---|---|---|
| 1 | **arm64 Knowhere에 DiskANN이 없을 수 있다.** 과거 DiskANN 구현은 x86 SIMD 전용이었다 | 0단계 | 막히면 (a) x86 이미지를 Rosetta/QEMU로 — 매우 느림 (b) Qdrant binary quantization 안으로 복귀 |
| 2 | **etcd가 HDD fsync에 걸려 스택이 불안정해질 수 있다.** etcd는 쓰기마다 fsync한다 | 0~1단계 | 사용자가 "전부 HDD"를 선택했으므로 그대로 진행하고, 실제로 걸리면 etcd 볼륨만 SSD로 옮긴다 (etcd 데이터는 작아 SSD 56 GB에 무리 없음) |
| 3 | **Milvus 공식 문서는 DISKANN에 NVMe SSD를 요구한다.** HDD는 지원 범위 밖 | 3~4단계 | 동작하면 그대로 진행. 문서 수치와 다른 것은 오히려 이 실험이 측정하려는 대상이다. 다만 "지원 범위 밖"임을 결과 보고에 명시한다 |
| 4 | **`upsert`가 대량 적재에 느리다.** Milvus upsert는 delete + insert다 | 1단계 | 1M건에서 병목이면 신규 적재 경로를 `insert`로 전환 |
| 5 | **VARCHAR 길이 상한 초과 시 적재 중단.** `text_max_bytes=65535` 초과 레코드가 있으면 예외 | 1단계 | 실제로 나면 상한을 올리거나 해당 레코드를 기록하고 건너뛴다 |
| 6 | 47M 전체로 갈 때 MinIO가 ~190 GB를 더 먹는다 (인덱스와 별도) | 이 스펙 범위 밖 | HDD 1.6 TB 여유로 감당 가능. 계산은 다음 단계에서 |

---

## 7. 실험 타당성에 대한 메모

이 변경은 처치(HDD)를 약화시키지 않는다. 벡터는 여전히 HDD에 있고 여전히 디스크에서 읽힌다.
바뀌는 것은 **"어떤 자료구조가 그 디스크를 읽는가"** 뿐이다.

- HNSW: 탐색 중 방문 노드마다 원본 벡터를 디스크에서 읽음 → 쿼리당 수만 회 랜덤 읽기
- DiskANN: 탐색은 램의 PQ 압축본으로, 디스크는 후보 재순위 때만 → 쿼리당 수백 회

둘 다 "HDD에서 읽는" 구성이지만, 후자만이 실제 시스템이 채택하는 구성이다.
따라서 SSD ↔ HDD 비교가 "저장매체 효과"를 재는 것이 되고,
"부적합한 자료구조 선택의 효과"를 재는 것이 아니게 된다.

**대조군 제약은 유지된다**: SSD baseline도 동일하게 Milvus DISKANN, 동일한 `search_list`,
동일한 샤드 수, 동일한 Docker VM 램으로 구축해야 비교가 성립한다.

---

## 8. 산출물

1. `wikipedia/src/wikipedia/milvus_runtime.py` (신규)
2. `wikipedia/src/wikipedia/milvus.py`, `cli.py`, `inspect.py` (수정)
3. `wikipedia/tests/test_milvus_runtime.py` (신규) — compose/config 렌더링과 상태 판정 단위 테스트
4. `repro/experiments/wikipedia_hdd/DEVLOG.md` (갱신) — 지금까지 겪은 문제 일람 + 결정 교체 + 리스크 추가
