# wikipedia_hdd — 외장 HDD Wikipedia 벡터DB 적재 (2026-07-31 ~ 08-01)

브랜치 `wikipedia_hdd`. **적재는 완주했고, 검색 지연이 미해결 블로커입니다.**

---

## 1. 목적

decode-RAG 연구에서 **저장매체를 독립변수로** 두기 위한 처치(treatment) 구축.
내장 SSD = baseline, 외장 USB HDD = treatment. 검색이 느려질수록 e2e 중
retrieval 대기 비중이 커지고, 그게 decode-RAG prefetch가 숨길 수 있는 상한이 된다.

> **HDD가 느린 건 제약이 아니라 실험 그 자체.** 캐싱·float16·SSD 복귀 등으로
> "성능 개선"하면 측정 대상이 사라진다. 자세한 배경은 메모리
> `wikipedia-hdd-slowdown-intent` 참조.

**공정성 제약**: SSD 대조군은 여기 쓴 모든 설정(float32, on_disk, 세그먼트 구성,
batch size, Docker VM 램)을 **동일하게** 맞춰야 한다.

---

## 2. 최종 상태 — 적재 완료 ✅

```
Ingested 47,018,430 records
points 47,018,430 | indexed 47,018,430 | status green | optimizer ok | segments 96
manifest: status=complete, partial=false, 471/471 shards, float16=false
원본 parquet 잔여 0개 (스토리지 상한 정상 동작)
소요 12시간 58분 (778분) — 2026-08-01 00:12경 종료
```

| 항목 | 값 |
|---|---|
| 번들 | `/Volumes/agentic_rag/wikipedia` |
| 디스크 | WD My Passport 2628, USB, APFS(Case-sensitive), 1.8 TB |
| 데이터셋 | `Upstash/wikipedia-2024-06-bge-m3` @ `ce4e0ea4…`, en, 471샤드 |
| 모델 | `BAAI/bge-m3` @ `babcf60c…` (2.1 GB, `models/bge-m3`) |
| 컬렉션 | `wikipedia_2024_06_bge_m3_en_v1` (Qdrant, Docker `wikipedia-qdrant`) |
| Qdrant 스토리지 | `/Volumes/agentic_rag/wikipedia/qdrant` (바인드 마운트) |

### 실행 명령 (재개도 동일)

```bash
cd /Users/imdonghyeon/agentic_rag
nohup caffeinate -ims uv run wikipedia-ingest qdrant \
  --bundle-dir /Volumes/agentic_rag/wikipedia \
  --disable-xet --download-timeout 30 \
  > wikipedia_hdd_ingest.log 2>&1 &
```

`--float16` **미사용**(float32), `--max-workers` 기본 4, `--high-performance` 미사용.
로그 `wikipedia_hdd_ingest.log`, PID `wikipedia_hdd_ingest.pid` (둘 다 gitignore).

---

## 3. 실측 환경값

| 경로 | 실측 |
|---|---|
| 회선 (HF, 1스트림) | 6.4 MB/s |
| 회선 (4스트림 병렬) | 6.0 MB/s ← **병렬로 안 늘어남 = 회선 포화 (~50 Mbps)** |
| HDD 순차 쓰기 (dd 1GB) | 84 MB/s |
| HDD 4-way 동시 쓰기 | 73 MB/s |
| 실제 적재 처리량 | ~1,000–1,120 rec/s ≈ 4.9 MB/s |
| Docker VM | CPUs 12, **Memory 8.2 GB** (맥 전체 36 GB) |

**적재는 네트워크 병목이었다.** HDD도 Qdrant도 여유였고(적재 중 Qdrant CPU 0.3~30%,
최적화 구간에만 840%), 소요 13시간은 회선이 정한 값이다.

---

## 4. 도중에 해결한 문제 (재현 시 주의)

### 4.1 hf-xet 스톨 → `--disable-xet` 필수
xet 전송이 두 번 모두 ~270 MB에서 **바이트 증가 0**으로 멈춤. README에 적힌 대로
`--disable-xet --download-timeout 30`(resumable HTTP)으로 전환하니 즉시 정상화.
**재개할 때도 이 플래그를 반드시 다시 붙여야 한다.**

### 4.2 Docker Desktop 5일째 고장 → 고아 VM 제거
증상이 3단으로 얽혀 있었다.

```
com.apple.Virtualization.VirtualMachine (PID 83067, PPID=1, 4일 22시간 경과)
  ← HILIPS_SAM2 작업하던 옛 Docker VM이 고아로 잔존, fd 1,200개+ 점유
  → com.docker.backend: accept: too many open files
  → daemon이 "starting"에서 107시간 정지
  → 새 Docker 기동 시 Rosetta 설치 실패 (VZErrorDomain Code=1) → 엔진 기동 불가
```

Docker Desktop 완전 종료 후 **고아 VM PID를 kill**하고 재기동하니 125초 만에 정상화
(server 29.1.5). 다른 프로젝트 컨테이너들은 restart policy로 자동 복귀, 유실 없음.

### 4.3 다운로드 에러 9건 — 전부 자동 복구
`read operation timed out` 1건 + `peer closed connection` 8건.
`Error 9건 = Trying to resume 9건`으로 정확히 대응, 미복구 0건.
324~327 샤드에서 워커 4개가 동시에 끊긴 구간이 있었는데 회선 순간 단절로 보이며
resumable HTTP가 이어받아 완주. 이 구간만 속도가 1,016 → 728 rec/s로 떨어졌다가
이후 1,120으로 완전 회복.

### 4.4 측정 함정 두 가지
- **초반 ~20분은 속도 측정 불가.** 워커 4개가 각자 첫 샤드를 받는 동안 upsert가
  하나도 안 끝나서 `records this run`이 0에 머문다. 이걸 정체로 오독하기 쉽다
  (실제로 한 번 "3일 걸림"으로 잘못 추정했다가 정정).
- **`.incomplete` 파일 합계로 속도를 재면 안 된다.** 샤드가 완료되면 해당 파일이
  사라져 합계가 음수로 나온다. `records this run`을 써야 한다.

---

## 5. 미해결 블로커 — 검색 15분 초과 🔴

```
기본 ef        : 900초 타임아웃에도 미완료
hnsw_ef=16, limit=1 : 240초 타임아웃에도 미완료
검색 중 Qdrant CPU : 0.37%   ← 계산이 아니라 디스크 대기
```

### 원인 분석

세그먼트 1개 내부 구성:
```
vector_storage  1.9 G
payload_storage 291 M
vector_index     29 M      ← HNSW 그래프
```

96개 합산:

| 구성요소 | 크기 | 비고 |
|---|---|---|
| 벡터 본체 | **182.9 GB** | 어떤 캐시에도 안 들어감 |
| payload + id_tracker | ~23 GB | |
| **HNSW 그래프** | **2.8 GB** | 전체의 1.3% — 램에 충분히 들어감 |
| **Docker VM 램** | **8.2 GB** | |

**핵심**: 진짜 비용은 그래프 탐색이 아니라, 탐색 중 방문하는 노드마다 **벡터를 읽어
거리를 계산**하는 부분이다. 세그먼트 96개를 전부 훑으므로 질의 1회에 수만 번의
랜덤 읽기가 발생하고, 183 GB 벡터는 8.2 GB 캐시에 들어갈 수 없다.
`ef`를 128→16으로 8배 줄여도 안 빨라진 것이 이 해석을 뒷받침한다
(그래프 탐색량 문제였다면 비례해서 빨라졌어야 함).

### 컬렉션 설정 (현재)
```
on_disk: True,  hnsw: {m:16, ef_construct:100, on_disk:True, full_scan_threshold:10000}
optimizers: {indexing_threshold:10000, default_segment_number:0(auto→96)}
```

---

## 6. 다음 세션 선택지

> **⚠️ 이 절은 2026-08-03에 폐기됐다.** 아래 표는 A~E 중에서 고르던 시점의 기록이다.
> 실제 결정은 **Milvus DiskANN 이관**이며 A(램 증설)는 검증 후 기각됐다.
> 현재 방침은 [`DEVLOG.md` 6절](DEVLOG.md)을 보라.

아래는 당시 검토 내용이다. 아직 **아무것도 적용하지 않았다.** 사용자 결정 대기 중.

| 안 | 내용 | 장점 | 비용/위험 |
|---|---|---|---|
| **A. Docker 램 8→28 GB** | Docker Desktop 설정 변경 후 재측정 | 무료·즉시·가역, 재색인 불필요. 그래프 2.8 GB 상주 + 벡터 25 GB치 캐싱. **벡터 183 GB는 여전히 HDD에서 읽으므로 처치 유지** | 캐시 적중률에 달려 효과 불확실. 맥 전체 36 GB 중 28 GB를 VM에 주는 부담 |
| **B. 세그먼트 96→4~8 병합** | `optimizers_config.default_segment_number` 조정 | 질의당 작업량 12~24배 감소. 검색 의미 불변 → 실험 타당성 보존 | 209 GB를 HDD 위에서 재작성 → 수 시간 |
| **C. A+B 동시** | | 가장 확실 | 램 증설 단독 효과를 따로 못 잼 |
| **D. 코퍼스 축소** | 부분집합 별도 컬렉션 | 워킹셋이 캐시에 들어감 | 13시간 적재분 상당수 미사용 |
| **E. 양자화(scalar/binary)** | | 가장 큰 속도 이득 | **검색 정확도가 변해 실험 타당성 훼손** — 권장 안 함 |

**권장 순서: A 먼저 재측정 → 부족하면 B.**
A는 되돌리기 쉽고 재색인이 없어 실패해도 잃는 게 없다.
어떤 안을 택하든 **SSD baseline에 동일 설정을 적용**해야 비교가 성립한다.

---

## 7. 상태 확인 명령

```bash
# 컬렉션 상태
curl -s http://localhost:6333/collections/wikipedia_2024_06_bge_m3_en_v1 | python3 -m json.tool

# 세그먼트별 구성 크기
du -sh /Volumes/agentic_rag/wikipedia/qdrant/collections/*/0/segments/*/vector_* | sort -h | tail

# Docker VM 자원
docker info --format 'CPUs {{.NCPU}}  Memory {{.MemTotal}}'

# 검색 (서버 기본 타임아웃 60초 → 쿼리 파라미터로 연장, body의 timeout은 무시됨)
curl -s -X POST "http://localhost:6333/collections/wikipedia_2024_06_bge_m3_en_v1/points/search?timeout=900" \
  -H 'Content-Type: application/json' -d "{\"vector\": $(python3 -c 'print([0.01]*1024)'), \"limit\": 3}"
```
