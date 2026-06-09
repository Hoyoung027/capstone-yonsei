# LLM Decode Attention 최적화 연구보고서

**FlashInfer Split-K Scheduler Heuristic 개선을 중심으로**

---

## 목차

1. [연구 개요](#1-연구-개요)
2. [배경 및 동기](#2-배경-및-동기)
3. [실험 환경](#3-실험-환경)
4. [실험 1: Prefill KV Tile 크기 탐색](#4-실험-1-prefill-kv-tile-크기-탐색)
5. [실험 2: Decode Cuda-Core KV Tile 탐색](#5-실험-2-decode-cuda-core-kv-tile-탐색)
6. [실험 3: Tensor-Core Decode × Split-K 전체 탐색](#6-실험-3-tensor-core-decode--split-k-전체-탐색)
7. [실험 4: Split-K Heuristic 시뮬레이션 및 실제 패치 검증](#7-실험-4-split-k-heuristic-시뮬레이션-및-실제-패치-검증)
8. [핵심 코드 설명](#8-핵심-코드-설명)
9. [결과 요약](#9-결과-요약)
10. [결론 및 향후 과제](#10-결론-및-향후-과제)

---

## 1. 연구 개요

본 연구는 LLM inference 서빙에서 널리 사용되는 attention 라이브러리 **FlashInfer**의 decode 경로를 대상으로, kernel 파라미터 및 split-K scheduler 설정이 latency에 미치는 영향을 실측하고, 더 나은 split-K 선택 heuristic을 제안·검증한다.

FlashInfer의 auto scheduler는 GPU 점유율(occupancy)을 기준으로 split-K chunk 수를 결정한다. 본 연구는 이 정책이 작은 batch size 환경에서 **over-splitting**을 유발하고, 이로 인해 불필요한 partial attention merge overhead가 발생할 수 있음을 실험으로 보인다. 제안하는 heuristic은 기존 정책의 점유율 지향 선택을 유지하면서, 최소 chunk work 보장 조건을 추가하는 간단한 guard 방식이다.

**핵심 기여:**

- FlashInfer tensor-core decode 경로에서 KV tile 크기(`NUM_MMA_KV`)와 split-K chunk size를 교차 탐색하는 실험 프레임워크 설계 및 구현
- Split-K oracle 분석을 통해 FlashInfer default scheduler의 over-splitting 구간 식별
- 최소 chunk work 조건 기반 split-K heuristic 제안 및 FlashInfer scheduler 직접 패치를 통한 실측 검증
- LLaMA-3 8B 조건에서 batch size 2, 4에서 geomean latency **약 4.9%** 개선 확인

---

## 2. 배경 및 동기

### 2.1 LLM 추론에서 Decode의 특성

LLM 추론은 크게 두 단계로 나뉜다.

- **Prefill**: 입력 토큰 전체를 한 번에 처리. query 길이가 길어 compute-bound 특성을 가진다.
- **Decode**: 토큰을 하나씩 생성. query 길이는 1(또는 batch 내 각 요청당 1)이고, 긴 KV cache를 읽어야 하므로 memory-bound 특성이 강하다.

Decode에서는 연산량보다 KV cache 메모리 접근 방식, 병렬화 전략, reduction overhead가 latency를 결정하는 주요 요인이 된다.

### 2.2 FlashInfer Tensor-Core Decode의 구조

FlashInfer의 `BatchDecodeWithPagedKVCacheWrapper`는 `use_tensor_cores=True` 옵션을 사용할 경우, 내부적으로 일반 decode 커널이 아니라 **FA2 batch prefill 커널**을 재사용한다. 이 경로는 prefill 커널의 tile 파라미터(`NUM_MMA_KV`, `CTA_TILE_KV`)와 split-K scheduler를 그대로 사용한다.

```
BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)
  └─> get_batch_prefill_module(...)
        └─> FA2 batch prefill kernel
              ├─ NUM_MMA_KV  (KV 방향 MMA tile 크기)
              └─ split-K scheduler  (scheduler.cuh)
```

### 2.3 Split-K란 무엇인가

Split-K는 긴 KV cache를 여러 chunk로 나누어 병렬 처리한 뒤 partial attention 결과를 merge하는 방식이다. chunk가 많을수록 병렬 실행 block 수가 늘어나 GPU occupancy는 높아지지만, merge 단계의 reduction overhead도 함께 증가한다.

```
kv_len = 8192,  fixed_split_size = 512
num_chunks = ceil(8192 / 512) = 16
```

FlashInfer의 auto scheduler는 SM당 동시 실행 가능한 block 수(`num_blocks_per_sm = 2`)와 KV head 수를 기반으로 이진탐색으로 chunk size를 결정한다.

```
목표 분할 작업 수 = floor(2 × SM 수 / KV head 수)
                 = floor(2 × 82 / 8)  [RTX 3090, llama3_8b]
                 = 20

→ 이진탐색으로 총 분할 작업 수 ≤ 20이 되는 가장 작은 chunk size 탐색
→ BS=2, KV=8192에서 요청당 10 chunks 선택 (총 20 분할 작업)
```

이 방식은 GPU 활용률을 최대화하지만, 짧은 KV length나 큰 batch 환경에서는 불필요하게 많은 chunk를 만들어 merge overhead가 지배적이 될 수 있다.

---

## 3. 실험 환경

### 하드웨어 및 소프트웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (82 SMs, 24GB) |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| FlashAttention | 2.8.3 |
| FlashInfer | source build |
| CUDA | 12.1 |

### 실험 대상 모델

| 모델 | num_qo_heads | num_kv_heads | head_dim | GQA group size |
|---|---|---|---|---|
| llama3_8b | 32 | 8 | 128 | 4 |
| llama3_70b | 64 | 8 | 128 | 8 |
| qwen2.5_72b | 64 | 8 | 128 | 8 |
| gemma2_9b | 16 | 8 | 256 | 2 |
| gemma2_27b | 32 | 16 | 128 | 2 |

### 공통 측정 설정

```
batch_size   = 8 (실험 3·4는 1, 2, 4, 8, 16 sweep)
page_size    = 16
kv_len       = 128, 256, ..., 8192  (128 간격, 총 64 포인트)
warmup       = 100회
repeat       = 100회
dtype        = float16
backend      = fa2
```

---

## 4. 실험 1: Prefill KV Tile 크기 탐색

### 4.1 목적

FlashInfer prefill 경로에서 KV 방향 MMA tile 크기(`NUM_MMA_KV`)가 latency에 미치는 영향을 측정한다. FlashInfer의 auto 선택이 항상 최적인지 확인한다.

### 4.2 조작 변수

```
NUM_MMA_KV = auto, 1, 2, 4, 8
```

`NUM_MMA_KV`는 FA2 batch prefill 커널에서 KV 방향으로 한 번에 처리하는 MMA tile 수를 결정하는 template 파라미터다. 값이 클수록 `CTA_TILE_KV`가 커지고, 레지스터·shared memory 사용량이 증가한다.

### 4.3 패치 방법

FlashInfer `prefill.cuh`의 `DISPATCH_NUM_MMA_KV` 매크로를 직접 수정해 특정 값으로 고정한다.

```python
# patch_prefill.py 핵심 로직
ORIGINAL = "#define DISPATCH_NUM_MMA_KV(num_mma_kv, ...) \\"
FORCED   = f"#define DISPATCH_NUM_MMA_KV(num_mma_kv, ...) \\\n  __VA_ARGS__(/*NUM_MMA_KV=*/{value})"
```

패치는 실험 전 적용, 실험 후 자동 복원된다 (`run_tile_kv.sh`의 `trap` 처리).

### 4.4 실험 흐름

```
baseline_before  → (FlashInfer auto)
forced_mma1      → NUM_MMA_KV=1 강제
forced_mma2      → NUM_MMA_KV=2 강제
forced_mma4      → NUM_MMA_KV=4 강제
forced_mma8      → NUM_MMA_KV=8 강제
baseline_after   → (FlashInfer auto, JIT drift 확인용)
```

각 phase마다 FlashInfer JIT 캐시(`/root/.cache/flashinfer/`)를 삭제해 재컴파일을 유도하고 측정 조건을 균일하게 만든다.

### 4.5 결과 위치

```
prefill_kv_tile_experiment/results/data/tile_kv_results.csv
prefill_kv_tile_experiment/results/plots/
```

---

## 5. 실험 2: Decode Cuda-Core KV Tile 탐색

### 5.1 목적

FlashInfer **cuda-core** decode 경로(`use_tensor_cores=False`)에서 KV tile 크기(`tile_size_per_bdx`)가 decode latency에 미치는 영향을 측정한다.

### 5.2 조작 변수

```
tile_size_per_bdx = auto, 1, 2, 4, 8
```

cuda-core decode는 thread block의 각 thread가 처리하는 KV 토큰 수를 `tile_size_per_bdx`로 결정한다. llama3_8b 조건에서 실제 처리되는 토큰 수는 다음과 같다.

```
tile_size_per_bdx=1 → KV_TILE_TOKENS = 1 × BDY × BDZ = 8
tile_size_per_bdx=2 → KV_TILE_TOKENS = 16
tile_size_per_bdx=4 → KV_TILE_TOKENS = 32
tile_size_per_bdx=8 → KV_TILE_TOKENS = 64
```

### 5.3 패치 대상

```
decode.cuh   : DISPATCH_GQA_GROUP_SIZE → tile_size_per_bdx 강제
scheduler.cuh: decode tile 파라미터 선택 로직
```

### 5.4 제외 모델

FlashInfer의 `DISPATCH_GQA_GROUP_SIZE`는 `GROUP_SIZE ∈ {1, 2, 3, 4, 8}`만 지원한다. 따라서 `llama3_405b` (GROUP_SIZE=16), `qwen2.5_7b` (GROUP_SIZE=7)는 이 실험에서 제외된다.

### 5.5 결과 위치

```
decode_kv_tile_experiment/results/data/decode_kv_results.csv
decode_kv_tile_experiment/results/plots/
```

---

## 6. 실험 3: Tensor-Core Decode × Split-K 전체 탐색

### 6.1 목적

FlashInfer tensor-core decode 경로에서 `NUM_MMA_KV`와 split-K chunk size를 교차 탐색(full factorial sweep)하여, 각 (kv_len, batch_size) 조건에서 latency-optimal한 설정을 실측한다.

### 6.2 조작 변수

**축 1: NUM_MMA_KV**

```
auto   FlashInfer 기본 선택
1      NUM_MMA_KV=1 강제
2      NUM_MMA_KV=2 강제
```

`NUM_MMA_KV=3`, `4`는 RTX 3090 / llama3_8b / head_dim=128 조건에서 kernel launch 시 `invalid argument` 오류가 발생하여 제외한다. 이는 `NUM_MMA_KV`를 크게 강제할 때 `CTA_TILE_KV`와 shared memory / register 요구량이 커지고, FlashInfer의 dispatch 로직이 원래 피하려던 invalid configuration이 만들어지기 때문이다.

**축 2: split-K chunk size**

| 설정 이름 | 내용 |
|---|---|
| `split_auto` | FlashInfer 자동 결정 |
| `split_off` | `disable_split_kv=1`, split-K 비활성화 |
| `fixed_16` ~ `fixed_8192` | KV chunk size를 해당 값으로 고정 |

### 6.3 교차 탐색 조합

```
12 split-K 설정 × 3 NUM_MMA_KV 설정 = 36 조합
각 kv_len(64개) × batch_size(5개)에 대해 실행
```

각 split-K 설정 내 실행 순서:

```
baseline_before  (FlashInfer auto, 측정 전 기준)
forced_mma1      (NUM_MMA_KV=1)
forced_mma2      (NUM_MMA_KV=2)
baseline_after   (FlashInfer auto, JIT drift 확인)
```

`baseline_before`와 `baseline_after`를 모두 측정해 JIT 재컴파일, GPU 상태, 시간 흔들림을 확인하며, plot에서는 두 baseline의 평균을 기준으로 사용한다.

### 6.4 핵심 코드: 측정 및 tile 파라미터 추출

```python
# bench_utils.py
def get_tensor_core_decode_tile_params(wrapper, q, kv_cache, ...):
    """실제 실행된 커널의 tile 파라미터를 FlashInfer JIT 로그에서 추출"""
    # FlashInfer는 JIT 컴파일 시 커널 파라미터를 로그에 기록
    # CTA_TILE_Q, CTA_TILE_KV, NUM_MMA_KV, NUM_WARPS_KV 파싱
    ...

def bench_ms(fn, warmup=100, repeat=100):
    """GPU event 기반 latency 측정"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat
```

### 6.5 결과 위치

```
decode_tensor_core_experiment/results/data/decode_tc_results_fp16.csv
decode_tensor_core_experiment/results/data/decode_tc_results_bf16.csv
decode_tensor_core_experiment/results/plots/
  ├── split_k/          # split-K별 speedup/latency
  ├── mma/              # NUM_MMA_KV별 speedup/latency
  ├── split_k_geomean/  # split-K geomean 비교
  ├── split_k_chunks/   # oracle chunk 선택 분석
  └── max_splitk/       # 최대 split-K 효과 분석
```

### 6.6 Oracle 분석 결과

각 `(kv_len, batch_size)` 포인트에서 실측된 36개 조합 중 가장 빠른 설정을 **Oracle**로 정의한다. Oracle speedup은 FlashInfer default 대비 성능 개선의 이론적 상한선을 나타낸다.

Oracle latency 분석에서 발견된 **saturation knee** 패턴:

```
BS=1  → knee ≈ KV 8192 이상 (측정 범위 밖)
BS=2  → knee ≈ KV 4096
BS=4  → knee ≈ KV 2048
BS=8  → knee ≈ KV 1024
BS=16 → knee ≈ KV 512

공통 패턴: batch_size × kv_len ≈ 8192 tokens 지점에서 latency 기울기 전환
```

Knee 이전 구간에서는 적절한 tiling과 split-K를 통해 남은 병렬 실행 여유를 활용할 수 있어 latency가 완만하게 증가한다. Knee 이후에는 병렬 처리 여유가 소진되어 latency가 KV length에 선형적으로 증가한다.

---

## 7. 실험 4: Split-K Heuristic 시뮬레이션 및 실제 패치 검증

### 7.1 목적

실험 3에서 수집한 데이터를 기반으로 개선된 split-K 선택 규칙을 제안하고, FlashInfer scheduler를 직접 패치해 실측으로 검증한다.

### 7.2 제안 Heuristic

FlashInfer auto scheduler는 occupancy를 기준으로 chunk 수를 최대화한다. 제안 방식은 이 선택을 두 가지 조건으로 제한한다.

**제안 규칙:**

```
proposed_chunks = min(default_chunks,
                      floor(kv_len / (alpha × CTA_TILE_KV)),
                      floor(beta / batch_size))
```

- **work cap**: 각 KV chunk가 최소 `alpha × CTA_TILE_KV` 토큰을 포함하도록 보장 → over-splitting 방지
- **batch cap**: batch size가 이미 충분히 크면 split 수를 추가로 제한 → merge overhead 절감

**실험에서 사용한 하이퍼파라미터:**

```
alpha         = 16
beta          = 16
CTA_TILE_KV   = 64  (llama3_8b, NUM_MMA_KV auto 커널에서 실측된 값)
```

**적용 예시 (BS=2, KV=8192):**

| 항목 | 값 |
|---|---|
| FlashInfer default chunks | 10 |
| Work cap: `8192 / (16 × 64)` | 8 |
| Batch cap: `16 / 2` | 8 |
| **Proposed chunks** | **8** |

### 7.3 시뮬레이션 (CSV 기반 오프라인 평가)

실험 3에서 수집한 CSV에는 `split_auto`, `k_1` ~ `k_20`(fixed chunk count) 각각의 latency가 이미 측정되어 있다. 시뮬레이션 단계에서는 FlashInfer를 재실행하지 않고, 제안 규칙이 선택했을 `k_N`에 해당하는 latency를 CSV에서 lookup해 성능을 추정한다.

```python
# simulate_splitk_heuristic.py 핵심 로직
def propose_chunks(kv_len, default_chunks, alpha, beta, batch_size, cta_tile_kv):
    work_cap  = int(kv_len / (alpha * cta_tile_kv))
    batch_cap = int(beta / batch_size)
    return max(1, min(default_chunks, work_cap, batch_cap))

# 가장 가까운 실측 k_N으로 매핑
closest_k = min(available_k_values, key=lambda k: abs(k - proposed_chunks))
```

**비교 대상 3가지:**

| 정책 | 설명 |
|---|---|
| FlashInfer default | `split_auto`, `NUM_MMA_KV auto` |
| Oracle | 각 `(kv_len, batch_size)`에서 가장 빠른 k_N 선택 |
| Proposed heuristic | 제안 규칙으로 k_N 선택 |

### 7.4 실제 패치 검증

시뮬레이션 결과로 유효성을 확인한 후, FlashInfer `scheduler.cuh`를 직접 수정해 실제 decode 경로에서 재측정한다.

**패치 대상:**

```
/root/capstone-yonsei/venv/lib/python3.10/site-packages/
  flashinfer/data/include/flashinfer/attention/scheduler.cuh
```

**패치 핵심 코드 (`patch_flashinfer_splitk_heuristic.py`):**

```cpp
// BEGIN capstone split-k heuristic guard
constexpr int64_t splitk_alpha = 16;
constexpr int64_t splitk_beta  = 16;
constexpr int64_t cta_tile_kv_proxy_tokens = 64;

// 현재 batch의 최대 kv_len 계산
int64_t max_effective_kv_len = *std::max_element(
    effective_kv_len_arr.begin(), effective_kv_len_arr.end());

// work cap: chunk 하나당 최소 alpha × CTA_TILE_KV 토큰 보장
const int64_t min_work_per_chunk_pages = std::max<int64_t>(
    int64_t(min_kv_chunk_size),
    ceil_div(splitk_alpha * cta_tile_kv_proxy_tokens, int64_t(page_size)));

const int64_t default_chunks =
    std::max<int64_t>(1, ceil_div(max_effective_kv_len, kv_chunk_size));
const int64_t work_cap =
    std::max<int64_t>(1, max_effective_kv_len / min_work_per_chunk_pages);

// batch cap: batch size 증가 시 추가 split 억제
const int64_t batch_cap =
    std::max<int64_t>(1, splitk_beta / int64_t(batch_size));

// 세 조건 중 최솟값 적용
const int64_t proposed_chunks =
    std::max<int64_t>(1, std::min(default_chunks,
                                  std::min(work_cap, batch_cap)));

kv_chunk_size = std::max<int64_t>(
    int64_t(min_kv_chunk_size),
    ceil_div(max_effective_kv_len, proposed_chunks));
split_kv = enable_cuda_graph || kv_chunk_size < max_effective_kv_len;
// END capstone split-k heuristic guard
```

패치 적용/복원은 명령줄로 관리한다.

```bash
python patch_flashinfer_splitk_heuristic.py apply   # 패치 적용
python patch_flashinfer_splitk_heuristic.py restore  # 원본 복원
python patch_flashinfer_splitk_heuristic.py status   # 상태 확인
```

### 7.5 실제 패치 실험 설정

```
model        = llama3_8b
batch_sizes  = 1, 2, 4, 8, 16
kv_len       = 128 ~ 8192 (128 간격)
alpha        = 16, beta = 16
dtype        = float16
backend      = fa2
```

결과 CSV:

```
splitk_heuristic_simulation/results/data/
  ├── decode_tc_results_fp16_patched_split_auto_alpha16_beta16.csv
  └── decode_tc_results_fp16_reference_default_oracle.csv
```

---

## 8. 핵심 코드 설명

### 8.1 디렉터리 구조

```
capstone-yonsei/
├── prefill_kv_tile_experiment/
│   ├── test_tile_kv.py          # prefill latency 측정
│   ├── patch_prefill.py         # prefill.cuh NUM_MMA_KV 패치/복원
│   ├── bench_utils.py           # 공통 측정 유틸
│   └── run_tile_kv.sh           # 전체 실험 드라이버
│
├── decode_kv_tile_experiment/
│   ├── test_decode_kv.py        # cuda-core decode latency 측정
│   ├── patch_decode.py          # decode.cuh / scheduler.cuh 패치/복원
│   └── run_decode_kv.sh         # 전체 실험 드라이버
│
├── decode_tensor_core_experiment/
│   ├── test_decode_tc.py        # tensor-core decode latency 측정
│   ├── patch_decode_tc.py       # prefill.cuh NUM_MMA_KV 패치/복원
│   ├── bench_utils.py           # tile 파라미터 추출 포함
│   ├── plot.py                  # split-K / MMA tile 결과 시각화
│   ├── run_decode_tc_split_sweep.sh  # split-K × NUM_MMA_KV sweep 드라이버
│   └── run_decode_tc_experience.sh  # dtype × model 전체 순차 실행
│
└── splitk_heuristic_simulation/
    ├── simulate_splitk_heuristic.py      # CSV 기반 오프라인 시뮬레이션
    ├── patch_flashinfer_splitk_heuristic.py  # scheduler.cuh 직접 패치
    ├── run_patched_split_auto_experiment.py  # 패치 후 실측 벤치마크
    ├── compare_patched_results.py            # default / patched / oracle 비교
    └── prepare_reference_results.py          # reference CSV 생성
```

### 8.2 FlashInfer 패치 방식

모든 패치는 **가역적(reversible)**이다. 원본 파일을 백업한 뒤 대상 파일을 수정하고, 복원 시 백업에서 원상복구한다.

```python
# 공통 패치 패턴 (patch_*.py)
BACKUP = SCHEDULER.with_suffix(SCHEDULER.suffix + ".before_patch")

def apply_patch(...):
    text = TARGET.read_text()
    if PATCH_MARKER in text:
        raise SystemExit("already patched")
    BACKUP.write_text(text)  # 백업
    TARGET.write_text(text.replace(ORIGINAL, PATCHED))

def restore():
    TARGET.write_text(BACKUP.read_text())  # 복원
```

### 8.3 Paged KV Cache 구성

FlashInfer decode는 contiguous dense KV가 아니라 paged KV cache 형식을 사용한다.

```python
# test_decode_tc.py
def make_paged_kv(seq_lens, num_kv_heads, head_dim, page_size, dtype):
    pages_per_seq = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages = sum(pages_per_seq)

    # Layout: [num_pages, K/V=2, page_size, num_kv_heads, head_dim]
    paged_kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )
    # kv_indptr: 각 sequence의 page 시작 인덱스
    # kv_last_page_len: 마지막 page의 실제 토큰 수
    ...
    return paged_kv_cache, kv_indptr, kv_indices, kv_last_page_len
```

### 8.4 Speedup 계산 기준

모든 실험에서 speedup은 FlashInfer auto baseline 대비로 계산한다.

```
speedup = baseline_ms / experiment_ms
```

`speedup > 1.0`이면 해당 설정이 FlashInfer auto보다 빠른 것이다. Geomean speedup은 kv_len 전체 포인트에 대한 기하평균으로 계산한다.

---

## 9. 결과 요약

### 9.1 실험 3: Tensor-Core Decode Split-K 탐색

- FlashInfer auto split-K는 단순히 GPU occupancy를 채우는 방식으로 동작하며, 각 batch / kv_len 조건에서 최적 chunk 수와 다를 수 있다.
- **Split-K off가 짧은 KV length에서 default보다 빠른 구간이 존재**한다. 이는 merge overhead 없이 단순 decode가 더 효율적인 구간이 있음을 의미한다.
- `NUM_MMA_KV=1` vs `2` 차이는 split-K 설정만큼 일관된 영향을 보이지 않았다. Decode 경로에서는 tile 크기보다 **split-K 선택이 더 큰 영향**을 준다.
- Oracle 분석에서 `batch_size × kv_len ≈ 8192 tokens` 지점에서 saturation knee가 관찰된다.

### 9.2 실험 4: 제안 Heuristic 실측 결과

| Batch Size | 측정 포인트 수 | Proposed Speedup (geomean) | Oracle Speedup (geomean) |
|---:|---:|---:|---:|
| 1 | 64 | 1.013× | 1.057× |
| 2 | 64 | **1.049×** | 1.071× |
| 4 | 64 | **1.049×** | 1.065× |
| 8 | 64 | 1.029× | 1.053× |
| 16 | 64 | 1.001× | ~1.000× |

- **Batch 2, 4에서 geomean latency 약 4.9% 개선** (oracle gap의 약 70% 회복)
- Batch 8에서 약 2.9% 개선
- Batch 16에서는 FlashInfer default가 이미 split 수를 적게 선택하므로 개선 여지 없음
- 점별 최대 speedup: BS=2에서 **1.305×** (특정 kv_len 구간)

### 9.3 해석

제안 heuristic이 효과적인 조건:

```
작은 batch size (BS ≤ 8) × 짧은~중간 KV length
  → FlashInfer auto가 과도한 split-K 선택 → merge overhead 지배
  → work_cap / batch_cap이 default_chunks보다 작게 작동 → 개선 효과
```

개선 여지가 없는 조건:

```
큰 batch size (BS = 16)
  → GPU block 목표를 채우기 위해 FlashInfer auto가 이미 split 수를 적게 선택
  → 제안 규칙의 cap이 default와 동일하게 작동 → 차이 없음
```

---

## 10. 결론 및 향후 과제

### 10.1 결론

본 연구는 FlashInfer tensor-core decode 경로에서 split-K scheduler가 latency에 결정적인 영향을 미침을 실측으로 보였다. FlashInfer의 occupancy 중심 split-K 선택은 소형 batch에서 over-splitting을 유발할 수 있으며, 최소 chunk work 조건을 추가하는 간단한 guard(`proposed_chunks = min(default, work_cap, batch_cap)`)만으로도 실측 geomean latency를 최대 4.9% 개선할 수 있다.

이 규칙은 기존 scheduler 로직을 대체하지 않고 위에 덧씌우는 방식이라, 추후 FlashInfer 업스트림에 최소한의 수정으로 통합 가능하다.

### 10.2 한계

- **실험 환경 한정**: RTX 3090, LLaMA-3 8B, float16 조건에서 도출된 결과이며, 다른 GPU(A100, H100 등)나 모델 구성에 대한 일반화는 별도 검증이 필요하다.
- **하이퍼파라미터 의존성**: `alpha=16`, `beta=16`, `CTA_TILE_KV=64`는 이 환경에서 최적화된 값이다. 다른 `head_dim`, `page_size`, SM 수를 가진 하드웨어에서는 재탐색이 필요하다.
- **측정 노이즈**: 소규모 배치의 짧은 KV 구간에서 측정 흔들림이 있으며, 일부 구간에서 `speedup < 1.0`인 포인트도 존재한다.

### 10.3 향후 과제

| 과제 | 내용 |
|---|---|
| 다중 GPU 검증 | A100, H100에서 동일 heuristic 재측정 및 파라미터 재탐색 |
| 동적 CTA_TILE_KV 감지 | 현재 `CTA_TILE_KV=64`는 상수로 고정; JIT 컴파일 시 실제 kernel 파라미터를 읽어 동적으로 설정 |
| 다중 모델 일반화 | llama3_70b, qwen2.5_72b 등 다른 head config에서 동일 heuristic 유효성 검증 |
| vLLM / SGLang 통합 | FlashInfer를 backend로 사용하는 서빙 프레임워크에 패치 적용 |
| 런타임 alpha/beta 자동 탐색 | GPU 특성(SM 수, shared memory 크기)에 따라 alpha, beta를 자동으로 결정하는 heuristic 설계 |

---

## 부록: 주요 실행 명령어

### 실험 3: Tensor-Core Split-K × MMA 전체 sweep

```bash
cd /root/capstone-yonsei/decode_tensor_core_experiment

# 전체 study sweep (kv_len=128~8192)
nohup env SPLIT_MODES=study KV_LENS="$(seq -s ' ' 128 128 8192)" \
  bash run_decode_tc_split_sweep.sh llama3_8b \
  > results/logs/run_split_study_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 실험 4: Heuristic 시뮬레이션

```bash
cd /root/capstone-yonsei

# CSV 기반 오프라인 시뮬레이션
python splitk_heuristic_simulation/simulate_splitk_heuristic.py \
  --model llama3_8b --baseline before --alphas "2 4 8 16" --beta 16
```

### 실험 4: 실제 패치 후 측정

```bash
# reference CSV 준비
python splitk_heuristic_simulation/prepare_reference_results.py --model llama3_8b

# FlashInfer scheduler 패치 후 벤치마크 실행
nohup bash splitk_heuristic_simulation/run_patched_split_auto_experiment.sh \
  > splitk_heuristic_simulation/results/logs/run_patched_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 결과 비교 및 시각화
python splitk_heuristic_simulation/compare_patched_results.py --model llama3_8b
```

### 패치 상태 관리

```bash
# 패치 상태 확인
python splitk_heuristic_simulation/patch_flashinfer_splitk_heuristic.py status

# 원본 복원
python splitk_heuristic_simulation/patch_flashinfer_splitk_heuristic.py restore
```
