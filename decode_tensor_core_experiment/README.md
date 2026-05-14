# Decode Tensor-Core Split-k / MMA Tile Experiment

FlashInfer `BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)` 경로에서
decode latency가 tensor-core tile 설정과 split-k 설정에 따라 어떻게 달라지는지 측정하는 실험입니다.

현재 주 실험은 여러 모델과 dtype에 대해 다음 두 축을 함께 sweep합니다.

```text
1. tensor-core KV tile 크기
   - FlashInfer auto
   - forced NUM_MMA_KV=1
   - forced NUM_MMA_KV=2

2. split-k 설정
   - FlashInfer auto split-k
   - split-k off
   - fixed_split_size = 16, 32, 64, ..., 8192
```

## 실험 목적

decode attention은 prefill보다 query 길이가 매우 짧고 KV cache를 길게 읽는 형태라서,
연산량보다 KV 메모리 접근, 병렬화 방식, reduction overhead의 영향을 크게 받을 수 있습니다.

이 실험은 다음 질문을 확인하기 위한 것입니다.

- FlashInfer의 tensor-core decode auto 설정이 항상 최적인가?
- `NUM_MMA_KV`를 1 또는 2로 강제했을 때 latency가 달라지는가?
- split-k를 끄거나, `fixed_split_size`를 강제로 설정하면 긴 KV length에서 성능이 좋아지는가?
- split-k 설정과 tensor-core tile 설정 사이에 상호작용이 있는가?

## 조작하는 값

### NUM_MMA_KV

`NUM_MMA_KV`는 FlashInfer tensor-core decode가 내부적으로 사용하는 FA2 batch prefill kernel의
KV 방향 MMA tile 크기를 결정하는 template 파라미터입니다.

FlashInfer의 `use_tensor_cores=True` decode는 내부적으로 `get_batch_prefill_module(...)` 경로를 사용하므로,
이 실험에서는 `prefill.cuh`의 `DISPATCH_NUM_MMA_KV`를 패치해서 값을 강제합니다.

실험에서 사용하는 값:

```text
auto  FlashInfer 기본 선택
1     forced NUM_MMA_KV=1
2     forced NUM_MMA_KV=2
```

### split-k

split-k는 하나의 긴 KV cache를 여러 KV chunk로 나누어 병렬 처리한 뒤 결과를 합치는 방식입니다.
FlashInfer scheduler에서는 `fixed_split_size`가 KV chunk 크기로 사용됩니다.

예를 들어:

```text
kv_len = 8192
fixed_split_size = 512
num_chunks_kv = ceil(8192 / 512) = 16
```

실험에서 사용하는 split-k 설정:

```text
split_auto    FlashInfer가 split-k chunk size를 자동 결정
split_off     disable_split_kv=1, split-k 비활성화
fixed_16      fixed_split_size=16
fixed_32      fixed_split_size=32
fixed_64      fixed_split_size=64
fixed_128     fixed_split_size=128
fixed_256     fixed_split_size=256
fixed_512     fixed_split_size=512
fixed_1024    fixed_split_size=1024
fixed_2048    fixed_split_size=2048
fixed_4096    fixed_split_size=4096
fixed_8192    fixed_split_size=8192
```

## 수행하는 조합

`SPLIT_MODES=study`는 아래 모든 split-k 설정에 대해 `MMA auto`, `MMA=1`, `MMA=2`를 측정합니다.

```text
split_auto    + MMA auto, 1, 2
split_off     + MMA auto, 1, 2
fixed_16      + MMA auto, 1, 2
fixed_32      + MMA auto, 1, 2
fixed_64      + MMA auto, 1, 2
fixed_128     + MMA auto, 1, 2
fixed_256     + MMA auto, 1, 2
fixed_512     + MMA auto, 1, 2
fixed_1024    + MMA auto, 1, 2
fixed_2048    + MMA auto, 1, 2
fixed_4096    + MMA auto, 1, 2
fixed_8192    + MMA auto, 1, 2
```

각 split-k 설정마다 내부 실행 순서는 다음과 같습니다.

```text
baseline_before  FlashInfer tensor-core auto NUM_MMA_KV
forced_mma1      NUM_MMA_KV=1
forced_mma2      NUM_MMA_KV=2
baseline_after   FlashInfer tensor-core auto NUM_MMA_KV
```

`baseline_before`와 `baseline_after`를 모두 측정하는 이유는 JIT 재컴파일, GPU 상태, 시간에 따른
측정 흔들림을 확인하기 위해서입니다. plot에서는 기본적으로 두 baseline의 평균을 기준으로 사용합니다.

## MMA를 1, 2로 제한하는 이유

현재 RTX 3090 / `llama3_8b` / `head_dim=128` 조건에서 `NUM_MMA_KV=3`과 `NUM_MMA_KV=4`를 강제하면
FlashInfer FA2 tensor-core kernel launch가 `invalid argument`로 실패하는 것이 확인되었습니다.

tensor-core decode는 decode 형태의 짧은 query 길이에서 FA2 batch prefill kernel을 사용합니다.
이때 `NUM_MMA_KV`를 크게 강제하면 내부적으로 `CTA_TILE_KV`와 shared memory/register 요구량이 커지고,
FlashInfer의 원래 dispatch 로직이 피하려던 invalid kernel configuration이 만들어질 수 있습니다.

따라서 정식 sweep에서는 안정적으로 실행되는 `NUM_MMA_KV=1`, `NUM_MMA_KV=2`만 사용합니다.
`NUM_MMA_KV=3` 이상은 별도 탐색용으로만 시도하는 것이 좋습니다.

## 기본 실험 설정

```text
model = llama3_8b
batch_size = 8
page_size = 16
backend = fa2
dtype = fp16 또는 bf16
kv_len = 128..8192, step 128
warmup = 100
repeat = 100
correctness = skip
```

전체 `kv_len` sweep은 다음 길이를 사용합니다.

```text
128, 256, 384, ..., 8064, 8192
```

## 실행

정식 study 실험:

```bash
cd /root/capstone-yonsei/decode_tensor_core_experiment

nohup env SPLIT_MODES=study KV_LENS="$(seq -s ' ' 128 128 8192)" bash run_decode_tc_split_sweep.sh llama3_8b > results/logs/run_decode_tc_split_study_llama3_8b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

모든 dtype/model을 순차 실행:

```bash
nohup bash run_decode_tc_experience.sh > results/logs/run_decode_tc_experience_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

기본 조합:

```text
DTYPES = fp16, bf16
MODELS = llama3_8b, llama3_70b, qwen2.5_72b, gemma2_9b, gemma2_27b
```

일부만 실행:

```bash
DTYPES="fp16" MODELS="llama3_70b gemma2_27b" bash run_decode_tc_experience.sh
```

`run_decode_tc_experience.sh`는 dtype/model을 동시에 실행하지 않고 순차 실행합니다.
성능 측정 간 GPU 간섭을 피하기 위한 방식입니다. 각 dtype/model의 상세 로그는 별도 파일로 저장됩니다.

```text
results/logs/run_decode_tc_split_study_fp16_llama3_8b_*.log
results/logs/run_decode_tc_split_study_bf16_llama3_8b_*.log
```

로그 확인:

```bash
tail -f "$(ls -t results/logs/run_decode_tc_experience*.log | head -n 1)"
```

빠른 pilot:

```bash
bash run_decode_tc_split_sweep.sh llama3_8b
```

pilot은 다음 축소 조합만 실행합니다.

```text
split modes = auto, off, fixed_256, fixed_512, fixed_1024, fixed_2048
kv_len = 128, 512, 1024, 2048, 4096, 8192
```

실험 중 특정 split-k 설정이 실패해도 기본적으로 다음 설정으로 계속 진행합니다.
실패 시 즉시 중단하려면:

```bash
STOP_ON_ERROR=1 bash run_decode_tc_split_sweep.sh llama3_8b
```

## 결과

CSV:

```text
results/data/decode_tc_results_fp16.csv
results/data/decode_tc_results_bf16.csv
```

주요 컬럼:

```text
label              실험 라벨
kv_len             KV cache length
fixed_split_size   강제 split-k chunk size
disable_split_kv   split-k off 여부
CTA_TILE_Q         실제 실행된 tensor-core CTA Q tile
CTA_TILE_KV        실제 실행된 tensor-core CTA KV tile
NUM_MMA_KV         실제 실행된 NUM_MMA_KV
ms                 평균 latency
tflops             decode FLOPS proxy
gb_per_s_est       K/V read 기준 bandwidth proxy
```

label 예시:

```text
[baseline_before] llama3_8b_fp16_split_auto
[experiment] llama3_8b_fp16_split_fixed_512_num_mma_kv_2
```

## Plot

split-k별 speedup/latency:

```bash
/root/capstone-yonsei/venv/bin/python plot.py --split --model llama3_8b --mma auto
/root/capstone-yonsei/venv/bin/python plot.py --split --model llama3_8b --mma 1
/root/capstone-yonsei/venv/bin/python plot.py --split --model llama3_8b --mma 2
```

bf16 결과를 그릴 때:

```bash
/root/capstone-yonsei/venv/bin/python plot.py --csv results/data/decode_tc_results_bf16.csv --split --model llama3_8b --mma 2
```

speedup 해석:

```text
speedup = split_auto_ms / split_mode_ms
```

`speedup > 1`이면 해당 split-k 설정이 FlashInfer auto split-k보다 빠른 것입니다.

NUM_MMA_KV별 speedup/latency:

```bash
/root/capstone-yonsei/venv/bin/python plot.py --model llama3_8b
```

## 파일 역할

```text
run_decode_tc_experience.sh     dtype x model 전체 순차 실행 드라이버
run_decode_tc_split_sweep.sh  split-k x NUM_MMA_KV 전체 sweep 드라이버
run_decode_tc.sh              단일 split-k 조건에서 baseline/MMA=1/MMA=2 실행
test_decode_tc.py             실제 FlashInfer decode benchmark
patch_decode_tc.py            FlashInfer prefill.cuh NUM_MMA_KV 패치/복원
bench_utils.py                latency 측정, tile parameter 추출
plot.py                       결과 plot 생성
```
