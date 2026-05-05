# Decode KV Tile Experiment

FlashInfer cuda-core `BatchDecodeWithPagedKVCacheWrapper`에서
`tile_size_per_bdx`를 강제로 바꿔가며 decode latency를 측정하는 실험입니다.

## 대상

`prefill_kv_tile_experiment`와 같은 모델 target 구조를 사용합니다.
인자를 주지 않으면 `llama`, `qwen`, `gemma` 그룹 전체를 실행합니다.

```text
batch_size = 8
page_size = 16
dtype = fp16
```

지원 모델:

```text
llama3_8b    num_qo_heads=32   num_kv_heads=8   head_dim=128
llama3_70b   num_qo_heads=64   num_kv_heads=8   head_dim=128
qwen2.5_72b  num_qo_heads=64   num_kv_heads=8   head_dim=128
gemma2_9b    num_qo_heads=16   num_kv_heads=8   head_dim=256
gemma2_27b   num_qo_heads=32   num_kv_heads=16  head_dim=128
```

현재 FlashInfer decode의 `DISPATCH_GQA_GROUP_SIZE`는
`GROUP_SIZE = num_qo_heads / num_kv_heads`가 `1, 2, 3, 4, 8`인 경우만
지원합니다. 그래서 아래 prefill preset은 decode 실험에서 제외합니다.

```text
llama3_405b  GROUP_SIZE=16
qwen2.5_7b   GROUP_SIZE=7
```

`use_tensor_cores=False`, `backend="auto"`로 고정해서 `decode.cuh`의 cuda-core
batch decode 경로를 사용합니다.

## 파일 역할

- `run_decode_kv.sh`: 전체 실험 드라이버
- `test_decode_kv.py`: 단일 모델 설정으로 `kv_len=128..8192`를 128 간격으로 측정
- `patch_decode.py`: FlashInfer JIT include의 `decode.cuh`/`scheduler.cuh` 패치/복원
- `bench_utils.py`: warmup/repeat 측정, tile 파라미터 추출, TFLOPS/GB/s 계산
- `plot.py`: baseline 대비 tile별 speedup/latency/baseline drift plot 생성

현재 source/editable 환경에서는 보통 아래 경로를 패치합니다.

```text
/root/flashinfer/flashinfer/data/include/flashinfer/attention/decode.cuh
/root/flashinfer/flashinfer/data/include/flashinfer/attention/scheduler.cuh
```

venv site-packages에 설치된 환경이면 `patch_decode.py`가 그 경로를 우선 사용합니다.

## 실행 방법

```bash
cd /root/capstone-yonsei/decode_kv_tile_experiment
bash run_decode_kv.sh
```

특정 모델 그룹만 실행:

```bash
bash run_decode_kv.sh llama
bash run_decode_kv.sh qwen
bash run_decode_kv.sh gemma
```

특정 모델만 실행:

```bash
bash run_decode_kv.sh llama3_8b
bash run_decode_kv.sh llama3_70b
bash run_decode_kv.sh qwen2.5_72b
bash run_decode_kv.sh gemma2_9b
bash run_decode_kv.sh gemma2_27b
```

기본 실행 스크립트는 `/root/venv/bin/python`을 사용합니다. 다른 Python을 쓰려면:

```bash
PYTHON_BIN=/path/to/python bash run_decode_kv.sh
```

FlashInfer JIT는 `ninja` 실행 파일을 PATH에서 찾습니다. `run_decode_kv.sh`는
`PYTHON_BIN`이 있는 디렉터리를 PATH 앞에 붙입니다.

일부 tile만 smoke test:

```bash
TILE_SIZE_PER_BDX_VALS="1 2 4 8" bash run_decode_kv.sh llama3_8b
```

짧은 KV length로 smoke test:

```bash
KV_LENS="128 1024" TILE_SIZE_PER_BDX_VALS="1 4" bash run_decode_kv.sh llama3_8b
```

correctness reference를 생략하고 latency만 측정:

```bash
SKIP_CORRECTNESS=1 bash run_decode_kv.sh
```

batch size 변경:

```bash
BATCH_SIZE=16 bash run_decode_kv.sh
```

## 실험 순서

```text
baseline_before  FlashInfer auto
forced_tile1     tile_size_per_bdx=1
forced_tile2     tile_size_per_bdx=2
...
forced_tile8     tile_size_per_bdx=8
baseline_after   FlashInfer auto
```

각 phase마다 `/root/.cache/flashinfer/`를 삭제해서 JIT 재컴파일을 유도합니다.
각 `kv_len`은 warmup 100회 후 timed repeat 100회로 latency를 측정합니다.

## 결과 위치

```text
results/data/decode_kv_results.csv
```

Plot 생성:

```bash
cd /root/capstone-yonsei/decode_kv_tile_experiment
/root/venv/bin/python plot.py
```

다른 모델 plot 생성:

```bash
/root/venv/bin/python plot.py --model qwen2.5_7b
```

필요 패키지가 없다면:

```bash
/root/venv/bin/pip install matplotlib pandas
```

생성 파일:

```text
results/plots/llama3_8b_decode_speedup_vs_baseline.png
results/plots/llama3_8b_decode_latency.png
results/plots/llama3_8b_decode_baseline_drift.png
```

주요 컬럼:

- `label`: 실험 라벨
- `kv_len`: KV cache length
- `TILE_SIZE_PER_BDX`: 실제 실행된 decode 커널의 template 값
- `KV_TILE_TOKENS`: `TILE_SIZE_PER_BDX * BDY * BDZ`
- `ms`: 평균 latency
- `tflops`: decode FLOPS proxy
- `gb_per_s_est`: K/V read 기준 단순 bandwidth proxy

## 해석 기준

```text
speedup = baseline_ms / experiment_ms
```

`speedup > 1`이면 forced tile이 baseline보다 빠른 것입니다.

llama3_8b 기본 설정에서는 `GROUP_SIZE=4`, `HEAD_DIM=128`, fp16이라
대략 다음처럼 해석할 수 있습니다.

```text
tile_size_per_bdx=1 -> KV_TILE_TOKENS=8
tile_size_per_bdx=2 -> KV_TILE_TOKENS=16
tile_size_per_bdx=4 -> KV_TILE_TOKENS=32
tile_size_per_bdx=8 -> KV_TILE_TOKENS=64
```
