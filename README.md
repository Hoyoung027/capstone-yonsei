# Capstone Yonsei

FlashInfer attention kernel의 tile/scheduler 설정을 바꿔가며 LLM prefill/decode latency를 측정하는 실험 repo입니다.

현재 실험은 `/root/capstone-yonsei/venv` 가상환경을 기준으로 실행합니다.

```bash
cd /root/capstone-yonsei
source venv/bin/activate
```

환경 요약:

```text
GPU             NVIDIA GeForce RTX 3090
Python          3.10
PyTorch         2.5.1+cu121
FlashAttention  2.x
FlashInfer      source build
```

## 파일 구조

```text
capstone-yonsei/
├── README.md
├── requirements.txt
├── venv/
│
├── prefill_kv_tile_experiment/
│   ├── README.md
│   ├── run_tile_kv.sh
│   ├── test_tile_kv.py
│   ├── patch_prefill.py
│   ├── bench_utils.py
│   ├── plot.py
│   └── results/
│
├── decode_kv_tile_experiment/
│   ├── README.md
│   ├── run_decode_kv.sh
│   ├── test_decode_kv.py
│   ├── patch_decode.py
│   ├── bench_utils.py
│   ├── plot.py
│   └── results/
│
└── decode_tensor_core_experiment/
    ├── README.md
    ├── run_decode_tc.sh
    ├── run_decode_tc_split_sweep.sh
    ├── smoke_decode_tc.sh
    ├── test_decode_tc.py
    ├── patch_decode_tc.py
    ├── bench_utils.py
    ├── plot.py
    └── results/
```

## 실험 요약

### 1. `prefill_kv_tile_experiment`

FlashInfer prefill 경로에서 KV 방향 tensor-core tile 크기를 바꿔보는 실험입니다.

```text
조작 값: NUM_MMA_KV
대상: prefill.cuh
측정: seq_len=128..8192에서 FlashInfer auto baseline vs forced NUM_MMA_KV
```

### 2. `decode_kv_tile_experiment`

FlashInfer cuda-core decode 경로에서 decode KV tile 크기를 바꿔보는 실험입니다.

```text
조작 값: tile_size_per_bdx
대상: decode.cuh, scheduler.cuh
조건: BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=False)
측정: kv_len=128..8192에서 FlashInfer auto baseline vs forced tile_size_per_bdx
```

### 3. `decode_tensor_core_experiment`

FlashInfer tensor-core decode 경로에서 `NUM_MMA_KV`와 split-k 설정을 함께 바꿔 latency를 측정합니다.

```text
조건: BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend=fa2)
조작 값 1: NUM_MMA_KV = auto, 1, 2
조작 값 2: split-k = auto, off, fixed_16, fixed_32, ..., fixed_8192
대상: prefill.cuh
측정: llama3_8b, batch=8, page=16, kv_len=128..8192
```

`use_tensor_cores=True` decode는 내부적으로 FA2 batch prefill kernel을 사용하므로,
tensor-core decode tile 설정은 `decode.cuh`가 아니라 `prefill.cuh`의 `NUM_MMA_KV` dispatch를 패치합니다.

`NUM_MMA_KV`를 1, 2로 제한하는 이유:

```text
RTX 3090 / llama3_8b / head_dim=128 조건에서 NUM_MMA_KV=3,4는
FlashInfer FA2 tensor-core kernel launch가 invalid argument로 실패했습니다.
정식 sweep에서는 안정적으로 동작하는 NUM_MMA_KV=1,2만 사용합니다.
```

## 결과 위치

각 실험 디렉터리 아래에 결과가 저장됩니다.

```text
results/data/
results/logs/
results/plots/
```

## 환경 확인

```bash
cd /root/capstone-yonsei
source venv/bin/activate

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); print("available", torch.cuda.is_available(), "count", torch.cuda.device_count())'
python -c 'import flash_attn; print("flash_attn ok")'
python -c 'import flashinfer; print("flashinfer ok")'
nvidia-smi
```
