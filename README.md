# Capstone Yonsei — LLM 추론 서빙 환경 설정

## 환경 정보

| 항목 | 내용 |
|------|------|
| **Platform** | VESSL Workspace |
| **GPU** | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| **CUDA Driver** | 12.4 |
| **CUDA Toolkit (nvcc)** | 12.1 |
| **Python** | 3.10.12 |
| **PyTorch** | 2.5.1+cu121 |
| **FlashAttention** | 2.8.3 |
| **FlashInfer** | 0.6.7 |

---

## 패키지 설치

> FlashInfer stable wheel이 CUDA 12.6+만 지원하므로 소스 빌드로 진행

### 1. 빌드 도구

```bash
pip install --no-cache-dir wheel ninja packaging psutil cmake
```

### 2. PyTorch (CUDA 12.1)

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. FlashAttention 2

```bash
pip install --no-cache-dir flash-attn --no-build-isolation
```

### 4. FlashInfer (소스 빌드)

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --no-cache-dir -v .
```

> 빌드 시간: ninja 병렬 빌드 기준 30~60분 예상  
> RAM 부족 시: `MAX_JOBS=4 pip install --no-cache-dir -v .`

---

## 설치 확인

### PyTorch + CUDA

```bash
python -c "
import torch
print('torch       :', torch.__version__)
print('CUDA avail  :', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU         :', torch.cuda.get_device_name(0))
"
```

### FlashAttention 2

```bash
python -c "
import torch
import flash_attn
from flash_attn import flash_attn_func

print('flash_attn version:', flash_attn.__version__)

# 동작 확인
b, s, h, d = 2, 512, 16, 128
q = torch.randn(b, s, h, d, device='cuda', dtype=torch.float16)
k = torch.randn(b, s, h, d, device='cuda', dtype=torch.float16)
v = torch.randn(b, s, h, d, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v, causal=True)
print('flash_attn output shape:', out.shape)
print('FlashAttention OK')
"
```

### FlashInfer

```bash
python -c "
import torch
import flashinfer

print('flashinfer version:', flashinfer.__version__)

# single decode 동작 확인
q = torch.randn(32, 128, device='cuda', dtype=torch.float16)      # [num_heads, head_dim]
k = torch.randn(128, 32, 128, device='cuda', dtype=torch.float16) # [kv_len, num_heads, head_dim]
v = torch.randn(128, 32, 128, device='cuda', dtype=torch.float16)
out = flashinfer.single_decode_with_kv_cache(q, k, v)
print('flashinfer output shape:', out.shape)
print('FlashInfer OK')
"
```

### 전체 한 번에 확인

```bash
python -c "
import torch, flash_attn, flashinfer
print(f'torch      {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')
print(f'flash_attn {flash_attn.__version__}')
print(f'flashinfer {flashinfer.__version__}')
"
```

---

## 실험 스크립트

### 파일 구조

```
capstone-yonsei/
├── inspect_tile_config.py   # FlashInfer 커널 URI / 타일 설정 관찰
├── bench_attention.py       # FlashAttention vs FlashInfer 성능 벤치마크
└── results/
    └── bench_results.csv    # 벤치마크 결과 (실행 후 생성)
```

---

### 1. 커널 선택 / 타일 설정 관찰

`inspect_tile_config.py` — prefill/decode 상황에서 FlashInfer가 어떤 커널을 선택하고 타일 설정이 어떻게 구성되는지 출력한다.

```bash
python inspect_tile_config.py
```

**출력 내용:**
- GPU SM 버전 및 예상 `CTA_TILE_Q`
- 각 입력 조합(seq_len, head_dim, dtype)별로 선택된 **커널 URI**
- JIT 컴파일된 `.inc` 파일에서 파싱한 컴파일 타임 설정 (`HEAD_DIM`, `DType`, `POS_ENCODING_MODE` 등)
- RTX 3090 (SM 8.6) 기준: prefill `CTA_TILE_Q = 64` (SM≥9이면 128)

**타일 결정 규칙 요약:**

| 시나리오 | 설정 이름 | RTX 3090 값 | 결정 방식 |
|----------|-----------|-------------|-----------|
| Prefill  | `CTA_TILE_Q` | 64 | SM < 9 → 64, SM ≥ 9 → 128 |
| Prefill  | `CTA_TILE_KV` | head_dim 기반 | `NUM_MMA_KV × NUM_WARPS_KV × 16` |
| Decode   | `tile_size_per_bdx` | 4 | head_dim / (vec_size × bdx) |

---

### 2. 성능 벤치마크

`bench_attention.py` — PyTorch SDPA / FlashAttention 2 / FlashInfer를 prefill과 decode 시나리오에서 비교한다.

```bash
python bench_attention.py
```

**출력 내용:**
- 각 입력 조합별 선택된 FlashInfer URI
- `latency (ms)` / `TFLOPS` / `speedup vs SDPA`
- FlashInfer vs FlashAttention 2 speedup

**예시 출력:**
```
  prefill seq= 2048 hd=128 float16  [CTA_TILE_Q=64]
  URI : single_prefill_with_kv_cache_...
  커널          latency(ms)   TFLOPS   speedup vs SDPA
  ────────────────────────────────────────────────────
  PyTorch SDPA       x.xxxx    x.xxx            1.00x
  FlashAttn2         x.xxxx    x.xxx            x.xxx
  FlashInfer         x.xxxx    x.xxx            x.xxx
  FlashInfer vs FlashAttn2 speedup: x.xx x
```

결과는 `results/bench_results.csv`에 저장된다.

---

## API 로깅 (상세 디버그)

FlashInfer 실행 흐름을 상세히 보려면 환경 변수로 로깅을 활성화한다.

```bash
# 레벨: 0=off, 1=기본, 3=상세, 5=통계
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=stdout
python bench_attention.py
```

---

## GPU 상태 확인

```bash
nvidia-smi
```
