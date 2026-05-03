# KV Tile Experiment

FlashInfer prefill 커널의 KV 방향 tile 크기를 강제로 바꿔가며
baseline 대비 latency / TFLOPS 변화를 측정하는 실험 코드입니다.

## 파일 역할

- `run_tile_kv.sh`: 전체 실험 드라이버
- `test_tile_kv.py`: 단일 모델 설정에 대해 `seq_len=128..8192` 측정
- `patch_prefill.py`: 가상환경 안의 FlashInfer `prefill.cuh`를 패치/복원
- `bench_utils.py`: warmup/repeat 측정, tile 파라미터 추출, TFLOPS 계산 유틸

## 실험 순서


### FlashInfer (소스 빌드)

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --no-cache-dir -v .
```

`run_tile_kv.sh`는 아래 순서로 실행합니다.

```text
baseline_before  FlashInfer auto
forced_mma1      NUM_MMA_KV=1
forced_mma2      NUM_MMA_KV=2
forced_mma4      NUM_MMA_KV=4
forced_mma8      NUM_MMA_KV=8
baseline_after   FlashInfer auto
```

모든 run에서 correctness check를 수행합니다.

## 실행 방법

```bash
cd /root/capstone-yonsei/kv_tile_experiment
bash run_tile_kv.sh
```

특정 모델 그룹만 실행할 수도 있습니다.

```bash
bash run_tile_kv.sh llama
bash run_tile_kv.sh qwen
bash run_tile_kv.sh gemma
```

특정 모델만 실행할 수도 있습니다.

```bash
bash run_tile_kv.sh llama3_8b
bash run_tile_kv.sh llama3_70b
bash run_tile_kv.sh llama3_405b
bash run_tile_kv.sh qwen2.5_7b
bash run_tile_kv.sh qwen2.5_72b
bash run_tile_kv.sh gemma2_9b
bash run_tile_kv.sh gemma2_27b
```

백그라운드 실행 예시:

```bash
mkdir -p results/logs
nohup bash run_tile_kv.sh > results/logs/run_tile_kv_$(date +%Y%m%d).log 2>&1 &
```

## 결과 위치

새 실험 결과는 이 디렉터리 아래에 생성됩니다.

```text
kv_tile_experiment/results/data/tile_kv_results.csv
```

CSV의 주요 컬럼:

- `label`: 실험 라벨
- `seq_len`: sequence length
- `CTA_TILE_KV`, `NUM_MMA_KV`: 실제 실행된 FlashInfer 커널 tile 파라미터
- `ms`: 평균 latency
- `tflops`: 계산된 TFLOPS

라벨 형식:

```text
[baseline_before] llama3_8b
[baseline_after] llama3_8b
[experiment] llama3_8b_num_mma_kv_2
```

## 측정 설정

`bench_utils.py`에서 warmup과 반복 횟수를 지정합니다.

```python
WARMUP = 100
REPEAT = 100
```

각 `seq_len`마다 대략 다음 순서로 실행됩니다.

```text
tile 파라미터 추출 1회
warmup 100회
timed repeat 100회
correctness check 1회
```

## 가상환경 파일 패치 주의

`patch_prefill.py`는 아래 FlashInfer 헤더를 직접 수정합니다.

```text
/root/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh
```

`run_tile_kv.sh`에는 종료 시 자동 복원을 위한 trap이 들어 있습니다.
그래도 강제 종료 후 상태가 의심되면 수동으로 복원하세요.

```bash
cd /root/capstone-yonsei/kv_tile_experiment
python patch_prefill.py restore
rm -rf /root/.cache/flashinfer/
```

## 해석 기준

궁극적인 비교는 baseline 대비 forced tile의 speedup입니다.

```text
speedup = baseline_ms / experiment_ms
```

`speedup > 1`이면 forced 설정이 baseline보다 빠른 것입니다.

단, FlashInfer auto baseline과 forced 실험의 실제 커널 파라미터가 동일하면
latency도 거의 동일해야 합니다. 예를 들어 둘 다 아래와 같다면 큰 차이가 나면 안 됩니다.

```text
NUM_MMA_KV=2
CTA_TILE_KV=32
NUM_WARPS_KV=1
CTA_TILE_Q=128
```

동일 tile인데 큰 차이가 나면 tile 효과가 아니라 JIT cache, GPU 클럭/온도,
실행 순서, 외부 프로세스 간섭 같은 측정 조건을 먼저 의심해야 합니다.
