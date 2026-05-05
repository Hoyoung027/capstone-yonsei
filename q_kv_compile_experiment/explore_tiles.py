"""
Q × KV 타일 전수 탐색 실험

Q, KV 각각 16~256 (16 배수) → 256개 조합 시도.
구조적으로 불가능한 조합은 GPU 없이 즉시 판별 후 스킵.
나머지는 패치 → JIT 캐시 삭제 → subprocess 실행 → 결과 기록.

사용법:
    python explore_tiles.py                    # llama3_8b, seq=1024
    python explore_tiles.py --seq_len 512
    python explore_tiles.py --resume           # 기존 결과 이어서

출력:
    results/data/q_kv_explore.csv
    results/logs/q{Q}_kv{KV}.log
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import patch_q_kv as patcher

RESULTS_CSV  = HERE / "results" / "data" / "q_kv_explore.csv"
LOGS_DIR     = HERE / "results" / "logs"
JIT_CACHE    = Path("/root/.cache/flashinfer")
TEST_SCRIPT  = HERE / "test_one_tile.py"

CSV_FIELDS = [
    "q_tile", "kv_tile",
    "num_warps_q", "num_mma_q", "num_warps_kv", "num_mma_kv",
    "theory_status", "cuda_status",
    "latency_ms", "output_shape", "error",
]

SUBPROCESS_TIMEOUT = 180  # seconds


def get_warps_q(q_tile: int) -> int:
    q_div = q_tile // 16
    if q_div % 4 == 0: return 4
    if q_div % 2 == 0: return 2
    return 1


def theory_check(q_tile: int, kv_tile: int) -> dict:
    """GPU 없이 이론적 feasibility 판단."""
    num_warps_q  = get_warps_q(q_tile)
    num_mma_q    = (q_tile // 16) // num_warps_q
    num_warps_kv = 4 // num_warps_q

    if kv_tile % (num_warps_kv * 16) != 0:
        return dict(num_warps_q=num_warps_q, num_mma_q=num_mma_q,
                    num_warps_kv=num_warps_kv, num_mma_kv=None,
                    theory_status="kv_not_factorable")

    num_mma_kv = kv_tile // (num_warps_kv * 16)

    # 공유 메모리 추정 (Q + KV smem 기본, head_dim=128, fp16)
    HEAD_DIM = 128
    smem = q_tile * HEAD_DIM * 2 + kv_tile * HEAD_DIM * 2 * 2
    MAX_SMEM = 99 * 1024  # RTX 3090

    if smem > MAX_SMEM:
        theory_status = "smem_warn"  # 경고만, 실제 실행은 시도
    else:
        theory_status = "candidate"

    return dict(num_warps_q=num_warps_q, num_mma_q=num_mma_q,
                num_warps_kv=num_warps_kv, num_mma_kv=num_mma_kv,
                theory_status=theory_status)

    return dict(num_warps_q=num_warps_q, num_mma_q=num_mma_q,
                num_warps_kv=num_warps_kv, num_mma_kv=num_mma_kv,
                theory_status="candidate")


def clear_jit_cache() -> None:
    if JIT_CACHE.exists():
        shutil.rmtree(JIT_CACHE)


def run_subprocess(q_tile: int, kv_tile: int, seq_len: int) -> dict:
    log_path = LOGS_DIR / f"q{q_tile}_kv{kv_tile}.log"
    cmd = [sys.executable, str(TEST_SCRIPT), str(q_tile), str(kv_tile), str(seq_len)]

    # venv bin이 PATH에 없으면 ninja를 못 찾으므로 명시적으로 추가
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    if venv_bin not in env.get("PATH", ""):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, env=env
        )
        log_path.write_text(proc.stdout + proc.stderr)

        # stdout 마지막 JSON 줄 파싱
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        return {"status": "no_json_output", "error": proc.stderr[-300:]}

    except subprocess.TimeoutExpired:
        log_path.write_text(f"TIMEOUT after {SUBPROCESS_TIMEOUT}s\n")
        return {"status": "timeout", "error": f"exceeded {SUBPROCESS_TIMEOUT}s"}


def load_done(csv_path: Path) -> set[tuple[int, int]]:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            done.add((int(row["q_tile"]), int(row["kv_tile"])))
    return done


def write_row(writer, row: dict) -> None:
    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--resume", action="store_true",
                        help="기존 results CSV 이어서 실행")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # helper 함수 일반화 (최초 1회)
    print("=== helper 함수 초기화 ===")
    patcher.init_helpers()
    patcher._ensure_batch_backup()

    # 기존 결과 로드 (resume 모드)
    done = load_done(RESULTS_CSV) if args.resume else set()
    if done:
        print(f"  이어서 실행: {len(done)}개 완료")

    # 전체 조합 생성
    combos = [(q, kv) for q in range(16, 272, 16) for kv in range(16, 272, 16)]
    total  = len(combos)

    csv_mode = "a" if (args.resume and RESULTS_CSV.exists()) else "w"
    with open(RESULTS_CSV, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if csv_mode == "w":
            writer.writeheader()

        for idx, (q_tile, kv_tile) in enumerate(combos, 1):
            if (q_tile, kv_tile) in done:
                continue

            theory = theory_check(q_tile, kv_tile)
            row = dict(q_tile=q_tile, kv_tile=kv_tile, **theory)

            prefix = f"[{idx:3d}/{total}] Q={q_tile:3d} KV={kv_tile:3d}"

            if theory["theory_status"] == "kv_not_factorable":
                row.update(cuda_status="", latency_ms="", output_shape="", error="")
                write_row(writer, row)
                f.flush()
                print(f"{prefix} → {theory['theory_status']}")
                continue

            # 패치 → 실행
            print(f"{prefix} → 실험 중...", end=" ", flush=True)
            t_start = time.time()

            try:
                patcher.apply(q_tile, theory["num_mma_kv"])
                clear_jit_cache()
                result = run_subprocess(q_tile, kv_tile, args.seq_len)
            except Exception as e:
                result = {"status": "patch_error", "error": str(e)[:300]}
            finally:
                patcher.restore()

            elapsed = time.time() - t_start
            row["cuda_status"]  = result.get("status", "")
            row["latency_ms"]   = result.get("latency_ms", "")
            row["output_shape"] = str(result.get("output_shape", ""))
            row["error"]        = result.get("error", "")[:200]

            write_row(writer, row)
            f.flush()
            print(f"{result.get('status')} ({elapsed:.0f}s)")

    print(f"\n완료: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
