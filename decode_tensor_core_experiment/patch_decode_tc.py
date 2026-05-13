"""
Tensor-core decode NUM_MMA_KV patch script.

BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True) uses FlashInfer's
FA2 batch prefill module internally. This script patches prefill.cuh's
DISPATCH_NUM_MMA_KV blocks so tensor-core decode runs with a forced NUM_MMA_KV.

Usage:
    python patch_decode_tc.py apply 1
    python patch_decode_tc.py apply 2
    python patch_decode_tc.py apply 4
    python patch_decode_tc.py apply 8
    python patch_decode_tc.py restore
"""

from pathlib import Path
import shutil
import sys


ATTN_DIR_CANDIDATES = [
    Path("/root/capstone-yonsei/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/attention"),
    Path("/root/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/attention"),
    Path("/root/flashinfer/flashinfer/data/include/flashinfer/attention"),
    Path("/root/flashinfer/build/lib/flashinfer/data/include/flashinfer/attention"),
]

VALID_NUM_MMA_KV = {1, 2, 3, 4, 5, 6, 7, 8}
DISPATCH_TOKEN = "DISPATCH_NUM_MMA_KV("
ISINVALID_EXPR = (
    "NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256"
)


def _find_prefill_cuh() -> Path:
    for attn_dir in ATTN_DIR_CANDIDATES:
        path = attn_dir / "prefill.cuh"
        if path.exists():
            print(f"  prefill.cuh: {path}")
            return path
    searched = "\n".join(f"  - {p / 'prefill.cuh'}" for p in ATTN_DIR_CANDIDATES)
    raise FileNotFoundError(f"FlashInfer prefill.cuh를 찾지 못했습니다:\n{searched}")


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".decode_tc_tile_orig")


def _read(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines))


def _dispatch_count(lines: list[str]) -> int:
    return sum(1 for line in lines if DISPATCH_TOKEN in line)


def _ensure_backup(prefill_cuh: Path) -> None:
    backup = _backup_path(prefill_cuh)
    current = _read(prefill_cuh)
    current_dispatch_count = _dispatch_count(current)
    if current_dispatch_count == 3:
        shutil.copy2(prefill_cuh, backup)
        print(f"  backup refreshed: {backup}")
        return

    if not backup.exists():
        raise RuntimeError(
            f"현재 prefill.cuh는 원본 상태가 아닙니다. DISPATCH_NUM_MMA_KV={current_dispatch_count}. "
            "FlashInfer를 복원/재설치한 뒤 다시 실행하세요."
        )
    print(f"  using existing backup: {backup}")


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def _find_dispatch_close(lines: list[str], open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(lines)):
        depth += _brace_delta(lines[idx])
        if idx > open_idx and depth == 0:
            return idx
    raise RuntimeError(f"DISPATCH_NUM_MMA_KV 닫힘을 찾지 못함: line {open_idx + 1}")


def _patch_dispatch_blocks(lines: list[str], num_mma_kv: int) -> tuple[list[str], int]:
    open_indices = [idx for idx, line in enumerate(lines) if DISPATCH_TOKEN in line]
    if not open_indices:
        raise RuntimeError("DISPATCH_NUM_MMA_KV 블록을 찾지 못했습니다.")

    patched = 0
    for open_idx in reversed(open_indices):
        close_idx = _find_dispatch_close(lines, open_idx)
        open_line = lines[open_idx]
        close_line = lines[close_idx]

        if close_line.strip() not in {"})", "});"}:
            raise RuntimeError(
                f"예상한 매크로 닫힘 `}})`/`}});`가 아님: "
                f"line {close_idx + 1}: {close_line.rstrip()}"
            )

        open_pad = " " * (len(open_line) - len(open_line.lstrip()))
        close_pad = " " * (len(close_line) - len(close_line.lstrip()))
        lines[open_idx] = f"{open_pad}constexpr size_t NUM_MMA_KV = {num_mma_kv};\n"
        lines.insert(open_idx + 1, f"{open_pad}{{\n")
        lines[close_idx + 1] = f"{close_pad}}}\n"
        patched += 1
        print(f"  patched DISPATCH_NUM_MMA_KV: lines {open_idx + 1}-{close_idx + 1}")

    return lines, patched


def _patch_isinvalid(lines: list[str], num_mma_kv: int) -> None:
    if num_mma_kv != 8:
        return

    for idx, line in enumerate(lines):
        if ISINVALID_EXPR in line:
            lines[idx] = line.replace(">= 256", ">= 512")
            print(f"  IsInvalid: line {idx + 1} >= 256 -> >= 512")
            return
    print("  warning: IsInvalid threshold expression not found; skipped")


def apply(num_mma_kv: int) -> None:
    if num_mma_kv not in VALID_NUM_MMA_KV:
        raise ValueError(f"NUM_MMA_KV는 {sorted(VALID_NUM_MMA_KV)} 중 하나여야 합니다.")

    prefill_cuh = _find_prefill_cuh()
    _ensure_backup(prefill_cuh)
    backup = _backup_path(prefill_cuh)
    lines = _read(backup)
    if _dispatch_count(lines) == 0:
        raise RuntimeError(f"백업 파일에 DISPATCH_NUM_MMA_KV가 없습니다: {backup}")

    _patch_isinvalid(lines, num_mma_kv)
    lines, patched = _patch_dispatch_blocks(lines, num_mma_kv)
    _write(prefill_cuh, lines)
    print(f"  NUM_MMA_KV={num_mma_kv} 적용 완료 ({patched} blocks)")


def restore() -> None:
    prefill_cuh = _find_prefill_cuh()
    backup = _backup_path(prefill_cuh)
    if not backup.exists():
        print(f"  backup 없음, skip: {backup}")
        return
    shutil.copy2(backup, prefill_cuh)
    print(f"  원본 복원 완료: {prefill_cuh}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "apply":
        if len(sys.argv) < 3:
            print("값을 지정하세요: python patch_decode_tc.py apply <1..8>")
            sys.exit(1)
        apply(int(sys.argv[2]))
    elif cmd == "restore":
        restore()
    else:
        print(f"알 수 없는 명령: {cmd}  (apply | restore)")
        sys.exit(1)
