"""
prefill.cuh 자동 패치 스크립트

사용법:
    python patch_prefill.py apply 1
    python patch_prefill.py apply 2
    python patch_prefill.py apply 4
    python patch_prefill.py apply 8   # IsInvalid 우회 포함
    python patch_prefill.py restore

라인 번호가 아니라 DISPATCH_NUM_MMA_KV 매크로 블록 자체를 찾아 수정한다.
FlashInfer를 재설치한 뒤 실행하면 새 prefill.cuh를 백업으로 갱신한다.
"""

from pathlib import Path
import shutil
import sys


PREFILL_CUH = Path(
    "/root/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh"
)
BACKUP = PREFILL_CUH.with_suffix(PREFILL_CUH.suffix + ".orig")

VALID_NUM_MMA_KV = {1, 2, 3, 4, 5, 6, 7, 8}
DISPATCH_TOKEN = "DISPATCH_NUM_MMA_KV("
ISINVALID_EXPR = (
    "NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256"
)


def _read(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines))


def _has_dispatch(lines: list[str]) -> bool:
    return any(DISPATCH_TOKEN in line for line in lines)


def _dispatch_count(lines: list[str]) -> int:
    return sum(1 for line in lines if DISPATCH_TOKEN in line)


def _ensure_backup() -> None:
    if not PREFILL_CUH.exists():
        raise FileNotFoundError(f"prefill.cuh 없음: {PREFILL_CUH}")

    current = _read(PREFILL_CUH)
    current_dispatch_count = _dispatch_count(current)
    if current_dispatch_count == 3:
        shutil.copy2(PREFILL_CUH, BACKUP)
        print(f"  backup refreshed: {BACKUP}")
        return

    if not BACKUP.exists():
        raise RuntimeError(
            f"현재 prefill.cuh는 원본 상태가 아닙니다. DISPATCH_NUM_MMA_KV={current_dispatch_count}. "
            "flashinfer를 재설치한 뒤 다시 실행하세요."
        )

    print(f"  using existing backup: {BACKUP}")


def _brace_delta(line: str) -> int:
    # prefill.cuh의 대상 블록에는 raw string/comment trick이 없어서 단순 카운트로 충분하다.
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
        raise RuntimeError("DISPATCH_NUM_MMA_KV 블록을 찾지 못했습니다. 이미 패치된 파일인지 확인하세요.")

    patched = 0
    for open_idx in reversed(open_indices):
        close_idx = _find_dispatch_close(lines, open_idx)
        open_line = lines[open_idx]
        close_line = lines[close_idx]

        close_stripped = close_line.strip()
        if close_stripped not in {"})", "});"}:
            raise RuntimeError(
                f"예상한 매크로 닫힘 `}})`/`}});`가 아님: "
                f"line {close_idx + 1}: {close_line.rstrip()}"
            )

        open_indent = len(open_line) - len(open_line.lstrip())
        close_indent = len(close_line) - len(close_line.lstrip())
        open_pad = " " * open_indent
        close_pad = " " * close_indent

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

    _ensure_backup()
    lines = _read(BACKUP)
    if not _has_dispatch(lines):
        raise RuntimeError(f"백업 파일에 DISPATCH_NUM_MMA_KV가 없습니다: {BACKUP}")

    _patch_isinvalid(lines, num_mma_kv)
    lines, patched = _patch_dispatch_blocks(lines, num_mma_kv)
    _write(PREFILL_CUH, lines)
    print(f"  NUM_MMA_KV={num_mma_kv} 적용 완료 ({patched} blocks)")


def restore() -> None:
    if not BACKUP.exists():
        print("  백업 파일 없음. flashinfer 재설치 후 apply를 먼저 실행하세요.")
        return
    shutil.copy2(BACKUP, PREFILL_CUH)
    print(f"  원본 복원 완료: {PREFILL_CUH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "apply":
        if len(sys.argv) < 3:
            print("NUM_MMA_KV 값을 지정하세요: python patch_prefill.py apply <1|2|4|8>")
            sys.exit(1)
        apply(int(sys.argv[2]))
    elif cmd == "restore":
        restore()
    else:
        print(f"알 수 없는 명령: {cmd}  (apply | restore)")
        sys.exit(1)
