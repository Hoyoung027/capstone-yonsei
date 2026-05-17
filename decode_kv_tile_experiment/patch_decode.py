"""
decode.cuh/scheduler.cuh 자동 패치 스크립트.

사용법:
    python patch_decode.py apply 1
    python patch_decode.py apply 2
    ...
    python patch_decode.py apply 8
    python patch_decode.py restore

FlashInfer JIT가 참조하는 가상환경 내부 헤더를 수정한다.
decode.cuh와 scheduler.cuh의 tile_size_per_bdx 산식을 같은 값으로 맞춘다.
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

VALID_TILE_SIZE_PER_BDX = set(range(1, 9))
ORIGINAL_EXPR = (
    "constexpr uint32_t tile_size_per_bdx = "
    "GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;"
)
SINGLE_DECODE_EXPR = (
    "constexpr uint32_t tile_size_per_bdx = "
    "GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;"
)
PATCH_MARKER = "decode_kv_tile_experiment forced"


def _find_attention_dir() -> Path:
    for path in ATTN_DIR_CANDIDATES:
        if (path / "decode.cuh").exists() and (path / "scheduler.cuh").exists():
            return path
    searched = "\n".join(f"  - {p}" for p in ATTN_DIR_CANDIDATES)
    raise FileNotFoundError(f"FlashInfer attention include dir를 찾지 못했습니다:\n{searched}")


def _targets() -> list[Path]:
    attn_dir = _find_attention_dir()
    print(f"  attention include dir: {attn_dir}")
    return [
        attn_dir / "decode.cuh",
        attn_dir / "scheduler.cuh",
    ]


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".decode_tile_orig")


def _read(path: Path) -> str:
    return path.read_text()


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def _ensure_backup(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"헤더 파일 없음: {path}")

    backup = _backup_path(path)
    text = _read(path)
    if PATCH_MARKER not in text:
        shutil.copy2(path, backup)
        print(f"  backup refreshed: {backup}")
        return

    if not backup.exists():
        raise RuntimeError(f"패치된 파일인데 백업이 없습니다: {backup}")

    print(f"  using existing backup: {backup}")


def _patch_text(path: Path, text: str, tile_size_per_bdx: int) -> tuple[str, int]:
    forced = (
        f"constexpr uint32_t tile_size_per_bdx = {tile_size_per_bdx}U; "
        f"// {PATCH_MARKER}\n"
    )
    count = 0

    if path.name == "decode.cuh":
        count += text.count(ORIGINAL_EXPR)
        text = text.replace(ORIGINAL_EXPR, forced.rstrip())
        count += text.count(SINGLE_DECODE_EXPR)
        text = text.replace(SINGLE_DECODE_EXPR, forced.rstrip())
    else:
        count += text.count(ORIGINAL_EXPR)
        text = text.replace(ORIGINAL_EXPR, forced.rstrip())

    return text, count


def apply(tile_size_per_bdx: int) -> None:
    if tile_size_per_bdx not in VALID_TILE_SIZE_PER_BDX:
        raise ValueError(
            f"tile_size_per_bdx는 {sorted(VALID_TILE_SIZE_PER_BDX)} 중 하나여야 합니다."
        )

    for path in _targets():
        _ensure_backup(path)
        backup = _backup_path(path)
        text = _read(backup)
        patched, count = _patch_text(path, text, tile_size_per_bdx)
        if count == 0:
            raise RuntimeError(f"패치 대상 tile_size_per_bdx 식을 찾지 못했습니다: {path}")
        _write(path, patched)
        print(f"  {path.name}: tile_size_per_bdx={tile_size_per_bdx} 적용 ({count} sites)")


def restore() -> None:
    for path in _targets():
        backup = _backup_path(path)
        if not backup.exists():
            print(f"  backup 없음, skip: {backup}")
            continue
        shutil.copy2(backup, path)
        print(f"  원본 복원 완료: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "apply":
        if len(sys.argv) < 3:
            print("값을 지정하세요: python patch_decode.py apply <1..8>")
            sys.exit(1)
        apply(int(sys.argv[2]))
    elif cmd == "restore":
        restore()
    else:
        print(f"알 수 없는 명령: {cmd}  (apply | restore)")
        sys.exit(1)
