"""
Q × KV 타일 조합 실험을 위한 패치 스크립트

사용법:
    python patch_q_kv.py init_helpers          # helper 함수 일반화 (최초 1회)
    python patch_q_kv.py apply 64 4            # CTA_TILE_Q=64, NUM_MMA_KV=4 강제
    python patch_q_kv.py restore               # 원본 복원
    python patch_q_kv.py status                # 현재 패치 상태 확인

수정 파일:
    prefill.cuh   - get_num_warps_q / get_num_mma_q 일반화 + NUM_MMA_KV 강제
    batch_prefill.cu - DISPATCH_CTA_TILE_Q → 직접 호출로 교체
"""

from pathlib import Path
import shutil
import sys

PREFILL_CUH = Path(
    "/root/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh"
)
BATCH_CU = Path(
    "/root/venv/lib/python3.10/site-packages/flashinfer/data/csrc/batch_prefill.cu"
)

PREFILL_BACKUP = PREFILL_CUH.with_suffix(".cuh.orig")
BATCH_BACKUP   = BATCH_CU.with_suffix(".cu.orig")

# ── helper 함수 교체 텍스트 ────────────────────────────────────
OLD_WARPS_Q = (
    "constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {\n"
    "  if (cta_tile_q > 16) {\n"
    "    return 4;\n"
    "  } else {\n"
    "    return 1;\n"
    "  }\n"
    "}"
)
NEW_WARPS_Q = (
    "constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {\n"
    "  uint32_t q_div = cta_tile_q / 16;\n"
    "  if (q_div % 4 == 0) return 4;\n"
    "  if (q_div % 2 == 0) return 2;\n"
    "  return 1;\n"
    "}"
)

OLD_MMA_Q = (
    "constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {\n"
    "  if (cta_tile_q > 64) {\n"
    "    return 2;\n"
    "  } else {\n"
    "    return 1;\n"
    "  }\n"
    "}"
)
NEW_MMA_Q = (
    "constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {\n"
    "  return (cta_tile_q / 16) / get_num_warps_q(cta_tile_q);\n"
    "}"
)

ISINVALID_EXPR = (
    "NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256"
)
DISPATCH_MMA_KV_TOKEN = "DISPATCH_NUM_MMA_KV("
DISPATCH_CTA_Q_TOKEN  = "DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q"
PAGED_KERNEL_TOKEN    = "BatchPrefillWithPagedKVCacheDispatched<"


def _get_num_warps_q(q_tile: int) -> int:
    q_div = q_tile // 16
    if q_div % 4 == 0: return 4
    if q_div % 2 == 0: return 2
    return 1


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def _find_close(lines: list[str], open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(lines)):
        depth += _brace_delta(lines[idx])
        if idx > open_idx and depth == 0:
            return idx
    raise RuntimeError(f"닫힘 브레이스 못 찾음: line {open_idx + 1}")


# ── prefill.cuh 패치 ──────────────────────────────────────────

def _ensure_prefill_backup() -> None:
    if not PREFILL_CUH.exists():
        raise FileNotFoundError(f"prefill.cuh 없음: {PREFILL_CUH}")
    text = PREFILL_CUH.read_text()
    if OLD_WARPS_Q in text or NEW_WARPS_Q in text:
        if not PREFILL_BACKUP.exists():
            shutil.copy2(PREFILL_CUH, PREFILL_BACKUP)
            print(f"  prefill.cuh 백업: {PREFILL_BACKUP}")
        else:
            print(f"  prefill.cuh 기존 백업 사용")
    else:
        raise RuntimeError("prefill.cuh에서 helper 함수를 찾지 못했습니다.")


def _ensure_batch_backup() -> None:
    if not BATCH_CU.exists():
        raise FileNotFoundError(f"batch_prefill.cu 없음: {BATCH_CU}")
    if not BATCH_BACKUP.exists():
        shutil.copy2(BATCH_CU, BATCH_BACKUP)
        print(f"  batch_prefill.cu 백업: {BATCH_BACKUP}")
    else:
        print(f"  batch_prefill.cu 기존 백업 사용")


def init_helpers() -> None:
    """get_num_warps_q / get_num_mma_q를 16 배수 전체 지원으로 일반화 (최초 1회)."""
    _ensure_prefill_backup()
    text = PREFILL_CUH.read_text()

    if NEW_WARPS_Q in text:
        print("  helper 함수 이미 일반화되어 있음")
        return

    if OLD_WARPS_Q not in text:
        raise RuntimeError("get_num_warps_q 원본을 찾지 못했습니다.")
    if OLD_MMA_Q not in text:
        raise RuntimeError("get_num_mma_q 원본을 찾지 못했습니다.")

    text = text.replace(OLD_WARPS_Q, NEW_WARPS_Q)
    text = text.replace(OLD_MMA_Q, NEW_MMA_Q)
    PREFILL_CUH.write_text(text)
    print("  helper 함수 일반화 완료")


def _patch_prefill_mma_kv(lines: list[str], num_mma_kv: int) -> list[str]:
    open_indices = [i for i, l in enumerate(lines) if DISPATCH_MMA_KV_TOKEN in l]
    if not open_indices:
        raise RuntimeError("DISPATCH_NUM_MMA_KV 블록 없음 (이미 패치된 파일?)")

    for open_idx in reversed(open_indices):
        close_idx = _find_close(lines, open_idx)
        pad = " " * (len(lines[open_idx]) - len(lines[open_idx].lstrip()))
        lines[open_idx] = f"{pad}constexpr size_t NUM_MMA_KV = {num_mma_kv};\n"
        lines.insert(open_idx + 1, f"{pad}{{\n")
        lines[close_idx + 1] = f"{pad}}}\n"
        print(f"  NUM_MMA_KV={num_mma_kv} 패치: lines {open_idx+1}-{close_idx+1}")

    return lines


def _patch_prefill_isinvalid(lines: list[str], num_mma_kv: int, q_tile: int) -> list[str]:
    num_mma_q = (q_tile // 16) // _get_num_warps_q(q_tile)
    needs_bypass = (num_mma_q >= 4) or (num_mma_kv >= 8)
    if not needs_bypass:
        return lines

    for i, line in enumerate(lines):
        if ISINVALID_EXPR in line:
            lines[i] = line.replace(">= 256", ">= 512")
            print(f"  IsInvalid 우회: line {i+1} >= 256 -> >= 512")
            return lines

    print("  warning: IsInvalid 표현식 못 찾음")
    return lines


def _patch_batch_cu(lines: list[str], q_tile: int) -> list[str]:
    """batch_prefill.cu의 Paged 버전 DISPATCH_CTA_TILE_Q를 직접 호출로 교체."""
    # DISPATCH_CTA_TILE_Q 블록 두 개 중 Paged 버전만 패치
    open_indices = [i for i, l in enumerate(lines) if DISPATCH_CTA_Q_TOKEN in l]
    if not open_indices:
        raise RuntimeError("DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q 블록 없음")

    patched = 0
    for open_idx in reversed(open_indices):
        close_idx = _find_close(lines, open_idx)
        block_body = "".join(lines[open_idx:close_idx + 1])
        if PAGED_KERNEL_TOKEN not in block_body:
            continue  # Ragged 버전은 건너뜀

        pad = " " * (len(lines[open_idx]) - len(lines[open_idx].lstrip()))
        lines[open_idx] = f"{pad}constexpr uint32_t CTA_TILE_Q = {q_tile};\n"
        lines.insert(open_idx + 1, f"{pad}{{\n")
        lines[close_idx + 1] = lines[close_idx + 1].replace("});", "}")
        patched += 1
        print(f"  CTA_TILE_Q={q_tile} 패치 (Paged): lines {open_idx+1}-{close_idx+1}")

    if patched == 0:
        raise RuntimeError("Paged 버전 DISPATCH_CTA_TILE_Q 블록을 찾지 못했습니다.")
    return lines


def apply(q_tile: int, num_mma_kv: int) -> None:
    """CTA_TILE_Q와 NUM_MMA_KV를 강제 지정."""
    _ensure_prefill_backup()
    _ensure_batch_backup()

    # 백업에서 시작해서 helper 일반화 + NUM_MMA_KV 패치를 한 번에 적용
    backup_text = PREFILL_BACKUP.read_text()
    if OLD_WARPS_Q in backup_text:
        backup_text = backup_text.replace(OLD_WARPS_Q, NEW_WARPS_Q)
        backup_text = backup_text.replace(OLD_MMA_Q, NEW_MMA_Q)
    elif NEW_WARPS_Q not in backup_text:
        raise RuntimeError("백업 파일에서 helper 함수를 찾지 못했습니다.")
    prefill_lines = backup_text.splitlines(keepends=True)

    prefill_lines = _patch_prefill_isinvalid(prefill_lines, num_mma_kv, q_tile)
    prefill_lines = _patch_prefill_mma_kv(prefill_lines, num_mma_kv)
    PREFILL_CUH.write_text("".join(prefill_lines))

    # batch_prefill.cu: CTA_TILE_Q 강제
    batch_lines = BATCH_BACKUP.read_text().splitlines(keepends=True)
    batch_lines = _patch_batch_cu(batch_lines, q_tile)
    BATCH_CU.write_text("".join(batch_lines))

    print(f"  적용 완료: Q={q_tile}, NUM_MMA_KV={num_mma_kv}")


def restore() -> None:
    """두 파일 모두 원본으로 복원."""
    restored = False
    for src, dst in [(PREFILL_BACKUP, PREFILL_CUH), (BATCH_BACKUP, BATCH_CU)]:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  복원: {dst.name}")
            restored = True
    if not restored:
        print("  백업 파일 없음")


def status() -> None:
    prefill_text = PREFILL_CUH.read_text()
    has_helpers = NEW_WARPS_Q in prefill_text
    has_mma_kv  = DISPATCH_MMA_KV_TOKEN not in prefill_text

    batch_text  = BATCH_CU.read_text()
    has_q_patch = DISPATCH_CTA_Q_TOKEN not in batch_text

    print(f"  prefill.cuh helper 일반화: {'✓' if has_helpers else '✗'}")
    print(f"  prefill.cuh NUM_MMA_KV 강제: {'✓' if has_mma_kv else '✗'}")
    print(f"  batch_prefill.cu CTA_TILE_Q 강제: {'✓' if has_q_patch else '✗'}")
    print(f"  prefill.cuh 백업: {'✓' if PREFILL_BACKUP.exists() else '✗'}")
    print(f"  batch_prefill.cu 백업: {'✓' if BATCH_BACKUP.exists() else '✗'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "init_helpers":
        init_helpers()
    elif cmd == "apply":
        if len(sys.argv) < 4:
            print("사용법: patch_q_kv.py apply <q_tile> <num_mma_kv>")
            sys.exit(1)
        apply(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "restore":
        restore()
    elif cmd == "status":
        status()
    else:
        print(f"알 수 없는 명령: {cmd}")
        sys.exit(1)
