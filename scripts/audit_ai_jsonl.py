#!/usr/bin/env python3
"""
Audit AI-generated arxiv-style JSONL: split into clean vs rejected, merge regenerated patches.

  python scripts/audit_ai_jsonl.py audit [--input PATH] [--output-clean PATH] [--output-rejects PATH]
  python scripts/audit_ai_jsonl.py merge --base PATH --patch PATH --output PATH [--validate]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from ai_jsonl_quality import (  # noqa: E402
    parse_ai_index_from_id,
    validate_record_dict,
)


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "dataset"


def audit_file(
    input_path: Path,
    output_clean: Path,
    output_rejects: Path,
    *,
    require_categories: bool,
) -> Tuple[int, int]:
    n_keep = 0
    n_reject = 0
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    output_rejects.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_clean.open(
        "w", encoding="utf-8"
    ) as fc, output_rejects.open("w", encoding="utf-8") as fr:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                fr.write(
                    json.dumps(
                        {"_raw": line[:500], "reasons": ["json_decode_error"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_reject += 1
                continue
            ok, reasons = validate_record_dict(obj, require_categories=require_categories)
            if ok:
                fc.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_keep += 1
            else:
                out = dict(obj)
                out["reasons"] = reasons
                fr.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_reject += 1
    return n_keep, n_reject


def merge_patches(
    base_path: Path,
    patch_path: Path,
    output_path: Path,
    *,
    validate: bool,
    require_categories: bool,
    still_bad_path: Path | None,
) -> int:
    base_lines: List[str] = []
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base_lines.append(line.rstrip("\n"))

    patches: Dict[int, str] = {}
    with patch_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Warning: skip invalid patch line", file=sys.stderr)
                continue
            idx = parse_ai_index_from_id(obj.get("id"))
            if idx is None:
                print("Warning: skip patch line without ai-N id", file=sys.stderr)
                continue
            if idx < 0 or idx >= len(base_lines):
                print("Warning: patch id ai-%d out of base range" % idx, file=sys.stderr)
                continue
            patches[idx] = line

    out_lines = list(base_lines)
    for idx, pline in patches.items():
        out_lines[idx] = pline

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for ln in out_lines:
            fout.write(ln + "\n")

    if validate:
        bad: List[Dict[str, Any]] = []
        for i, ln in enumerate(out_lines):
            obj = json.loads(ln)
            ok, reasons = validate_record_dict(obj, require_categories=require_categories)
            if not ok:
                bad.append({"line_index": i, "id": obj.get("id"), "reasons": reasons})
        if bad and still_bad_path:
            still_bad_path.parent.mkdir(parents=True, exist_ok=True)
            with still_bad_path.open("w", encoding="utf-8") as fb:
                for b in bad:
                    fb.write(json.dumps(b, ensure_ascii=False) + "\n")
        if bad:
            print(
                "Validation: %d line(s) still fail checks (see %s)"
                % (len(bad), still_bad_path or "(no --still-bad output)"),
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    root = argparse.ArgumentParser(description="Audit / merge AI JSONL dataset rows")
    sub = root.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="Split input into clean and rejected JSONL")
    p_audit.add_argument(
        "--input",
        type=Path,
        default=_default_dataset_dir() / "ai_abstracts_20k.jsonl",
        help="Input JSONL (default: dataset/ai_abstracts_20k.jsonl)",
    )
    p_audit.add_argument(
        "--output-clean",
        type=Path,
        default=None,
        help="Output path for passing records (default: <input>.cleaned.jsonl)",
    )
    p_audit.add_argument(
        "--output-rejects",
        type=Path,
        default=None,
        help="Output path for failing records (default: <input>.rejected.jsonl)",
    )
    p_audit.add_argument(
        "--no-require-categories",
        action="store_true",
        help="Do not require non-empty categories",
    )

    p_merge = sub.add_parser("merge", help="Apply patch lines (by ai-N id) onto base JSONL")
    p_merge.add_argument("--base", type=Path, required=True, help="Original full JSONL")
    p_merge.add_argument(
        "--patch",
        type=Path,
        required=True,
        help="JSONL of replacement full records (id must be ai-<index>)",
    )
    p_merge.add_argument("--output", type=Path, required=True, help="Merged output JSONL")
    p_merge.add_argument(
        "--validate",
        action="store_true",
        help="After merge, validate every line; exit non-zero if any fail",
    )
    p_merge.add_argument(
        "--no-require-categories",
        action="store_true",
        help="When validating, do not require categories",
    )
    p_merge.add_argument(
        "--still-bad",
        type=Path,
        default=None,
        help="Write JSON lines of remaining failures when using --validate",
    )

    args = root.parse_args()
    if args.cmd == "audit":
        inp = args.input
        oc = args.output_clean or inp.with_name(inp.stem + ".cleaned.jsonl")
        orj = args.output_rejects or inp.with_name(inp.stem + ".rejected.jsonl")
        k, r = audit_file(inp, oc, orj, require_categories=not args.no_require_categories)
        print("Kept %d, rejected %d -> %s , %s" % (k, r, oc, orj))
        return 0

    if args.cmd == "merge":
        rc = merge_patches(
            args.base,
            args.patch,
            args.output,
            validate=args.validate,
            require_categories=not args.no_require_categories,
            still_bad_path=args.still_bad,
        )
        print("Wrote merged file to %s" % args.output)
        return rc

    return 1


if __name__ == "__main__":
    sys.exit(main())
