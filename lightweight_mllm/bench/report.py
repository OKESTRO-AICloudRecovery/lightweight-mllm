"""결과 요약표: 모델 × 벤치마크 정확도 (선택: 벤치마크별 목표치와 비교).

  python -m lightweight_mllm.bench.report [--tag full] [--out results] [--target mmlu=50 --target mmmu=52 ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import unicodedata

from .data import BENCH_LABEL


def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s, width: int, right: bool = False) -> str:
    s = str(s)
    gap = max(width - _w(s), 0)
    return " " * gap + s if right else s + " " * gap


def table(heads, rows) -> str:
    widths = [max(_w(r[i]) for r in [heads] + rows) for i in range(len(heads))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda cells: "| " + " | ".join(_pad(c, w, i > 1) for i, (c, w) in enumerate(zip(cells, widths))) + " |"  # noqa: E731
    return "\n".join([line, fmt(heads), line.replace("-", "="), *[fmt(r) for r in rows], line])


def collect(out: str, tag: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, tag, "*", "*.summary.json"))):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out", default="results")
    ap.add_argument("--target", action="append", default=[], help="벤치마크=목표 정확도(%%), 예: mmmu=52")
    args = ap.parse_args(argv)
    targets = {k: float(v) for k, v in (t.split("=", 1) for t in args.target)}

    summaries = collect(args.out, args.tag)
    if not summaries:
        print(f"결과가 없습니다: {os.path.join(args.out, args.tag)}")
        return 1
    heads = ["모델", "벤치마크", "문항 수", "정답 수", "정확도", "파싱 실패"] + (["목표", "결과"] if targets else [])
    rows = []
    for s in summaries:
        acc = round(s["accuracy"] * 100, 1)
        r = [s["model"], BENCH_LABEL.get(s["bench"], s["bench"]), f"{s['n']:,}", f"{round(s['accuracy'] * s['n']):,}",
             f"{acc:.1f}%", f"{s['parse_fail']:,}"]
        if targets:
            t = targets.get(s["bench"])
            r += [f"{t:g}% 이상" if t is not None else "-", ("충족" if acc >= t else "미충족") if t is not None else "-"]
        rows.append(r)
    print(table(heads, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
