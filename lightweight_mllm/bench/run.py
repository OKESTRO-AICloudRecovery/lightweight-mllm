"""벤치마크 실행.

  python -m lightweight_mllm.bench.run --model qwen25-omni-3b --bench mmlu [--limit 50 | --pilot 5] [--tag full]
  python -m lightweight_mllm.bench.run --model my-vlm --model-path ./models/x --backend mlx --bench mmmu

결과: <out>/<tag>/<model>/<bench>.jsonl (문항별, 중단 후 이어 달리기 가능) + <bench>.summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

from . import backends
from .data import BENCH_LABEL, GROUP_KEY, LOADERS, pilot_subset
from .parse import judge

# 평가 방식 표식: 방식이 바뀌면 이전 결과를 이어 쓰지 않는다
PROTOCOL = {"mmlu": "mmlu-0shot-free-v1", "mmmu": "mmmu-vlmevalkit-free-v1", "mmau": "mmau-free-v1"}


def summarize(rows, bench):
    key = GROUP_KEY[bench]
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        g = groups[r["group"]]
        g[0] += r["correct"]
        g[1] += 1
    n = len(rows)
    return {
        "bench": bench, "label": BENCH_LABEL[bench], "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "by_" + key: {k: {"n": v[1], "accuracy": v[0] / v[1]} for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))},
        "parse_fail": sum(r["parse"] == "none" for r in rows),
        "sec_per_item": sum(r["sec"] for r in rows) / n if n else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"모델 이름 (등록: {', '.join(backends.MODELS)})")
    ap.add_argument("--model-path", default="", help="가중치 경로 (기본: 등록된 경로)")
    ap.add_argument("--backend", default="", help="백엔드 종류 (등록되지 않은 모델일 때 필수: mlx | torch-omni)")
    ap.add_argument("--bench", required=True, choices=list(LOADERS))
    ap.add_argument("--pilot", type=int, default=0, help="과목/과제별 표본 수 (0 = 전체)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N문항만 (0 = 제한 없음)")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    items = LOADERS[args.bench]()
    if args.pilot:
        items = pilot_subset(items, args.bench, args.pilot)
    if args.limit:
        items = items[: args.limit]
    out_dir = os.path.join(args.out, args.tag, args.model)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.bench}.jsonl")
    proto = PROTOCOL[args.bench]
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("proto") == proto:
                    done[r["id"]] = r
    todo = [it for it in items if it.id not in done]
    print(f"[{args.model} · {BENCH_LABEL[args.bench]}] 문항 {len(items)} (완료 {len(done)}, 남음 {len(todo)})", flush=True)

    if todo:
        t0 = time.time()
        backend = backends.load_backend(args.model, args.model_path, args.backend)
        print(f"  모델 로딩 {time.time() - t0:.1f}s", flush=True)
        rng = random.Random(42)
        key = GROUP_KEY[args.bench]
        t_start = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for i, it in enumerate(todo, 1):
                t = time.time()
                try:
                    resp, err = backend(it), None
                except Exception as e:  # 문항 단위 실패는 오답으로 기록하고 계속
                    resp, err = "", f"{type(e).__name__}: {e}"[:300]
                j = judge(it, resp, rng)
                row = {"id": it.id, "proto": proto, "group": it.meta.get(key), "gold": it.gold, "pred": j["pred"],
                       "correct": bool(j["correct"]), "parse": j["parse"], "sec": round(time.time() - t, 3),
                       "response": resp}
                if err:
                    row["error"] = err
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[it.id] = row
                if i % 25 == 0 or i == len(todo):
                    rows = [done[x.id] for x in items if x.id in done]
                    acc = sum(r["correct"] for r in rows) / len(rows)
                    print(f"  [{len(rows):5d}/{len(items)}] 정확도 {acc * 100:5.1f}%  "
                          f"{(time.time() - t_start) / i:.2f}s/문항", flush=True)

    rows = [done[x.id] for x in items if x.id in done]
    s = summarize(rows, args.bench)
    s.update({"model": args.model, "tag": args.tag, "pilot_per_group": args.pilot, "limit": args.limit, "protocol": proto})
    with open(os.path.join(out_dir, f"{args.bench}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"  ▶ {BENCH_LABEL[args.bench]} 정확도 {s['accuracy'] * 100:.1f}% (n={s['n']}, 파싱 실패 {s['parse_fail']}, "
          f"{s['sec_per_item']:.2f}s/문항)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
