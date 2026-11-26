#!/usr/bin/env bash
# 공개 벤치마크 데이터셋(필수)과 평가할 모델 가중치(선택) 내려받기 — Hugging Face Hub
#
#   scripts/download.sh                      데이터셋 3종 → ./data
#   scripts/download.sh <HF 모델 저장소> ...  모델도 함께 → ./models/<저장소 이름>
#   QUANT_BITS=8 scripts/download.sh <저장소>  내려받은 모델을 mlx-vlm 으로 양자화한 사본 → ./models/<이름>-8bit (MLX)
#
# 데이터셋 라이선스: MMLU(cais/mmlu, MIT) · MMMU(MMMU/MMMU, Apache-2.0) · MMAU test-mini(gamma-lab-umd, CC BY-NC 4.0)
# 모델 가중치의 라이선스는 각 모델 저장소를 따른다.
set -euo pipefail
PY=${PYTHON:-python3}

"$PY" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download

jobs = [
    ("dataset", "cais/mmlu", ["all/test-*", "all/dev-*"], "data/mmlu"),
    ("dataset", "MMMU/MMMU", ["*/validation-*", "*/dev-*"], "data/mmmu"),
    ("dataset", "gamma-lab-umd/MMAU-test-mini", None, "data/mmau"),
]
jobs += [("model", repo, None, "models/" + repo.split("/")[-1]) for repo in sys.argv[1:]]
for kind, repo, pats, out in jobs:
    snapshot_download(repo_id=repo, repo_type=kind, allow_patterns=pats, local_dir=out, max_workers=8)
    print(f"[download] {repo} → {out}", flush=True)
PY

if [ -n "${QUANT_BITS:-}" ]; then
  for repo in "$@"; do
    name=${repo##*/}
    [ -f "models/$name-${QUANT_BITS}bit/config.json" ] || "$PY" -m mlx_vlm convert \
      --hf-path "models/$name" --mlx-path "models/$name-${QUANT_BITS}bit" -q --q-bits "$QUANT_BITS" --q-group-size 64
    echo "[quantize] models/$name-${QUANT_BITS}bit"
  done
fi
