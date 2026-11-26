"""모델 파라미터 수를 구성 요소별로 센다 (safetensors 헤더만 읽음, 모델을 메모리에 올리지 않음).

'3B 이하' 같은 경량 기준은 추론을 수행하는 언어모델 디코더(Transformer 블록) 파라미터 수로 판단한다.
임베딩·출력층, 비전·오디오 인코더, 음성 생성부는 따로 표시한다.

  python -m lightweight_mllm.bench.params <가중치 디렉터리> [--family qwen25-omni|qwen-vl|qwen-vl-mlx|gemma4]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
from collections import OrderedDict
from typing import Optional

DECODER = "언어모델 디코더"
LIMIT = 3.0e9

# 가중치 이름(부분 문자열) → 구성 요소. 먼저 맞는 규칙이 이긴다.
RULES = {
    "qwen25-omni": [
        ("thinker.audio_tower", "오디오 인코더"),
        ("thinker.visual", "비전 인코더"),
        ("thinker.model.embed_tokens", "입력 임베딩"),
        ("thinker.lm_head", "출력층(lm_head)"),
        ("thinker.model", DECODER),
        ("talker", "음성 생성부"),
        ("token2wav", "음성 합성부"),
    ],
    "qwen-vl": [
        ("visual", "비전 인코더"),
        ("embed_tokens", "입력 임베딩"),
        ("lm_head", "출력층(lm_head)"),
        ("mtp.", "추가 예측층(MTP)"),
        ("language_model", DECODER),
        ("model.layers", DECODER),
    ],
    "qwen-vl-mlx": [  # mlx-vlm 변환본 (가중치 이름이 다르다)
        ("vision_tower", "비전 인코더"),
        ("embed_tokens", "입력 임베딩"),
        ("lm_head", "출력층(lm_head)"),
        ("language_model", DECODER),
    ],
    "gemma4": [
        ("per_layer", "층별 임베딩(PLE)"),
        ("audio", "오디오 인코더"),
        ("vision", "비전 인코더"),
        ("embed_tokens", "입력 임베딩"),
        ("language_model", DECODER),
    ],
}


def _header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _config(path: str) -> dict:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def quant_bits(path: str) -> Optional[int]:
    return (_config(path).get("quantization") or {}).get("bits")


def guess_family(path: str) -> str:
    cfg = _config(path)
    mt = str(cfg.get("model_type", "")).lower()
    if "omni" in mt:
        return "qwen25-omni"
    if "gemma" in mt:
        return "gemma4"
    return "qwen-vl-mlx" if cfg.get("quantization") else "qwen-vl"


def count(path: str, family: str = "") -> "OrderedDict[str, int]":
    rules = RULES[family or guess_family(path)]
    parts: "OrderedDict[str, int]" = OrderedDict((label, 0) for _, label in rules)
    parts["기타"] = 0
    # MLX 양자화 가중치: uint32 한 칸에 (32/bits)개 값이 묶여 있다. 양자화 보조값(scales/biases)은 세지 않는다.
    bits = quant_bits(path) or 0
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"safetensors 파일이 없습니다: {path}")
    for f in files:
        header = _header(f)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if bits and name.endswith((".scales", ".biases")) and \
                    header.get(name.rsplit(".", 1)[0] + ".weight", {}).get("dtype") == "U32":
                continue
            n = math.prod(info["shape"])
            if bits and info["dtype"] == "U32":
                n = n * 32 // bits
            for key, label in rules:
                if key in name:
                    parts[label] += n
                    break
            else:
                parts["기타"] += n
    return OrderedDict((k, v) for k, v in parts.items() if v)


def decoder_params(path: str, family: str = "") -> int:
    return count(path, family).get(DECODER, 0)


def _pad(s: str, width: int) -> str:
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(width - w, 0)


def report(path: str, family: str = "", limit: float = LIMIT) -> str:
    parts = count(path, family)
    total = sum(parts.values())
    dec = parts.get(DECODER, 0)
    lines = [f"  가중치: {path}", f"  {_pad('구성 요소', 34)}{'파라미터':>10s}"]
    for k, v in parts.items():
        if v < 5e6:
            continue
        mark = "  ◀ 판단 기준" if k == DECODER else ""
        lines.append(f"  {_pad(k, 34)}{v / 1e9:>10.3f} B{mark}")
    lines.append(f"  {_pad('전체', 34)}{total / 1e9:>10.3f} B")
    bits = quant_bits(path)
    if bits:
        lines.append(f"  가중치 {bits}비트 양자화 적용 — 파라미터 수는 양자화 전과 같다")
    lines.append(f"  ▶ 언어모델 디코더 {dec / 1e9:.2f}B  → {limit / 1e9:g}B 이하: {'충족' if dec <= limit else '미충족'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="구성 요소별 파라미터 수")
    ap.add_argument("path", help="safetensors 가중치 디렉터리")
    ap.add_argument("--family", default="", choices=["", *RULES], help="가중치 이름 규칙 (기본: config.json 으로 추정)")
    ap.add_argument("--limit", type=float, default=3.0, help="경량 기준 (B 단위, 기본 3)")
    args = ap.parse_args(argv)
    print(report(args.path, args.family, args.limit * 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
