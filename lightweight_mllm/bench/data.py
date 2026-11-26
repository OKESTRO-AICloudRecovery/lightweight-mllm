"""벤치마크 로딩: MMLU(텍스트) · MMMU(이미지) · MMAU(음성/오디오).

모든 문항을 같은 형태로 바꾼다.
    Item(id, bench, prompt, images=[파일 경로], audios=[파일 경로], gold, meta)
이미지·오디오는 <데이터 디렉터리>/cache 아래에 한 번만 파일로 풀어 두고 경로로 넘긴다.
데이터 디렉터리는 환경변수 MMB_DATA (기본 ./data) 로 정한다.
"""
from __future__ import annotations

import ast
import glob
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd



def data_dir() -> str:
    return os.environ.get("MMB_DATA", "data")


def cache_dir() -> str:
    return os.path.join(data_dir(), "cache")

LETTERS = "ABCDEFGHIJ"


@dataclass
class Item:
    id: str
    bench: str
    prompt: str
    gold: str
    images: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- MMLU
def load_mmlu() -> List[Item]:
    """MMLU test 14,042문항, 0-shot. 정답은 A~D 글자."""
    df = pd.read_parquet(os.path.join(data_dir(), "mmlu", "all", "test-00000-of-00001.parquet"))
    items = []
    for i, r in enumerate(df.itertuples(index=False)):
        subject = r.subject.replace("_", " ")
        opts = "\n".join(f"{LETTERS[k]}. {c}" for k, c in enumerate(r.choices))
        prompt = (f"The following is a multiple choice question about {subject}.\n\n"
                  f"{r.question}\n{opts}\n\n"
                  "Answer with only the letter (A, B, C, or D) of the correct option.")
        items.append(Item(f"mmlu_{i:05d}", "mmlu", prompt, LETTERS[int(r.answer)],
                          meta={"subject": r.subject, "n_choices": len(r.choices), "n_letters": len(r.choices)}))
    return items


# --------------------------------------------------------------------------- MMMU
def load_mmmu() -> List[Item]:
    """MMMU validation 900문항(30과목). 객관식 847 + 주관식 53."""
    files = sorted(glob.glob(os.path.join(data_dir(), "mmmu", "*", "validation-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out_dir = os.path.join(cache_dir(), "mmmu_images")
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for r in df.itertuples(index=False):
        images = []
        for k in range(1, 8):
            img = getattr(r, f"image_{k}")
            if img is None or not isinstance(img, dict) or not img.get("bytes"):
                continue
            path = os.path.join(out_dir, f"{r.id}_{k}.png")
            if not os.path.exists(path):
                _save_png(img["bytes"], path)
            images.append(path)
        options = ast.literal_eval(r.options) if r.options else []
        # VLMEvalKit(MMMU_DEV_VAL) 기본 프롬프트 형식
        prompt = f"Question: {r.question}\n"
        if r.question_type == "multiple-choice":
            prompt += "Options:\n" + "".join(f"{LETTERS[k]}. {o}\n" for k, o in enumerate(options))
            prompt += "Please select the correct answer from the options above. \n"
        segments = _interleave(prompt, images)
        subject = r.id.split("_", 1)[1].rsplit("_", 1)[0]
        items.append(Item(r.id, "mmmu", prompt, str(r.answer), images=images,
                          meta={"subject": subject, "type": r.question_type, "options": options,
                                "segments": segments,
                                "n_letters": len(options) if r.question_type == "multiple-choice" else 0}))
    return items


def _interleave(text: str, images: List[str]):
    """VLMEvalKit split_MMMU: 본문의 '<image n>' 자리에 n번째 이미지를 끼워 넣는다 (보기 안 포함).
    자리표시가 없으면 이미지를 모두 앞에 둔다."""
    parts = text.split("<image ")
    if len(parts) == 1:
        return [("image", p) for p in images] + [("text", text)]
    segs = [("text", parts[0])]
    for seg in parts[1:]:
        idx = int(seg[0]) - 1
        segs.append(("image", images[idx]))
        segs.append(("text", seg[2:]))
    return segs


def _save_png(data: bytes, path: str) -> None:
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    tmp = path + ".tmp.png"
    im.save(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- MMAU
def load_mmau() -> List[Item]:
    """MMAU test-mini 1,000문항 (sound · music · speech 각 1/3). 보기 '(A) …' 형태."""
    import soundfile as sf
    df = pd.read_parquet(os.path.join(data_dir(), "mmau", "test_mini.parquet"))
    out_dir = os.path.join(cache_dir(), "mmau_audio")
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for i, r in enumerate(df.itertuples(index=False)):
        attrs = json.loads(r.other_attributes) if r.other_attributes else {}
        path = os.path.join(out_dir, f"mmau_{i:04d}.wav")
        if not os.path.exists(path):
            audio, sr = sf.read(io.BytesIO(r.context["bytes"]))
            sf.write(path + ".tmp.wav", audio, sr)
            os.replace(path + ".tmp.wav", path)
        choices = list(r.choices)
        prompt = (f"{r.instruction}\n" + "\n".join(choices) +
                  "\n\nSelect one option from the provided choices. Answer with only the option's letter.")
        items.append(Item(f"mmau_{i:04d}", "mmau", prompt, r.answer, audios=[path],
                          meta={"task": attrs.get("task"), "difficulty": attrs.get("difficulty"),
                                "choices": choices, "mmau_id": attrs.get("id"), "n_letters": len(choices)}))
    return items


LOADERS = {"mmlu": load_mmlu, "mmmu": load_mmmu, "mmau": load_mmau}
BENCH_LABEL = {"mmlu": "Text · MMLU", "mmmu": "Image · MMMU", "mmau": "Voice · MMAU"}
GROUP_KEY = {"mmlu": "subject", "mmmu": "subject", "mmau": "task"}


def pilot_subset(items: List[Item], bench: str, per_group: int) -> List[Item]:
    """과목/과제별로 고르게 간격을 두고 per_group개씩 뽑은 결정적 표본."""
    key = GROUP_KEY[bench]
    groups: Dict[str, List[Item]] = {}
    for it in items:
        groups.setdefault(it.meta.get(key), []).append(it)
    out = []
    for g in groups.values():
        n = min(per_group, len(g))
        step = len(g) / n
        out.extend(g[int(k * step)] for k in range(n))
    return sorted(out, key=lambda it: it.id)
