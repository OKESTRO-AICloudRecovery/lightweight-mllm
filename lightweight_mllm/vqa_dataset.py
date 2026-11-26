"""클라우드 운영 시각 데이터(차트·대시보드·로그 화면) 캡셔닝 및 VQA 학습 데이터셋 구축 모듈.

메트릭 시계열로 그린 차트와 로그 화면에 대해, 데이터에서 직접 계산한 사실(최댓값 계열·급증 시각·정상 대비
배수·추세·오류 줄 수 등)로 캡션과 질의응답(서술형·객관식)을 생성한다. 사람이 쓴 설명이 아니라 수치에서
결정적으로 만든 정답이므로 대규모로 만들 수 있고 정답이 틀리지 않는다. 결과는 instruction-following 대화
형식(JSONL)으로 내보내며 중복 제거와 해시 기반 학습/검증 분할을 지원한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LETTERS = "ABCDEFGH"


@dataclass
class ChartSample:
    id: str
    image: str                              # 렌더링된 이미지 경로 (렌더링은 외부에서 수행)
    title: str
    series: Dict[str, List[float]]          # 계열 이름 → 값
    times: Optional[List[str]] = None       # 각 점의 시각 표기
    unit: str = ""
    kind: str = "line"
    anomaly_from: Optional[int] = None      # 이상 구간 시작 인덱스 (없으면 None)
    source: str = "metric"


@dataclass
class LogShot:
    id: str
    image: str
    lines: List[str]
    service: Optional[str] = None


@dataclass
class QA:
    question: str
    answer: str
    kind: str
    options: Optional[List[str]] = None     # 객관식 보기
    answer_letter: Optional[str] = None


# --------------------------------------------------------------------------- 수치 사실
def series_stats(values: Sequence[float], anomaly_from: Optional[int] = None) -> Dict[str, float]:
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return {}
    mx, mn = max(v), min(v)
    mean = sum(v) / n
    half = n // 2
    first, second = (sum(v[:half]) / half if half else mean), (sum(v[half:]) / (n - half) if n - half else mean)
    st = {"n": n, "max": mx, "min": mn, "mean": mean, "argmax": v.index(mx), "argmin": v.index(mn),
          "trend": "상승" if second > first * 1.1 else "하락" if second < first * 0.9 else "유지"}
    if anomaly_from is not None and 0 < anomaly_from < n:
        base = sum(v[:anomaly_from]) / anomaly_from
        after = sum(v[anomaly_from:]) / (n - anomaly_from)
        st["baseline"] = base
        st["after"] = after
        st["ratio"] = (after / base) if base > 0 else float("inf") if after > 0 else 1.0
    return st


def _fmt(x: float, unit: str = "") -> str:
    s = f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}" if abs(x) >= 1 else f"{x:.3f}"
    return s + (unit if unit else "")


def _t(sample: ChartSample, i: int) -> str:
    return sample.times[i] if sample.times and i < len(sample.times) else f"{i + 1}번째 지점"


# --------------------------------------------------------------------------- 차트 캡션·QA
def caption_chart(s: ChartSample) -> str:
    names = list(s.series)
    stats = {k: series_stats(v, s.anomaly_from) for k, v in s.series.items()}
    top = max(names, key=lambda k: stats[k]["max"])
    parts = [f"'{s.title}' {('선' if s.kind == 'line' else '막대')} 차트로 {len(names)}개 계열({', '.join(names)})을 보여준다."]
    parts.append(f"최댓값은 {top} 계열의 {_fmt(stats[top]['max'], s.unit)}로 {_t(s, stats[top]['argmax'])}에 나타난다.")
    rising = [k for k in names if stats[k]["trend"] == "상승"]
    if rising:
        parts.append(f"{', '.join(rising)} 계열은 후반부로 갈수록 상승하는 추세다.")
    if s.anomaly_from is not None:
        jumps = sorted(((k, stats[k].get("ratio", 1.0)) for k in names), key=lambda x: -x[1])
        k, r = jumps[0]
        if r >= 1.5:
            parts.append(f"{_t(s, s.anomaly_from)} 이후 {k} 계열이 정상 구간 평균 대비 약 {r:.1f}배로 급증한다.")
    return " ".join(parts)


def _mc(question: str, answer: str, distractors: Sequence[str], kind: str, rng: random.Random) -> QA:
    opts = [answer] + [d for d in distractors if d != answer][:3]
    rng.shuffle(opts)
    letter = LETTERS[opts.index(answer)]
    q = question + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    return QA(q, letter, kind, opts, letter)


def qa_chart(s: ChartSample, n: int = 6, seed: int = 0, multiple_choice: bool = True) -> List[QA]:
    rng = random.Random(f"{s.id}:{seed}")
    names = list(s.series)
    stats = {k: series_stats(v, s.anomaly_from) for k, v in s.series.items()}
    qa: List[QA] = []
    top = max(names, key=lambda k: stats[k]["max"])
    if len(names) > 1:
        qa.append(_mc("차트에서 최댓값이 가장 큰 계열은 무엇인가?", top, names, "max_series", rng) if multiple_choice
                  else QA("차트에서 최댓값이 가장 큰 계열은 무엇인가?", top, "max_series"))
    i = stats[top]["argmax"]
    qa.append(QA(f"{top} 계열이 최댓값에 도달한 시점은 언제인가?", _t(s, i), "peak_time"))
    qa.append(QA(f"{top} 계열의 최댓값은 얼마인가?", _fmt(stats[top]["max"], s.unit), "peak_value"))
    if s.anomaly_from is not None and "ratio" in stats[top]:
        r = stats[top]["ratio"]
        qa.append(QA(f"{_t(s, s.anomaly_from)} 이후 {top} 계열은 정상 구간 평균 대비 약 몇 배인가?", f"약 {r:.1f}배", "ratio"))
        rose = [k for k in names if stats[k].get("ratio", 1.0) >= 1.5]
        qa.append(QA("이상 구간 진입 후 1.5배 이상 상승한 계열을 모두 고르면?", ", ".join(rose) if rose else "없음", "rose_after"))
    tr = stats[top]["trend"]
    qa.append(_mc(f"{top} 계열의 전체 추세는?", tr, ["상승", "하락", "유지"], "trend", rng) if multiple_choice
              else QA(f"{top} 계열의 전체 추세는?", tr, "trend"))
    qa.append(QA("차트에 표시된 계열 수는 몇 개인가?", f"{len(names)}개", "count"))
    return qa[:n]


# --------------------------------------------------------------------------- 로그 화면 캡션·QA
_LEVELS = ("FATAL", "ERROR", "WARN", "INFO", "DEBUG")


def _level_of(line: str) -> str:
    u = line.upper()
    for lv in _LEVELS:
        if lv in u:
            return lv
    return "OTHER"


def caption_logshot(l: LogShot) -> str:
    levels = [_level_of(x) for x in l.lines]
    cnt = {lv: levels.count(lv) for lv in _LEVELS if levels.count(lv)}
    desc = ", ".join(f"{k} {v}줄" for k, v in cnt.items()) or "레벨 표기 없음"
    s = f"{len(l.lines)}줄의 로그 화면{'(' + l.service + ')' if l.service else ''}으로 {desc}이다."
    errs = [x for x in l.lines if _level_of(x) in ("ERROR", "FATAL")]
    if errs:
        s += f" 마지막 오류 줄은 '{errs[-1][:80]}' 이다."
    return s


def qa_logshot(l: LogShot, n: int = 4) -> List[QA]:
    levels = [_level_of(x) for x in l.lines]
    errs = [x for x in l.lines if _level_of(x) in ("ERROR", "FATAL")]
    qa = [QA("화면에 ERROR 또는 FATAL 레벨 로그는 몇 줄인가?", f"{len(errs)}줄", "error_count"),
          QA("가장 많이 나타난 로그 레벨은 무엇인가?", max(set(levels), key=levels.count), "top_level"),
          QA("로그 줄 수는 모두 몇 줄인가?", f"{len(l.lines)}줄", "line_count")]
    if errs:
        qa.append(QA("마지막 오류 로그의 내용을 요약하면?", errs[-1][:120], "last_error"))
    return qa[:n]


# --------------------------------------------------------------------------- 데이터셋 빌더
def _h(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class DatasetBuilder:
    """instruction-following 대화 레코드 축적 → 중복 제거 → 해시 분할 → JSONL 내보내기."""

    def __init__(self, image_token: str = "<image>"):
        self.image_token = image_token
        self.records: List[Dict[str, object]] = []
        self._seen: set = set()

    def add_record(self, image: str, instruction: str, answer: str, kind: str, source_id: str, meta: Optional[Dict[str, object]] = None) -> bool:
        key = _h(image, instruction)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.records.append({"id": f"{source_id}:{kind}:{key[:8]}", "image": image, "kind": kind,
                             "conversations": [{"from": "human", "value": f"{self.image_token}\n{instruction}"},
                                               {"from": "gpt", "value": answer}], "meta": meta or {}})
        return True

    def add_chart(self, s: ChartSample, n_qa: int = 6, seed: int = 0) -> int:
        added = int(self.add_record(s.image, "이 차트를 설명하라.", caption_chart(s), "caption", s.id, {"source": s.source}))
        for q in qa_chart(s, n_qa, seed):
            added += int(self.add_record(s.image, q.question, q.answer, q.kind, s.id, {"options": q.options} if q.options else {}))
        return added

    def add_logshot(self, l: LogShot, n_qa: int = 4) -> int:
        added = int(self.add_record(l.image, "이 로그 화면을 설명하라.", caption_logshot(l), "caption", l.id, {"source": "log"}))
        for q in qa_logshot(l, n_qa):
            added += int(self.add_record(l.image, q.question, q.answer, q.kind, l.id))
        return added

    def split(self, val_ratio: float = 0.1) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        """이미지 해시 기준 분할 — 같은 이미지의 QA는 같은 쪽에 들어간다."""
        train, val = [], []
        for r in self.records:
            bucket = int(_h(str(r["image"]))[:8], 16) / 0xFFFFFFFF
            (val if bucket < val_ratio else train).append(r)
        return train, val

    def stats(self) -> Dict[str, object]:
        kinds: Dict[str, int] = {}
        for r in self.records:
            kinds[str(r["kind"])] = kinds.get(str(r["kind"]), 0) + 1
        return {"records": len(self.records), "images": len({r["image"] for r in self.records}), "by_kind": kinds}

    def export(self, out_dir: str, val_ratio: float = 0.1) -> Dict[str, object]:
        os.makedirs(out_dir, exist_ok=True)
        train, val = self.split(val_ratio)
        for name, rows in (("train", train), ("val", val)):
            with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    validate_record(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest = {"train": len(train), "val": len(val), **self.stats()}
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        return manifest


def validate_record(r: Dict[str, object]) -> None:
    conv = r.get("conversations")
    if not r.get("id") or not r.get("image") or not isinstance(conv, list) or len(conv) != 2:
        raise ValueError(f"invalid record: {r.get('id')}")
    if conv[0].get("from") != "human" or conv[1].get("from") != "gpt" or not conv[1].get("value"):
        raise ValueError(f"invalid conversation: {r.get('id')}")
