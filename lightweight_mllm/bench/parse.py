"""모델 응답 → 정답 판정.

- MMLU / MMAU: 응답에서 보기 글자(A, (A), "answer is A" …)를 찾아 정답 글자와 비교.
  글자가 없으면 보기 본문이 응답에 하나만 들어 있을 때 그 보기로 본다.
- MMMU: 공식 평가 코드(MMMU-Benchmark/MMMU, eval/utils/eval_utils.py, Apache-2.0)의 규칙을 옮겨 구현했다.
  객관식은 parse_multi_choice_response, 주관식은 parse_open_response + eval_open 에 대응한다.
  공식 코드는 보기를 못 찾으면 무작위로 고르므로 같은 결과가 나오도록 seed 42로 고정한다.
"""
from __future__ import annotations

import ast
import random
import re
from typing import Dict, List, Optional

LETTERS = "ABCDEFGHIJ"


def clean(resp: str) -> str:
    # 사고 과정 태그 등 제거
    resp = re.sub(r"<think>.*?</think>", " ", resp, flags=re.S)
    for end in ("<channel|>", "</think>"):  # 사고(풀이) 종료 표시 뒤가 최종 답
        if end in resp:
            resp = resp.split(end)[-1]
    return resp.strip()


def extract_letter(resp: str, n_choices: int) -> Optional[str]:
    letters = LETTERS[:n_choices]
    r = clean(resp)
    pats = [
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\*?\*?([{letters}])\b",
        rf"^\s*\(?\*?\*?([{letters}])\s*[\).:\*]",
        rf"^\s*\(?([{letters}])\)?\s*$",
        rf"\(([{letters}])\)",
        rf"\b([{letters}])\b",
    ]
    for p in pats:
        m = re.search(p, r, flags=re.I | re.M)
        if m:
            return m.group(1).upper()
    return None


def judge_letter(resp: str, gold_letter: str, choices_text: List[str]) -> Dict:
    """정답 글자 비교. choices_text: 글자 없는 보기 본문 목록."""
    letter = extract_letter(resp, len(choices_text))
    how = "letter"
    if letter is None:
        r = clean(resp).lower()
        hits = [i for i, c in enumerate(choices_text) if c and c.lower() in r]
        if len(hits) == 1:
            letter, how = LETTERS[hits[0]], "text"
        else:
            how = "none"
    return {"pred": letter, "correct": letter == gold_letter, "parse": how}


# --------------------------------------------------------------------------- MMMU (공식 규칙)
def mmmu_multi_choice(response: str, all_choices: List[str], index2ans: Dict[str, str], rng: random.Random) -> str:
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "
    index_ans, ans_with_brack, candidates = True, False, []
    for c in all_choices:
        if f"({c})" in response:
            candidates.append(c)
            ans_with_brack = True
    if not candidates:
        for c in all_choices:
            if f" {c} " in response:
                candidates.append(c)
    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(idx)
                index_ans = False
    if not candidates:
        return rng.choice(all_choices)
    if len(candidates) > 1:
        if index_ans:
            key = (lambda c: response.rfind(f"({c})")) if ans_with_brack else (lambda c: response.rfind(f" {c} "))
        else:
            key = lambda c: response.lower().rfind(index2ans[c].lower())  # noqa: E731
        return max(candidates, key=key)
    return candidates[0]


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(s: str):
    s = s.strip()
    if _check_is_number(s):
        s = s.replace(",", "")
        return [round(float(s), 2)]
    s = s.lower()
    return [" " + s, s + " "] if len(s) == 1 else [s]


def _extract_numbers(s: str) -> List[str]:
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+)(?![eE][+-]?\d+)(?![,\d])"
    return re.findall(pattern_commas, s) + re.findall(pattern_scientific, s) + re.findall(pattern_simple, s)


def mmmu_open(response: str) -> List:
    def key_responses(resp: str) -> List[str]:
        resp = resp.strip().strip(".").lower()
        subs = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        keys = []
        for i, s in enumerate(subs):
            cands = indicators + (["="] if i == len(subs) - 1 else [])
            shortest = None
            for ind in cands:
                if ind in s:
                    part = s.split(ind)[-1].strip()
                    if shortest is None or len(part) < len(shortest):
                        shortest = part
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", "'"]:
                keys.append(shortest)
        return keys or [resp]

    keys = key_responses(response)
    preds = list(keys)
    for k in keys:
        preds.extend(_extract_numbers(k))
    out = []
    for p in preds:
        out.extend(_normalize_str(p))
    return list(dict.fromkeys(out))


def mmmu_eval_open(gold, preds: List) -> bool:
    golds = gold if isinstance(gold, list) else [gold]
    norm = []
    for g in golds:
        norm.extend(_normalize_str(g))
    for p in preds:
        if isinstance(p, str):
            for g in norm:
                if isinstance(g, str) and g in p:
                    return True
        elif p in norm:
            return True
    return False


def judge_mmmu(resp: str, item_meta: Dict, gold: str, rng: random.Random) -> Dict:
    resp = clean(resp)
    if item_meta["type"] == "multiple-choice":
        opts = item_meta["options"]
        all_choices = [LETTERS[i] for i in range(len(opts))]
        index2ans = {LETTERS[i]: str(o) for i, o in enumerate(opts)}
        pred = mmmu_multi_choice(resp, all_choices, index2ans, rng)
        return {"pred": pred, "correct": pred == gold, "parse": "mmmu-mc"}
    preds = mmmu_open(resp)
    try:
        gold_v = ast.literal_eval(gold) if gold.startswith("[") else gold  # 공식 데이터의 리스트형 정답
    except Exception:
        gold_v = gold
    return {"pred": str(preds[:3]), "correct": mmmu_eval_open(gold_v, preds), "parse": "mmmu-open"}


def judge(item, resp: str, rng: random.Random) -> Dict:
    if item.bench == "mmmu":
        return judge_mmmu(resp, item.meta, item.gold, rng)
    if item.bench == "mmau":
        choices = [re.sub(r"^\([A-J]\)\s*", "", c) for c in item.meta["choices"]]
        gold_letter = re.match(r"^\(([A-J])\)", item.gold).group(1)
        return judge_letter(resp, gold_letter, choices)
    return judge_letter(resp, item.gold, [""] * item.meta["n_choices"])
