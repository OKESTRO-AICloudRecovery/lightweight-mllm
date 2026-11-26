import random

from lightweight_mllm.bench.data import Item
from lightweight_mllm.bench.parse import clean, extract_letter, judge


def test_extract_letter_variants():
    assert extract_letter("B", 4) == "B"
    assert extract_letter("The answer is (C).", 4) == "C"
    assert extract_letter("**D**. because ...", 4) == "D"
    assert extract_letter("no idea", 4) is None


def test_clean_strips_thinking():
    assert clean("<think>long reasoning A B</think> C") == "C"


def test_judge_mmlu_and_mmau():
    rng = random.Random(42)
    mmlu = Item("mmlu_00000", "mmlu", "q", "B", meta={"subject": "x", "n_choices": 4})
    assert judge(mmlu, "Answer: B", rng)["correct"]
    mmau = Item("mmau_0000", "mmau", "q", "(C) dog barking", meta={"task": "sound", "choices": ["(A) cat", "(B) bird", "(C) dog barking"]})
    assert judge(mmau, "C", rng)["correct"]
    assert judge(mmau, "Dog barking.", rng)["pred"] == "C"   # 글자가 없으면 보기 본문으로 판정


def test_judge_mmmu_multiple_choice_and_open():
    rng = random.Random(42)
    mc = Item("validation_Art_1", "mmmu", "q", "B", meta={"type": "multiple-choice", "options": ["red", "blue", "green"]})
    assert judge(mc, "The correct answer is (B) blue.", rng)["correct"]
    op = Item("validation_Math_2", "mmmu", "q", "12.5", meta={"type": "open", "options": []})
    assert judge(op, "So the final answer is 12.5", rng)["correct"]
    lst = Item("validation_Math_3", "mmmu", "q", "['3', '3.0']", meta={"type": "open", "options": []})
    assert judge(lst, "therefore 3", rng)["correct"]
