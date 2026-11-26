import json
import os

import pandas as pd

from lightweight_mllm.bench import backends, report, run
from lightweight_mllm.bench.data import load_mmlu, pilot_subset


def write_mmlu(base):
    d = os.path.join(base, "mmlu", "all")
    os.makedirs(d)
    rows = [{"subject": s, "question": f"{s} question {i}", "choices": ["w", "x", "y", "z"], "answer": i % 4}
            for s in ("anatomy", "astronomy") for i in range(4)]
    pd.DataFrame(rows).to_parquet(os.path.join(d, "test-00000-of-00001.parquet"))


class AlwaysA:
    def __init__(self, path):
        self.path = path

    def __call__(self, item):
        return "A"


def test_load_mmlu_and_pilot(tmp_path, monkeypatch):
    write_mmlu(str(tmp_path))
    monkeypatch.setenv("MMB_DATA", str(tmp_path))
    items = load_mmlu()
    assert len(items) == 8 and items[0].gold == "A" and "Answer with only the letter" in items[0].prompt
    assert len(pilot_subset(items, "mmlu", 1)) == 2


def test_run_with_registered_backend_and_resume(tmp_path, monkeypatch, capsys):
    write_mmlu(str(tmp_path / "data"))
    monkeypatch.setenv("MMB_DATA", str(tmp_path / "data"))
    backends.register_backend("always-a", AlwaysA)
    out = str(tmp_path / "results")
    args = ["--model", "dummy", "--backend", "always-a", "--bench", "mmlu", "--out", out, "--tag", "t"]
    assert run.main(args) == 0
    with open(os.path.join(out, "t", "dummy", "mmlu.summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    assert s["n"] == 8 and s["accuracy"] == 0.25 and s["parse_fail"] == 0
    assert run.main(args) == 0                      # 이어 달리기: 남은 문항 0
    assert "완료 8, 남음 0" in capsys.readouterr().out
    assert report.main(["--out", out, "--tag", "t", "--target", "mmlu=20"]) == 0
    assert "충족" in capsys.readouterr().out
