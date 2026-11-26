import json
import os

from lightweight_mllm import ChartSample, DatasetBuilder, LogShot, caption_chart, caption_logshot, qa_chart, qa_logshot, series_stats


def chart():
    base = [40 + (i % 3) for i in range(30)]
    return ChartSample("m1", "img/m1.png", "payment CPU 사용률",
                       {"payment": base[:20] + [85 + i for i in range(10)], "checkout": [30] * 30},
                       times=[f"10:{i:02d}" for i in range(30)], unit="%", anomaly_from=20)


def logshot():
    return LogShot("l1", "img/l1.png", ["10:00 INFO started", "10:01 ERROR db connection refused", "10:02 WARN retry"], "cart")


def test_series_stats_ratio_after_anomaly():
    st = series_stats([10] * 10 + [30] * 10, anomaly_from=10)
    assert st["max"] == 30 and st["ratio"] > 2.5


def test_chart_caption_and_qa_are_computed_from_values():
    s = chart()
    cap = caption_chart(s)
    assert "payment" in cap and "급증" in cap, cap
    qa = qa_chart(s, 6)
    assert any(q.kind == "max_series" and q.answer_letter for q in qa)
    assert any(q.kind == "ratio" and "배" in q.answer for q in qa), [(q.kind, q.answer) for q in qa]


def test_log_caption_and_qa():
    cap = caption_logshot(logshot())
    assert "ERROR" in cap or "오류" in cap
    qa = qa_logshot(logshot())
    assert any(q.kind == "error_count" and q.answer == "1줄" for q in qa)


def test_builder_dedup_split_and_export(tmp_path):
    b = DatasetBuilder()
    assert b.add_chart(chart()) >= 6 and b.add_chart(chart()) == 0   # 같은 이미지·질문은 중복 제거
    assert b.add_logshot(logshot()) >= 4
    m = b.export(str(tmp_path), 0.5)
    assert m["train"] + m["val"] == m["records"] and os.path.exists(os.path.join(tmp_path, "manifest.json"))
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        with open(os.path.join(tmp_path, name), encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    assert all(r["conversations"][0]["value"].startswith("<image>") for r in rows)
    # 같은 이미지의 질의응답은 학습/검증 한쪽에만 들어간다
    train_imgs = {r["image"] for r in rows[: m["train"]]}
    val_imgs = {r["image"] for r in rows[m["train"]:]}
    assert not (train_imgs & val_imgs)
