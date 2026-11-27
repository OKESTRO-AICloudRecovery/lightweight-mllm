# lightweight-mllm

클라우드 운영 화면(차트·대시보드·로그)을 이해하는 경량 멀티모달 LLM을 위한 **VQA 학습 데이터셋 구축 도구**와 **모달리티별 벤치마크 도구**입니다.

[![CI](https://github.com/OKESTRO-AICloudRecovery/lightweight-mllm/actions/workflows/ci.yml/badge.svg)](https://github.com/OKESTRO-AICloudRecovery/lightweight-mllm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

## 1. 운영 시각 데이터 VQA 데이터셋 (`lightweight_mllm.vqa_dataset`)

메트릭 차트와 로그 화면에 대한 캡션과 질의응답을 **데이터에서 직접 계산한 사실**로 만듭니다.
최댓값 계열과 시점, 추세(상승·하락·유지), 이상 구간 진입 후 정상 평균 대비 배수, 로그 레벨별 줄 수와 마지막 오류 메시지 같은 값을 수치에서 뽑으므로
사람이 일일이 쓰지 않아도 대규모로 만들 수 있고 정답이 틀리지 않습니다.

- 서술형·객관식 질의응답을 instruction-following 대화 형식(JSONL)으로 축적
- 이미지와 질문 해시로 중복 제거
- 이미지 단위 학습/검증 분할 (같은 이미지의 질의응답은 한쪽에만)

```python
from lightweight_mllm import ChartSample, DatasetBuilder, LogShot

chart = ChartSample("m1", "img/m1.png", "payment CPU 사용률",
                    {"payment": [41, 40, 42] * 6 + [85, 88, 91, 93], "checkout": [30] * 22},
                    times=[f"10:{i:02d}" for i in range(22)], unit="%", anomaly_from=18)
logs = LogShot("l1", "img/l1.png", ["10:00 INFO started", "10:01 ERROR db connection refused"], "cart")

b = DatasetBuilder()
b.add_chart(chart)      # 캡션 1건 + 질의응답 6건 (최댓값 계열, 급증 시각, 정상 대비 배수, 추세 ...)
b.add_logshot(logs)     # 캡션 1건 + 질의응답 (오류 줄 수, 최다 레벨, 마지막 오류 ...)
print(b.export("vqa_out", val_ratio=0.1))   # train.jsonl, val.jsonl, manifest.json
```

차트 이미지는 원하는 도구로 그린 뒤 경로만 넘기면 됩니다. 레코드 형식:

```json
{"id": "…", "image": "img/m1.png", "kind": "peak_time",
 "conversations": [{"from": "human", "value": "<image>\n…"}, {"from": "gpt", "value": "…"}], "meta": {…}}
```

## 2. 모달리티별 벤치마크 (`lightweight_mllm.bench`)

텍스트·이미지·음성 이해 능력을 공개 벤치마크로 측정합니다. 0-shot, 탐욕적 생성(temperature 0)이며 결정적입니다.

| 모달리티 | 벤치마크 | 문항 | 채점 |
|---|---|---|---|
| 텍스트 | MMLU test (57과목) | 14,042 | 응답에서 보기 글자 추출 |
| 이미지 | MMMU validation (30과목) | 900 | MMMU 공식 답 추출 규칙 (객관식·주관식) |
| 음성 | MMAU test-mini (sound·music·speech) | 1,000 | 보기 글자 또는 보기 본문 |

```bash
pip install -e ".[bench,mlx]"            # Apple Silicon MLX 백엔드 (또는 ".[bench,torch]")
scripts/download.sh <HF 모델 저장소>       # 데이터셋 3종 → ./data, 모델 → ./models
QUANT_BITS=8 scripts/download.sh <저장소>  # mlx-vlm 8비트 양자화 사본도 함께 생성

python -m lightweight_mllm.bench.params ./models/<모델>        # 구성 요소별 파라미터, 언어모델 디코더 기준 3B 이하 판정
python -m lightweight_mllm.bench.run --model my-vlm --model-path ./models/<모델> --backend mlx --bench mmmu --pilot 2
python -m lightweight_mllm.bench.run --model my-vlm --model-path ./models/<모델> --backend mlx --bench mmmu
python -m lightweight_mllm.bench.report --target mmlu=50 --target mmmu=52 --target mmau=65
```

- 결과는 `results/<tag>/<model>/<bench>.jsonl`(문항별 응답·판정)과 `.summary.json`(과목·과제별 정확도)에 쌓이며, 중단해도 이어서 실행됩니다.
- `--pilot N`은 과목/과제마다 N문항씩 고르게 뽑은 결정적 표본, `--limit N`은 앞에서부터 N문항입니다.
- 백엔드: `mlx`(mlx-vlm, 텍스트·이미지 VLM), `torch-omni`(transformers, 텍스트·이미지·음성 Omni 모델). 다른 추론 방식은 `backends.register_backend(종류, 생성 함수)`로 붙입니다.
- `params`는 safetensors 헤더만 읽어 모델을 메모리에 올리지 않고 세며, 비전·오디오 인코더·임베딩·음성 생성부를 언어모델 디코더와 나눠 보여 줍니다. MLX 양자화 가중치도 원래 파라미터 수로 셉니다.

## 테스트

```bash
pip install -e ".[dev]"
pytest -q
```

테스트는 합성 데이터와 가짜 백엔드를 써서 모델·데이터셋 없이 돌아갑니다.

## 라이선스

[Apache License 2.0](LICENSE). `bench/parse.py`의 MMMU 답 추출 규칙은 MMMU 공식 평가 코드(Apache-2.0)를 옮겨 구현했습니다([NOTICE](NOTICE)).
데이터셋(MMLU: MIT, MMMU: Apache-2.0, MMAU: CC BY-NC 4.0)과 모델 가중치는 저장소에 포함하지 않으며 각자의 라이선스를 따릅니다.

## 사사

이 소프트웨어는 2025년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구의 결과물입니다.
(RS-2025-25441560, 클라우드 장애극복을 위한 AI 어시스턴트 기반 운영·관리 자동화 기술개발)

---

**English.** (1) A deterministic VQA dataset builder for cloud-operations visuals: captions and QA pairs computed directly from the underlying metric series and log lines, exported as instruction-following JSONL with dedup and image-level train/val split. (2) A modality-wise benchmark harness for lightweight multimodal LLMs — MMLU (text), MMMU (image, official answer parsing) and MMAU (audio) — with resumable runs, MLX and PyTorch backends, and a header-only parameter counter that reports the language-model decoder size separately from encoders and embeddings.
