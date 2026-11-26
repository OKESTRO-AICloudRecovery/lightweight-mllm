"""lightweight_mllm: 클라우드 운영 시각 데이터 VQA 데이터셋 구축과 경량 멀티모달 LLM 벤치마크.

  vqa_dataset  차트·대시보드·로그 화면 → 수치에서 계산한 사실로 캡션·질의응답 생성, JSONL 내보내기
  bench        텍스트(MMLU)·이미지(MMMU)·음성(MMAU) 벤치마크 실행·채점·파라미터 산출
"""
from .vqa_dataset import (QA, ChartSample, DatasetBuilder, LogShot, caption_chart, caption_logshot, qa_chart, qa_logshot,
                          series_stats, validate_record)

__version__ = "2.0.0"
