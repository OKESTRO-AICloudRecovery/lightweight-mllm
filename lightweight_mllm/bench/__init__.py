"""경량 멀티모달 LLM 모달리티별 벤치마크: 텍스트 MMLU · 이미지 MMMU · 음성 MMAU (0-shot, 탐욕적 생성).

  data      벤치마크 로딩 → 공통 Item 형식 (이미지·오디오는 캐시 파일 경로로)
  parse     응답 → 정답 판정 (MMMU 는 공식 평가 규칙)
  backends  모델 실행 백엔드 (MLX, PyTorch Omni) 와 등록 API
  params    구성 요소별 파라미터 수 (언어모델 디코더 기준 경량 판단)
  run       벤치마크 실행 (문항별 기록, 이어 달리기)
  report    결과 요약표
"""
