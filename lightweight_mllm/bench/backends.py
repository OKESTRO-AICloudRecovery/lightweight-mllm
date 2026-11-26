"""모델 실행 백엔드: 탐욕적 자유 생성(temperature 0)으로 응답을 만들고, 답은 parse.py 규칙으로 추출한다.

- mlx        : Apple Silicon MLX (mlx-vlm) — 텍스트·이미지 VLM
- torch-omni : PyTorch (transformers) — 텍스트·이미지·음성 Omni 모델 (음성 생성부 Talker 끔)

백엔드는 Item(prompt, images, audios, ...)을 받아 응답 문자열을 돌려주는 호출 가능한 객체다.
다른 추론 방식을 쓰려면 같은 형태의 객체를 만들어 register_backend 로 등록하면 된다.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

# 이름 → (백엔드 종류, 기본 가중치 경로). 경로는 run --model-path 로 바꿀 수 있다.
MODELS: Dict[str, Tuple[str, str]] = {
    "qwen25-omni-3b": ("torch-omni", os.path.join("models", "Qwen2.5-Omni-3B")),
    "qwen25-vl-3b": ("mlx", os.path.join("models", "Qwen2.5-VL-3B-Instruct")),
    "qwen3-vl-2b": ("mlx", os.path.join("models", "Qwen3-VL-2B-Instruct")),
    "qwen35-2b": ("mlx", os.path.join("models", "Qwen3.5-2B")),
    "gemma4-e2b": ("mlx", os.path.join("models", "gemma-4-E2B-it")),
}
MAX_TOKENS = {"mmlu": 32, "mmmu": 256, "mmau": 32}
# 이미지 입력 해상도 범위 (VLMEvalKit 설정: min 1280×28×28, max 16384×28×28 픽셀)
IMG_MIN_PIXELS = 1280 * 28 * 28
IMG_MAX_PIXELS = 16384 * 28 * 28


class MLXBackend:
    def __init__(self, path: str):
        from mlx_vlm import apply_chat_template, generate, load
        self._apply, self._generate = apply_chat_template, generate
        self.model, self.processor = load(path)
        self.config = self.model.config

    def __call__(self, item) -> str:
        prompt = self._apply(self.processor, self.config, item.prompt, enable_thinking=False,
                             num_images=len(item.images), num_audios=len(item.audios))
        return self._generate(self.model, self.processor, prompt, image=item.images or None, audio=item.audios or None,
                              temperature=0.0, max_tokens=MAX_TOKENS[item.bench], verbose=False).text


class TorchOmniBackend:
    def __init__(self, path: str):
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.torch, self._mm = torch, process_mm_info
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16)
        self.model.disable_talker()
        self.model.to(self.device).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(path)

    def _content(self, item) -> List[dict]:
        if item.meta.get("segments"):
            # MMMU: 본문의 <image n> 자리에 이미지를 끼운다
            return [{"type": "image", "image": v, "min_pixels": IMG_MIN_PIXELS, "max_pixels": IMG_MAX_PIXELS}
                    if kind == "image" else {"type": "text", "text": v}
                    for kind, v in item.meta["segments"] if v]
        content = [{"type": "image", "image": p} for p in item.images]
        content += [{"type": "audio", "audio": p} for p in item.audios]
        content.append({"type": "text", "text": item.prompt})
        return content

    def __call__(self, item) -> str:
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": self._content(item)}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._mm(messages, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=False).to(self.device)
        for k, v in inputs.items():
            if self.torch.is_floating_point(v):
                inputs[k] = v.to(self.model.dtype)
        with self.torch.inference_mode():
            seq = self.model.thinker.generate(**inputs, do_sample=False, use_audio_in_video=False,
                                              max_new_tokens=MAX_TOKENS[item.bench])
        return self.processor.batch_decode(seq[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


BACKENDS: Dict[str, Callable[[str], Callable]] = {"mlx": MLXBackend, "torch-omni": TorchOmniBackend}


def register_backend(kind: str, factory: Callable[[str], Callable]) -> None:
    """새 백엔드 종류 등록: factory(가중치 경로) → item 을 받아 응답을 돌려주는 객체."""
    BACKENDS[kind] = factory


def load_backend(model: str, path: str = "", kind: str = ""):
    """등록된 모델 이름 또는 (경로, 종류)로 백엔드를 만든다."""
    default_kind, default_path = MODELS.get(model, ("", ""))
    kind = kind or default_kind
    if kind not in BACKENDS:
        raise ValueError(f"백엔드 종류를 알 수 없습니다: {kind!r} (모델 {model!r}, --backend 로 지정)")
    return BACKENDS[kind](path or default_path)
