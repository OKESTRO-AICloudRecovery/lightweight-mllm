import json
import os
import struct

from lightweight_mllm.bench.params import count, decoder_params, guess_family, report


def write_safetensors(path, tensors):
    """헤더만 의미 있는 가짜 safetensors (params 는 헤더만 읽는다)."""
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for s in shape:
            n *= s
        size = n * (4 if dtype in ("F32", "U32") else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)


def test_omni_components(tmp_path):
    d = str(tmp_path)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "qwen2_5_omni"}, f)
    write_safetensors(os.path.join(d, "model.safetensors"), {
        "thinker.model.layers.0.mlp.weight": ("BF16", [1000, 2000]),
        "thinker.model.embed_tokens.weight": ("BF16", [500, 100]),
        "thinker.lm_head.weight": ("BF16", [500, 100]),
        "thinker.audio_tower.conv.weight": ("BF16", [10, 10]),
        "talker.layer.weight": ("BF16", [30, 10]),
    })
    assert guess_family(d) == "qwen25-omni"
    parts = count(d)
    assert parts["언어모델 디코더"] == 2_000_000 and parts["입력 임베딩"] == 50_000 and parts["음성 생성부"] == 300
    assert decoder_params(d) == 2_000_000
    assert "3B 이하: 충족" in report(d)


def test_mlx_quantized_counts_unpacked_values(tmp_path):
    d = str(tmp_path)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5", "quantization": {"bits": 8, "group_size": 64}}, f)
    write_safetensors(os.path.join(d, "model.safetensors"), {
        "language_model.layers.0.mlp.weight": ("U32", [1000, 500]),   # 8비트: uint32 하나에 4개
        "language_model.layers.0.mlp.scales": ("BF16", [1000, 32]),
        "language_model.layers.0.mlp.biases": ("BF16", [1000, 32]),
        "vision_tower.blocks.0.weight": ("BF16", [100, 100]),
    })
    assert guess_family(d) == "qwen-vl-mlx"
    parts = count(d)
    assert parts["언어모델 디코더"] == 2_000_000 and parts["비전 인코더"] == 10_000
    assert "8비트 양자화" in report(d)
