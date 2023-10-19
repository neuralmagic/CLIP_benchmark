from typing import Union
import torch
from .deepsparse_clip import load_deepsparse_clip, load_onnx_clip
from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    "deepsparse_clip": load_deepsparse_clip,
    "onnx_clip": load_onnx_clip,
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        batch_size: int,
        device: Union[str, torch.device] = "cuda"
):
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    load_func = TYPE2FUNC[model_type]
    return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device, batch_size=batch_size)
