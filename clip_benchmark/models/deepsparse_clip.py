import open_clip

def load_onnx_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu", batch_size: int = None, **kwargs):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)

    model = CLIPOnnxModel(name=f"{model_name}::{pretrained}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    model.start_sessions(providers=providers, batch_size=batch_size)

    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

def load_deepsparse_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu", batch_size: int = None, **kwargs):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)

    model = CLIPDeepsparseModel(name=f"{model_name}::{pretrained}")
    model.start_sessions(providers=['CPUExecutionProvider'], batch_size=batch_size)

    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

import os
from typing import Dict, Optional
import numpy
import torch
import deepsparse

from clip_server.model.pretrained_models import (
    download_model,
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
)
from clip_server.model.clip_model import BaseCLIPModel

class CLIPOnnxModel(BaseCLIPModel):
    def __init__(
        self, name: str, model_path: str = None, dtype: Optional[str] = 'fp32'
    ):
        super().__init__(name)
        self._dtype = dtype
        if name in _MODELS:
            self._cache_dir = os.path.expanduser(
                f'./models-rs/{name.replace("/", "-").replace("::", "-")}'
                # f'~/.cache/clip/{name.replace("/", "-").replace("::", "-")}'
            )
            if not model_path:
                textual_model_name, textual_model_md5 = _MODELS[name][0]
                self._textual_path = download_model(
                    url=_S3_BUCKET_V2 + textual_model_name,
                    target_folder=self._cache_dir,
                    md5sum=textual_model_md5,
                    with_resume=True,
                )
                visual_model_name, visual_model_md5 = _MODELS[name][1]
                self._visual_path = download_model(
                    url=_S3_BUCKET_V2 + visual_model_name,
                    target_folder=self._cache_dir,
                    md5sum=visual_model_md5,
                    with_resume=True,
                )
            else:
                if os.path.isdir(model_path):
                    self._textual_path = os.path.join(model_path, 'textual.onnx')
                    self._visual_path = os.path.join(model_path, 'visual.onnx')
                    if not os.path.isfile(self._textual_path) or not os.path.isfile(
                        self._visual_path
                    ):
                        raise RuntimeError(
                            f'The given model path {model_path} does not contain `textual.onnx` and `visual.onnx`'
                        )
                else:
                    raise RuntimeError(
                        f'The given model path {model_path} should be a folder containing both '
                        f'`textual.onnx` and `visual.onnx`.'
                    )
        else:
            raise RuntimeError(
                'CLIP model {} not found or not supports ONNX backend; below is a list of all available models:\n{}'.format(
                    name,
                    ''.join(['\t- {}\n'.format(i) for i in list(_MODELS.keys())]),
                )
            )

    def start_sessions(
        self,
        batch_size,
        **kwargs,
    ):
        import onnxruntime as ort

        self._batch_size = batch_size

        # Optional overrides
        visual_model = os.environ.get("VISUAL_MODEL", "visual.onnx")
        textual_model = os.environ.get("TEXTUAL_MODEL", "textual.onnx")
        print(visual_model, textual_model)

        self._visual_path = f"{self._cache_dir}/{visual_model}"
        self._textual_path = f"{self._cache_dir}/{textual_model}"
        
        print("Compiling models..")
        print(self._visual_path)
        print(self._textual_path)

        self._visual_session = ort.InferenceSession(self._visual_path, **kwargs)
        self._textual_session = ort.InferenceSession(self._textual_path, **kwargs)
        self._visual_session.disable_fallback()
        self._textual_session.disable_fallback()

        self.max_samples = int(os.environ.get("MAX_SAMPLES", 500))
        self.textual_samples = []
        self.visual_samples = []

    def encode_image(self, image_input: Dict):
        pixel_values = numpy.array(image_input.cpu())
        for i in range(pixel_values.shape[0]):
            if len(self.visual_samples) < self.max_samples:
                self.visual_samples.append([pixel_values[i]])
        (visual_output,) = self._visual_session.run(None, {'pixel_values':pixel_values})
        return torch.Tensor(visual_output).to("cuda:0" if torch.cuda.is_available() else "cpu")

    # def encode_text(self, text_input: Dict):
    #     input_ids = numpy.array(text_input.cpu(), dtype=numpy.int32)
    #     attention_mask = (input_ids != 0).astype(numpy.int32)
    #     for i in range(input_ids.shape[0]):
    #         if len(self.textual_samples) < self.max_samples:
    #             self.textual_samples.append([input_ids[i], attention_mask[i]])
    #     (textual_output,) = self._textual_session.run(None, {'input_ids':input_ids, 'attention_mask':attention_mask})
    #     return torch.Tensor(textual_output).to("cuda:0" if torch.cuda.is_available() else "cpu")

    def encode_text(self, text_input: Dict):
        input_ids = numpy.array(text_input.cpu(), dtype=numpy.int32)
        # attention_mask = (input_ids != 0).astype(numpy.int32)
        (textual_output,) = self._textual_session.run(None, {'input_ids':input_ids})
        return torch.Tensor(textual_output).to("cuda:0" if torch.cuda.is_available() else "cpu")

    def sparsify_model(self, original_onnx_path, sample_inputs):
        from sparsifyml.one_shot import sparsify_fast
        import onnx
        model = onnx.load(original_onnx_path)
        IGNORE_LIST = [node.op_type for node in model.graph.node if node.op_type not in ["MatMul", "Gemm"]]
        print(f"Sparsifying {original_onnx_path} with {len(sample_inputs)} samples")
        for quant in [False, True]:
            for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                qstr = "-int8" if quant else "-fp32"
                sstr = f"-{int(sparsity*100)}sparse" if sparsity else "-dense"
                filename, file_extention = os.path.splitext(original_onnx_path)
                new_onnx_path = filename + f"{qstr}{sstr}-OBC" + file_extention
                model = sparsify_fast(
                    model=original_onnx_path,
                    sample_input=sample_inputs,
                    batches=len(sample_inputs),
                    sparsity=sparsity,
                    quantization=dict(ignore=IGNORE_LIST) if quant else False,
                    save_path=new_onnx_path
                    )
                print(f"Sparsified model at {new_onnx_path}")

    def __del__(self):
        # on destruction is when we sparsify!
        if os.environ.get("SPARSIFY"):
            self.sparsify_model(self._visual_path, self.visual_samples)
            self.sparsify_model(self._textual_path, self.textual_samples)

    # dummy function
    def eval(self):
        pass

class CLIPDeepsparseModel(CLIPOnnxModel):
    def __init__(
        self, name: str, model_path: str = None, dtype: Optional[str] = 'fp32'
    ):
        super().__init__(name)

    def start_sessions(
        self,
        batch_size,
        **kwargs,
    ):
        super().start_sessions(batch_size, **kwargs)

        # Override sessions with DeepSparse
        self._visual_session = deepsparse.Engine(self._visual_path, batch_size=batch_size, input_shapes=[[1,3,240,240]])
        # self._textual_session = deepsparse.Engine(self._textual_path, batch_size=batch_size, input_shapes=[[1,77],[1,77]])
        self._textual_session = deepsparse.Engine(self._textual_path, batch_size=batch_size, input_shapes=[[1,77], [1]])

    def encode_image(self, image_input: Dict):
        pixel_values = numpy.array(image_input.cpu())
        for i in range(pixel_values.shape[0]):
            if len(self.visual_samples) < self.max_samples:
                self.visual_samples.append([pixel_values[i]])
        (visual_output,) = self.batched_run(self._visual_session, [pixel_values])
        return torch.Tensor(visual_output).to("cuda:0" if torch.cuda.is_available() else "cpu")

    # def encode_text(self, text_input: Dict):
    #     input_ids = numpy.array(text_input.cpu(), dtype=numpy.int32)
    #     attention_mask = (input_ids != 0).astype(numpy.int32)
    #     for i in range(input_ids.shape[0]):
    #         if len(self.textual_samples) < self.max_samples:
    #             self.textual_samples.append([input_ids[i], attention_mask[i]])
    #     (textual_output,) = self.batched_run(self._textual_session, [input_ids, attention_mask])
    #     return torch.Tensor(textual_output).to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def encode_text(self, text_input):
        input_ids = numpy.array(text_input.cpu(), dtype=numpy.int32)
        seq_len = input_ids.argmax(axis=-1)
        (textual_output,) = self.batched_run(self._textual_session, [input_ids, seq_len])
        return torch.Tensor(textual_output).to("cuda:0" if torch.cuda.is_available() else "cpu")

    def pad_to_batch(self, x):
        orig_batch_size = x.shape[0]
        npad = [(0, 0)] * x.ndim
        npad[0] = (0, self._batch_size-x.shape[0])
        x_pad = numpy.pad(x, pad_width=npad)
        return x_pad, orig_batch_size

    def batched_run(self, session, inputs):
        # determine the number of batches
        num_batches = int(numpy.ceil(inputs[0].shape[0] / self._batch_size))

        # initialize the output list
        outputs = []

        for i in range(num_batches):
            start = i * self._batch_size
            end = (i + 1) * self._batch_size
            batch_inputs = [data[start:end] for data in inputs]

            # if the current batch is smaller than the batch size, pad it to the batch size
            orig_batch_size = batch_inputs[0].shape[0]
            if orig_batch_size < self._batch_size:
                batch_inputs = [self.pad_to_batch(data)[0] for data in batch_inputs]

            # run the session
            batch_outputs = session.run(batch_inputs)
            batch_outputs = [out[:orig_batch_size] for out in batch_outputs]

            # add the batch outputs to the output list
            outputs.extend(batch_outputs)

        # return the final outputs
        return [numpy.concatenate(outputs, axis=0)]


_S3_BUCKET = (
    'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/'  # Deprecated
)
_S3_BUCKET_V2 = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/'
_MODELS = {
    'RN50::openai': (
        ('RN50/textual.onnx', '722418bfe47a1f5c79d1f44884bb3103'),
        ('RN50/visual.onnx', '5761475db01c3abb68a5a805662dcd10'),
    ),
    'RN50::yfcc15m': (
        ('RN50-yfcc15m/textual.onnx', '4ff2ea7228b9d2337b5440d1955c2108'),
        ('RN50-yfcc15m/visual.onnx', '87daa9b4a67449b5390a9a73b8c15772'),
    ),
    'RN50::cc12m': (
        ('RN50-cc12m/textual.onnx', '78fa0ae0ea47aca4b8864f709c48dcec'),
        ('RN50-cc12m/visual.onnx', '0e04bf92f3c181deea2944e322ebee77'),
    ),
    'RN101::openai': (
        ('RN101/textual.onnx', '2d9efb7d184c0d68a369024cedfa97af'),
        ('RN101/visual.onnx', '0297ebc773af312faab54f8b5a622d71'),
    ),
    'RN101::yfcc15m': (
        ('RN101-yfcc15m/textual.onnx', '7aa2a4e3d5b960998a397a6712389f08'),
        ('RN101-yfcc15m/visual.onnx', '681a72dd91c9c79464947bf29b623cb4'),
    ),
    'RN50x4::openai': (
        ('RN50x4/textual.onnx', 'd9d63d3fe35fb14d4affaa2c4e284005'),
        ('RN50x4/visual.onnx', '16afe1e35b85ad862e8bbdb12265c9cb'),
    ),
    'RN50x16::openai': (
        ('RN50x16/textual.onnx', '1525785494ff5307cadc6bfa56db6274'),
        ('RN50x16/visual.onnx', '2a293d9c3582f8abe29c9999e47d1091'),
    ),
    'RN50x64::openai': (
        ('RN50x64/textual.onnx', '3ae8ade74578eb7a77506c11bfbfaf2c'),
        ('RN50x64/visual.onnx', '1341f10b50b3aca6d2d5d13982cabcfc'),
    ),
    'ViT-B-32::openai': (
        ('ViT-B-32/textual.onnx', 'bd6d7871e8bb95f3cc83aff3398d7390'),
        ('ViT-B-32/visual.onnx', '88c6f38e522269d6c04a85df18e6370c'),
    ),
    'ViT-B-32::laion2b_e16': (
        ('ViT-B-32-laion2b_e16/textual.onnx', 'aa6eac88fe77d21f337e806417957497'),
        ('ViT-B-32-laion2b_e16/visual.onnx', '0cdc00a9dfad560153d40aced9df0c8f'),
    ),
    'ViT-B-32::laion400m_e31': (
        ('ViT-B-32-laion400m_e31/textual.onnx', '832f417bf1b3f1ced8f9958eda71665c'),
        ('ViT-B-32-laion400m_e31/visual.onnx', '62326b925ae342313d4cc99c2741b313'),
    ),
    'ViT-B-32::laion400m_e32': (
        ('ViT-B-32-laion400m_e32/textual.onnx', '93284915937ba42a2b52ae8d3e5283a0'),
        ('ViT-B-32-laion400m_e32/visual.onnx', 'db220821a31fe9795fd8c2ba419078c5'),
    ),
    'ViT-B-32::laion2b-s34b-b79k': (
        ('ViT-B-32-laion2b-s34b-b79k/textual.onnx', '84af5ae53da56464c76e67fe50fddbe9'),
        ('ViT-B-32-laion2b-s34b-b79k/visual.onnx', 'a2d4cbd1cf2632cd09ffce9b40bfd8bd'),
    ),
    'ViT-B-16::openai': (
        ('ViT-B-16/textual.onnx', '6f0976629a446f95c0c8767658f12ebe'),
        ('ViT-B-16/visual.onnx', 'd5c03bfeef1abbd9bede54a8f6e1eaad'),
    ),
    'ViT-B-16::laion400m_e31': (
        ('ViT-B-16-laion400m_e31/textual.onnx', '5db27763c06c06c727c90240264bf4f7'),
        ('ViT-B-16-laion400m_e31/visual.onnx', '04a6a780d855a36eee03abca64cd5361'),
    ),
    'ViT-B-16::laion400m_e32': (
        ('ViT-B-16-laion400m_e32/textual.onnx', '9abe000a51b6f1cbaac8fde601b16725'),
        ('ViT-B-16-laion400m_e32/visual.onnx', 'd38c144ac3ad7fbc1966f88ff8fa522f'),
    ),
    'ViT-B-16-plus-240::laion400m_e31': (
        (
            'ViT-B-16-plus-240-laion400m_e31/textual.onnx',
            '2b524e7a530a98010cc7e57756937c5c',
        ),
        (
            'ViT-B-16-plus-240-laion400m_e31/visual.onnx',
            'a78989da3300fd0c398a9877dd26a9f1',
        ),
    ),
    'ViT-B-16-plus-240::laion400m_e32': (
        (
            'ViT-B-16-plus-240-laion400m_e32/textual.onnx',
            '53c8d26726b386ca0749207876482907',
        ),
        (
            'ViT-B-16-plus-240-laion400m_e32/visual.onnx',
            '7a32c4272c1ee46f734486570d81584b',
        ),
    ),
    'ViT-L-14::openai': (
        ('ViT-L-14/textual.onnx', '325380b31af4837c2e0d9aba2fad8e1b'),
        ('ViT-L-14/visual.onnx', '53f5b319d3dc5d42572adea884e31056'),
    ),
    'ViT-L-14::laion400m_e31': (
        ('ViT-L-14-laion400m_e31/textual.onnx', '36216b85e32668ea849730a54e1e09a4'),
        ('ViT-L-14-laion400m_e31/visual.onnx', '15fa5a24916e2a58325c5cf70350c300'),
    ),
    'ViT-L-14::laion400m_e32': (
        ('ViT-L-14-laion400m_e32/textual.onnx', '8ba5b76ba71992923470c0261b10a67c'),
        ('ViT-L-14-laion400m_e32/visual.onnx', '49db3ba92bd816001e932530ad92d76c'),
    ),
    'ViT-L-14::laion2b-s32b-b82k': (
        ('ViT-L-14-laion2b-s32b-b82k/textual.onnx', 'da36a6cbed4f56abf576fdea8b6fe2ee'),
        ('ViT-L-14-laion2b-s32b-b82k/visual.onnx', '1e337a190abba6a8650237dfae4740b7'),
    ),
    'ViT-L-14-336::openai': (
        ('ViT-L-14@336px/textual.onnx', '78fab479f136403eed0db46f3e9e7ed2'),
        ('ViT-L-14@336px/visual.onnx', 'f3b1f5d55ca08d43d749e11f7e4ba27e'),
    ),
    'ViT-H-14::laion2b-s32b-b79k': (
        ('ViT-H-14-laion2b-s32b-b79k/textual.onnx', '41e73c0c871d0e8e5d5e236f917f1ec3'),
        ('ViT-H-14-laion2b-s32b-b79k/visual.zip', '38151ea5985d73de94520efef38db4e7'),
    ),
    'ViT-g-14::laion2b-s12b-b42k': (
        ('ViT-g-14-laion2b-s12b-b42k/textual.onnx', 'e597b7ab4414ecd92f715d47e79a033f'),
        ('ViT-g-14-laion2b-s12b-b42k/visual.zip', '6d0ac4329de9b02474f4752a5d16ba82'),
    ),
    # older version name format
    'RN50': (
        ('RN50/textual.onnx', '722418bfe47a1f5c79d1f44884bb3103'),
        ('RN50/visual.onnx', '5761475db01c3abb68a5a805662dcd10'),
    ),
    'RN101': (
        ('RN101/textual.onnx', '2d9efb7d184c0d68a369024cedfa97af'),
        ('RN101/visual.onnx', '0297ebc773af312faab54f8b5a622d71'),
    ),
    'RN50x4': (
        ('RN50x4/textual.onnx', 'd9d63d3fe35fb14d4affaa2c4e284005'),
        ('RN50x4/visual.onnx', '16afe1e35b85ad862e8bbdb12265c9cb'),
    ),
    'RN50x16': (
        ('RN50x16/textual.onnx', '1525785494ff5307cadc6bfa56db6274'),
        ('RN50x16/visual.onnx', '2a293d9c3582f8abe29c9999e47d1091'),
    ),
    'RN50x64': (
        ('RN50x64/textual.onnx', '3ae8ade74578eb7a77506c11bfbfaf2c'),
        ('RN50x64/visual.onnx', '1341f10b50b3aca6d2d5d13982cabcfc'),
    ),
    'ViT-B/32': (
        ('ViT-B-32/textual.onnx', 'bd6d7871e8bb95f3cc83aff3398d7390'),
        ('ViT-B-32/visual.onnx', '88c6f38e522269d6c04a85df18e6370c'),
    ),
    'ViT-B/16': (
        ('ViT-B-16/textual.onnx', '6f0976629a446f95c0c8767658f12ebe'),
        ('ViT-B-16/visual.onnx', 'd5c03bfeef1abbd9bede54a8f6e1eaad'),
    ),
    'ViT-L/14': (
        ('ViT-L-14/textual.onnx', '325380b31af4837c2e0d9aba2fad8e1b'),
        ('ViT-L-14/visual.onnx', '53f5b319d3dc5d42572adea884e31056'),
    ),
    'ViT-L/14@336px': (
        ('ViT-L-14@336px/textual.onnx', '78fab479f136403eed0db46f3e9e7ed2'),
        ('ViT-L-14@336px/visual.onnx', 'f3b1f5d55ca08d43d749e11f7e4ba27e'),
    ),
    # MultilingualCLIP models
    'M-CLIP/LABSE-Vit-L-14': (
        ('M-CLIP-LABSE-Vit-L-14/textual.onnx', '03727820116e63c7d19c72bb5d839488'),
        ('M-CLIP-LABSE-Vit-L-14/visual.onnx', 'a78028eab30084c3913edfb0c8411f15'),
    ),
    'M-CLIP/XLM-Roberta-Large-Vit-B-32': (
        (
            'M-CLIP-XLM-Roberta-Large-Vit-B-32/textual.zip',
            '41f51ec9af4754d11c7b7929e2caf5b9',
        ),
        (
            'M-CLIP-XLM-Roberta-Large-Vit-B-32/visual.onnx',
            '5f18f68ac94e294863bfd1f695c8c5ca',
        ),
    ),
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus': (
        (
            'M-CLIP-XLM-Roberta-Large-Vit-B-16Plus/textual.zip',
            '6c3e55f7d2d6c12f2c1f1dd36fdec607',
        ),
        (
            'M-CLIP-XLM-Roberta-Large-Vit-B-16Plus/visual.onnx',
            '467a3ef3e5f50abcf850c3db9e705f8e',
        ),
    ),
    'M-CLIP/XLM-Roberta-Large-Vit-L-14': (
        (
            'M-CLIP-XLM-Roberta-Large-Vit-L-14/textual.zip',
            '3dff00335dc3093acb726dab975ae57d',
        ),
        (
            'M-CLIP-XLM-Roberta-Large-Vit-L-14/visual.onnx',
            'a78028eab30084c3913edfb0c8411f15',
        ),
    ),
}
