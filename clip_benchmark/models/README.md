# DeepSparse ONNX Backend for OpenCLIP

This code was adapted from the OpenCLIP compatible ONNX implementation from [jina-ai/clip-as-service](https://github.com/jina-ai/clip-as-service/blob/main/server/clip_server/model/clip_onnx.py)

## Setup

Start a local install of this CLIP_benchmark repository. Make sure you are on the right repo and branch
```
git clone https://github.com/neuralmagic/CLIP_benchmark.git
cd CLIP_benchmark
git checkout deepsparse
pip install -e .
```

Install other dependencies
* `clip-server[onnx]` aka [CLIP-as-service](https://github.com/jina-ai/clip-as-service#install)
* DeepSparse and SparseML (at least `1.5.0.20230420`)
* A compatible version of PyTorch
```
pip install "clip-server[onnx]"
pip install deepsparse-nightly sparseml-nightly
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116\n
```

Install `sparsifyml`

## Commons arguments and environment variables

#### Arguments to `clip_benchmark eval`

* `--model_type`: Defaults to `open_clip` PyTorch reference implementation. Set to `deepsparse_clip` for DeepSparse and `onnx_clip` for ONNXRuntime
* `--dataset`: Defaults to `cifar10`. `imagenet1k` is easy to run as well. Full list is [here](https://github.com/neuralmagic/CLIP_benchmark/blob/main/benchmark/datasets.txt)

#### Environment variables for chosing model for inference

* `VISUAL_MODEL`: Defaults to `visual.onnx`. Used to change to a custom or sparsified model present in the same cache directory pulled from OpenCLIP. For instance `VISUAL_MODEL=visual-fp32-50sparse.onnx`
* `TEXTUAL_MODEL`: Defaults to `visual.onnx`. Used to change to a custom or sparsified model present in the same cache directory pulled from OpenCLIP. For instance `VISUAL_MODEL=visual-fp32-50sparse.onnx`

#### Sparsify

* `SPARSIFY`: Defaults to `None` aka don't sparsify. Set to anything in order to sparsify the used textual and visual sides. Look at `def sparsify_model` in `deepsparse_clip.py` to change the sweep of generated models
* `MAX_SAMPLES`: Defaults to `500`. Controls the number of samples to collect from the evaulation process in order to calibration sparsify

## Command Examples

Run evaluation for ViT-B-16-plus-240::laion400m_e32 on CIFAR10 using reference OpenCLIP PyTorch implementation
```
clip_benchmark eval --dataset=cifar10 --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-16-plus-240 --output=result.json --batch_size=64 --model_type open_clip
```

Run evaluation for ViT-B-16-plus-240::laion400m_e32 on CIFAR10 using reference OpenCLIP PyTorch implementation
```
clip_benchmark eval --dataset=cifar10 --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-16-plus-240 --output=result.json --batch_size=64 --model_type open_clip
```

Run evaluation for ViT-B-16-plus-240::laion400m_e32 on ImageNet1k using base FP32 ONNX model
```
clip_benchmark eval --dataset=imagenet1k --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-16-plus-240 --output=result.json --batch_size=64 --model_type onnx_clip
```

Run evaluation for ViT-B-16-plus-240::laion400m_e32 on ImageNet1k using base FP32 ONNX model, then produce sparsified models from the first 1000 collected samples. Control what models are produced by editing `def sparsify_model`
```
SPARSIFY=1 MAX_SAMPLES=1000 clip_benchmark eval --dataset=imagenet1k --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-16-plus-240 --output=result.json --batch_size=64 --model_type onnx_clip
```

Run evaluation for ViT-B-16-plus-240::laion400m_e32 on ImageNet1k using previously exported 40% Pruned FP32 ONNX models
```
VISUAL_MODEL=visual-fp32-40sparse.onnx TEXTUAL_MODEL=textual-fp32-40sparse.onnx clip_benchmark eval --dataset=imagenet1k --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-16-plus-240 --output=result.json --batch_size=64 --model_type onnx_clip
```