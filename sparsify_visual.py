MODEL_NAME = "visual_export.onnx"
NUM_CALIBRATION_SAMPLES = 10000

import torch
import os, open_clip, torchvision, argparse
import numpy as np
from clip_server.model.pretrained_models import download_model

import onnx, os
from sparsifyml.one_shot import sparsify_fast

def sparsify_model(model_path, sample_inputs, quant=False, quantization_algo="MinMax", sparsity=None, sparsity_algo="FastOBCQ", str_append=None):
    assert(quantization_algo == "MinMax" or quantization_algo=="OBQ" or quantization_algo=="FastOBCQ")
    assert(sparsity_algo == "FastOBCQ" or sparsity_algo == "OBC")

    model = onnx.load(model_path)
    IGNORE_LIST = [node.op_type for node in model.graph.node if node.op_type not in ["MatMul", "Gemm"]]

    if quant and quantization_algo == "MinMax":
        qstr = "-int8-minmax"
    elif quant and quantization_algo == "OBQ":
        qstr = "-int8-obq"
    elif quant and quantization_algo == "FastOBCQ":
        qstr = "-int8-fastobcq"
    else:
        assert not quant
        qstr = "-fp32"
    
    if sparsity and sparsity_algo == "FastOBCQ":
        sstr = f"-{int(sparsity*100)}sparse-fastobcq" if sparsity else "-dense"
    elif sparsity and sparsity_algo == "OBC":
        sstr = f"-{int(sparsity*100)}sparse-obc" if sparsity else "-dense"
    else:
        assert not sparsity
        sstr = "-dense"
    
    filename, file_extention = os.path.splitext(model_path)
    if str_append is None:
        new_model_path = filename + f"{qstr}{sstr}" + file_extention
    else:
        new_model_path = filename + f"{qstr}{sstr}{str_append}" + file_extention

    model = sparsify_fast(
        model=model_path,
        sample_input=sample_inputs,
        batches=len(sample_inputs),
        sparsity=sparsity,
        sparsity_algo=sparsity_algo,
        quantization=dict(ignore=IGNORE_LIST) if quant else False,
        quantization_algo=quantization_algo,
        save_path=new_model_path
    )

    print(f"Sparsified model at {new_model_path}")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=MODEL_NAME)
parser.add_argument('--sparsity', type=float, default=None)
parser.add_argument('--sparsity_algo', type=str, default="FastOBCQ")
parser.add_argument('--quantization', action='store_true')
parser.add_argument('--quantization_algo', type=str, default="FastOBCQ")
parser.add_argument('--str_append', type=str, default=None)

_S3_BUCKET_V2 = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/'

_MODELS = {
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
}

def download(name, target_folder):
    textual_data =  _MODELS[name][0]
    visual_data =  _MODELS[name][1]

    textual_path = download_branch(textual_data, target_folder)
    visual_path = download_branch(visual_data, target_folder)

    return textual_path, visual_path

def download_branch(data, target_folder):
    name, md5 = data
    url = _S3_BUCKET_V2 + name

    return download_model(
        url=url,
        target_folder=target_folder,
        md5sum=md5,
        with_resume=True
    )

def main(model_path, quantization=False, sparsity=None, quantization_algo="FastOBCQ", sparsity_algo="FastOBCQ", str_append=None):
    
    # base_dir = "datasets/imagenet"
    base_dir = "/home/ubuntu/datasets/VOC_ex"
    model_name = "ViT-B-16-plus-240"
    pretrained = "laion400m_e32"

    name = f"{model_name}::{pretrained}"
    download_dir = os.path.expanduser(
        f'models-rs/{name.replace("/", "-").replace("::", "-")}'
    )

    if os.path.isdir(download_dir):
        visual_path = os.path.join(download_dir, model_path)
    else:
        _, visual_path = download(name, download_dir)

    print(f"VISUAL_PATH = {visual_path}")

    _, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    dataset = torchvision.datasets.ImageFolder(
        root=base_dir,
        transform=transform,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, sampler=None)

    visual_samples = []
    i = 0
    for batch in iter(dataloader):
        im = batch[0][0]
        visual_samples.append(np.array(im.cpu()))
        i+=1
        if i > NUM_CALIBRATION_SAMPLES:
            break
        

    sparsify_model(visual_path, 
                   visual_samples, 
                   quant=quantization, 
                   sparsity=sparsity, 
                   quantization_algo=quantization_algo,
                   sparsity_algo=sparsity_algo,
                   str_append=str_append)

if __name__ == "__main__":
    args = parser.parse_args()
    
    main(model_path=args.model_path,
         quantization=args.quantization, 
         sparsity=args.sparsity, 
         quantization_algo=args.quantization_algo, 
         sparsity_algo=args.sparsity_algo,
         str_append=args.str_append)