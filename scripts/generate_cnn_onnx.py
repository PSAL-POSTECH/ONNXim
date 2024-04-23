"""
ONNX File generator 
Optimizer onnx graph for inference
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import torch
import torchvision.models as models
import argparse
import pathlib
import os
import json

size_list = [1, 2, 4, 8, 16, 32]

HOME = os.getenv("ONNXIM_HOME", default="../")
parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--model', required=True, help="resnet18, resnet50, alexnet, vgg16, inception")
parser.add_argument('--weight', type=int, default=1, help="export weight, defulat=True")
args = parser.parse_args()

torchvision_models = {
  'resnet18' : models.resnet18(),
  'resnet50' : models.resnet50(),
  'alexnet' : models.alexnet(),
  'vgg16' :  models.vgg16(),
  'squeezenet' : models.squeezenet1_0(),
  'densenet' :  models.densenet161(),
  'inception' : models.inception_v3(),
  'googlenet' : models.googlenet(),
  'shufflenet' : models.shufflenet_v2_x1_0(),
  'mobilenet' : models.mobilenet_v2(),
  'resnext50_32x4d' : models.resnext50_32x4d(),
  'wide_resnet50_2' : models.wide_resnet50_2(),
  'mnasnet' :  models.mnasnet1_0(),
}

model = torchvision_models[args.model]
batch_size = 1
if args.model != 'inception':
  input = torch.randn(1, 3, 224, 224, requires_grad=True)
  input_shape = (3, 224, 224)
else:
  input = torch.randn(1, 3, 299, 299, requires_grad=True)
  input_shape = (3, 299, 299)

# Export PyTorch model to onnx
torch.onnx.export(
  model,
  input,
  'tmp.onnx',
  export_params = bool(args.weight),
  input_names = ['input'],
  output_names = ['output'],
  dynamic_axes = {
    'input' : {0 : 'batch_size'},
    'output' : {0 : 'batch_size'}}
)

# Create output folder
pathlib.Path(f'{HOME}/models/{args.model}/').mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)

# Optimzied exported onnx file
print(f"Converting ONNX FILE: {args.model}")
opt = rt.SessionOptions()
opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
opt.optimized_model_filepath = f'{HOME}/models/{args.model}/{args.model}.onnx'
sess = rt.InferenceSession('tmp.onnx', sess_options=opt)

# Generate model_list json file
for size in size_list:
    config = {
        "models": [
                {
                    "name": f"{args.model}",
                    "batch_size": size,
                    "request_time": 0
                 }
            ]
        }
    with open(f"{HOME}/model_lists/{args.model}_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
print("DONE")