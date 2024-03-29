"""
ONNX File generator 
Optimizer onnx graph for inference
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import torch
import torchvision.models as models
# import pytorch2timeloop
import argparse
import pathlib
import os
import json

size_list = [1, 2, 4, 8, 16, 32]

HOME = os.getenv("ONNXIM_HOME", default="../")
parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--models')
parser.add_argument('--weight', type=int, default=1)
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

model_list = args.models.split(',')
for model_name in model_list:
  model = torchvision_models[model_name]
  batch_size = 1
  if model_name != 'inception':
    input = torch.randn(1, 3, 224, 224, requires_grad=True)
    input_shape = (3, 224, 224)
  else:
    input = torch.randn(1, 3, 299, 299, requires_grad=True)
    input_shape = (3, 299, 299)

  top_dir = os.path.join(HOME, "models")
  convert_fc = True
  exception_module_names = []

  # pytorch2timeloop.convert_model(model, input_shape, batch_size, args.model, top_dir, convert_fc, exception_module_names) 

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

  opt = rt.SessionOptions()
  # enable level 3 optimizations
  print(f"Converting ONNX FILE: {model_name}")
  pathlib.Path(f'{HOME}/models/{model_name}/').mkdir(parents=True, exist_ok=True)
  opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
  opt.optimized_model_filepath = f'{HOME}/models/{model_name}/{model_name}.onnx'
  sess = rt.InferenceSession('tmp.onnx', sess_options=opt)

pathlib.Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
config = {
    "models": [
        ]
    }
for model_name in model_list:
  config["models"].append(
    {
      "name": f"{model_name}",
      "batch_size": 1,
    }
  )

file_name = '_'.join(model_list)

with open(f"{HOME}/model_lists/{file_name}.json", "w") as json_file:
    json.dump(config, json_file, indent=4)
print("DONE")