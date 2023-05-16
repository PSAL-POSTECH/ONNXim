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


parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--model')
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

top_dir = '../models'
convert_fc = True
exception_module_names = []

# pytorch2timeloop.convert_model(model, input_shape, batch_size, args.model, top_dir, convert_fc, exception_module_names) 

torch.onnx.export(
  model,
  input,
  'tmp.onnx',
  export_params = True,
  input_names = ['input'],
  output_names = ['output'],
  dynamic_axes = {
    'input' : {0 : 'batch_size'},
    'output' : {0 : 'batch_size'}}
)

opt = rt.SessionOptions()
# enable level 3 optimizations
print("Converting ONNX FILE")
opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opt.optimized_model_filepath = f'../models/{args.model}/{args.model}.onnx'
sess = rt.InferenceSession('tmp.onnx', opt)
print("DONE")
