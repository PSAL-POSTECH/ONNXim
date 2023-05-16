"""
ONNX Test File generator 
"""

import torch

N = 1
H = 1
W = 1
C = 512

M = 1000
R = 1
S = 1

class TestModel(torch.nn.Module):
  def __init__(self) -> None:
    super(TestModel, self).__init__()
    #self.conv = torch.nn.Conv2d(C, M, R, padding=3, stride=2)
    self.fc = torch.nn.Linear(512, 1000)

  def forward(self, x):
    x = torch.flatten(x, 1)
    return self.fc(x)


input = torch.randn(N, C, H, W, requires_grad=True)

input_shape = (C, H, W) 

model = TestModel()


torch.onnx.export(
  model,
  input,
  'resnet18_fc.onnx',
  export_params = True,
  input_names = ['input'],
  output_names = ['output'],
  dynamic_axes = {
    'input' : {0 : 'batch_size'},
    'output' : {0 : 'batch_size'}}
)