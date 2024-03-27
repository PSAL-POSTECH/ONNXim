import torch
from pathlib import Path
import json

size_list = [64, 256, 1024]
dtype = torch.float32

class size_conv(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = torch.nn.Conv2d(64, size, 3, padding=1, bias=False, dtype=dtype)

    def forward(self, x):
        return self.fc(x)

Path(f"../conv_model_lists").mkdir(parents=True, exist_ok=True)
for size in size_list:
    Path(f"../models/conv_{size}").mkdir(parents=True, exist_ok=True)
    m = size_conv(size)
    A = torch.zeros([1,64, 14, 14], dtype=dtype)
    torch.onnx.export(m, A, f"../models/conv_{size}/conv_{size}.onnx", export_params=True, input_names = ['input'], output_names=['output'])
    config = {"models": [{"name": f"conv_{size}"}]}
    with open(f"../conv_model_lists/conv_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
