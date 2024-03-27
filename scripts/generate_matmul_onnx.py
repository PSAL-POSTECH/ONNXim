import torch
from pathlib import Path
import json
import os

size_list = [32, 64, 128, 256, 512, 1024, 2048]
dtype = torch.float16

HOME = os.getenv("ONNXIM_HOME", default="../")

class size_matmul(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = torch.nn.Linear(size, size, dtype=dtype, bias=False)

    def forward(self, x):
        return self.fc(x)

Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
for size in size_list:
    Path(f"{HOME}/models/matmul_{size}").mkdir(parents=True, exist_ok=True)
    m = size_matmul(size)
    A = torch.zeros([size, size], dtype=dtype)
    onnx_path = Path(f"{HOME}/models/matmul_{size}/matmul_{size}.onnx")
    if not onnx_path.is_file():
        torch.onnx.export(m, A, onnx_path, export_params=True, input_names = ['input'], output_names=['output'])
    config = {"models": [{"name": f"matmul_{size}"}]}
    with open(f"{HOME}/model_lists/matmul_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
