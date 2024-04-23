import torch
from pathlib import Path
import json
import os

size_list = [64, 256, 1024]
dtype = torch.float32

HOME = os.getenv("ONNXIM_HOME", default="../")

# Test Convolution model
class size_conv(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = torch.nn.Conv2d(64, size, 3, padding=1, bias=False, dtype=dtype)

    def forward(self, x):
        return self.fc(x)

# Create output folder
Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
for size in size_list:
    # Export PyTorch model to onnx
    Path(f"{HOME}/models/conv_{size}").mkdir(parents=True, exist_ok=True)
    m = size_conv(size)
    A = torch.zeros([1,64, 14, 14], dtype=dtype)
    onnx_path = Path(f"{HOME}/models/conv_{size}/conv_{size}.onnx")
    if not onnx_path.is_file():
        torch.onnx.export(m, A, onnx_path, export_params=True, input_names = ['input'], output_names=['output'])

    # Generate model_list json file
    config = {
        "models": [
            {
                "name": f"conv_{size}",
                "request_time": 0
            }
        ]
    }
    with open(f"{HOME}/model_lists/conv_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
