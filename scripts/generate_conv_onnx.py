import torch
from pathlib import Path
import json
import os

size_list = [128]#64, 256, 1024]
dtype = torch.float32
C_in = 128
C_out = 128
K_sz = 3
padding = 1
H = 14 * 4
W = 14 * 4
stride=2
HOME = os.getenv("ONNXIM_HOME", default="../")

size_name = f"{C_in}_{C_out}_{K_sz}_{H}_{W}"
# Test Convolution model
class size_conv(torch.nn.Module):
    def __init__(self, C_in, C_out, K_sz, padding=padding):
        super().__init__()
        self.fc = torch.nn.Conv2d(C_in, C_out, K_sz, stride=stride, padding=padding, bias=False, dtype=dtype)

    def forward(self, x):
        return self.fc(x)

# Create output folder
Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
for size in size_list:

    # Export PyTorch model to onnx
    Path(f"{HOME}/models/conv_{size_name}").mkdir(parents=True, exist_ok=True)
    m = size_conv(C_in, C_out, K_sz, padding)
    A = torch.zeros([1,C_in, H, W], dtype=dtype)
    onnx_path = Path(f"{HOME}/models/conv_{size_name}/conv_{size_name}.onnx")
    torch.onnx.export(m, A, onnx_path, export_params=True, input_names = ['input'], output_names=['output'])

    # Generate model_list json file
    config = {
        "models": [
            {
                "name": f"conv_{size_name}",
                "request_time": 0
            }
        ]
    }
    with open(f"{HOME}/model_lists/conv_{size_name}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
