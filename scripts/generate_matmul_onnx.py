import torch
from pathlib import Path
import json
import os

#size_list = [[512, 768, 2304],[512, 768, 512],[512, 768, 768], [512, 512, 768], [512, 768, 50257]]#32, 64, 128, 256, 512, 1024, 2048]
#size_list = [[512, 512, 1024],[512, 1024, 2],[512, 1024, 512], [512, 1024, 1024], [512, 1024, 3072], [512, 768, 3072], [512, 1024, 4096], [512, 4096, 1024]]#32, 64, 128, 256, 512, 1024, 2048]
size_list = [[1, 1024*8, 1024*8]] #[32,32,32], [64,64,64],[128]*3, [256]*3, [512]*3, [1024]*3, [2048]*3, [4096]*3, [8192]*3]
dtype = torch.float16

HOME = os.getenv("ONNXIM_HOME", default="../")

# Test matmul model
class size_matmul(torch.nn.Module):
    def __init__(self, size2, size3):
        super().__init__()
        self.fc = torch.nn.Linear(size2, size3, dtype=dtype, bias=False) #size, size, dtype=dtype, bias=False)

    def forward(self, x):
        return self.fc(x)

# Create output folder
Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
for size1, size2, size3 in size_list:
    # Export PyTorch model to onnx
    Path(f"{HOME}/models/matmul_{size1}_{size2}_{size3}").mkdir(parents=True, exist_ok=True)
    m = size_matmul(size2, size3)
    A = torch.zeros([size1, size2], dtype=dtype)
    onnx_path = Path(f"{HOME}/models/matmul_{size1}_{size2}_{size3}/matmul_{size1}_{size2}_{size3}.onnx")
    torch.onnx.export(m, A, onnx_path, export_params=True, input_names = ['input'], output_names=['output'])

    # Generate model_list json file
    config = {
        "models": [
            {
                "name": f"matmul_{size1}_{size2}_{size3}",
                "request_time": 0
            }
        ]
    }
    with open(f"{HOME}/model_lists/matmul_{size1}_{size2}_{size3}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
