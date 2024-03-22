import onnxruntime as rt
from onnxruntime.transformers import optimizer
# import pytorch2timeloop
import argparse
import pathlib

import os

parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--model')
args = parser.parse_args()
pathlib.Path(f'../models/{args.model}/').mkdir(parents=True, exist_ok=True)
if args.model == "gpt2":
    os.system(f"python -m onnxruntime.transformers.models.{args.model}.convert_to_onnx -m {args.model} --model_class GPT2LMHeadModel -t 1 -r 1 --output {args.model}.onnx -p fp32")
    optimized_model = optimizer.optimize_model(f"{args.model}.onnx", model_type=args.model, num_heads=12, hidden_size=768)
elif args.model == "bert":
    os.system(f"python -m onnxruntime.transformers.models.bert.eval_squad --provider CPUExecutionProvider")
    optimized_model = optimizer.optimize_model(f"bert-large-uncased-whole-word-masking-finetuned-squad/model.onnx", model_type=args.model, num_heads=12, hidden_size=768)
optimized_model.save_model_to_file(f'../models/{args.model}/{args.model}.onnx')