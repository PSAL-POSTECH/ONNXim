import json
from onnxruntime.transformers import optimizer
from optimum.onnxruntime import ORTModelForQuestionAnswering
import argparse
import pathlib
import os
HOME = os.getenv("ONNXIM_HOME", default="../")

size_list = [1, 2, 4, 8, 16, 32]

parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--model')
args = parser.parse_args()
if args.model == "gpt2":
    onnx_path = pathlib.Path(f"{args.model}.onnx")
    if not onnx_path.is_file():
        os.system(f"python -m onnxruntime.transformers.models.{args.model}.convert_to_onnx -m {args.model} --model_class GPT2LMHeadModel -t 1 -r 1 --output {args.model}.onnx -p fp32")
    optimized_model = optimizer.optimize_model(f"{args.model}.onnx", model_type=args.model, num_heads=12, hidden_size=768)
elif args.model == "bert":
    onnx_path = pathlib.Path(f"bert/model.onnx")
    if not onnx_path.is_file():
        model = ORTModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", export=True, provider="CPUExecutionProvider")
        model.save_pretrained("bert")
    optimized_model = optimizer.optimize_model(f"bert/model.onnx", model_type=args.model, num_heads=12, hidden_size=768)
else:
    print("Only gpt2, bert are supported...!")
    exit(1)

pathlib.Path(f'{HOME}/models/{args.model}/').mkdir(parents=True, exist_ok=True)
optimized_model.save_model_to_file(f'{HOME}/models/{args.model}/{args.model}.onnx')

pathlib.Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
for size in size_list:
    # GPT2 summarize json
    config = {
        "models": [
                {
                    "name": f"gpt2",
                    "batch_size": size,
                    "nr_atten": -1,
                    "sequence_length": 1024,
                    "seq_len": 1024,
                    "past_seq_len": 0,
                    "total_seq_len": 1024,
                    "request_time": 0
                 }
            ]
        }
    with open(f"{HOME}/model_lists/gpt2_s_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)

    # GPT2 generation
    config = {
        "models": [
                {
                    "name": f"gpt2",
                    "batch_size": size,
                    "nr_atten": -1,
                    "sequence_length": 1,
                    "seq_len": 1,
                    "past_seq_len": 1023,
                    "total_seq_len": 1024,
                    "request_time": 0
                 }
            ]
        }

    with open(f"{HOME}/model_lists/gpt2_g_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)

    # BERT
    config = {
        "models": [
                {
                    "name": f"bert",
                    "batch_size": size,
                    "nr_atten": -1,
                    "sequence_length": 1024,
                    "seq_len": 1024,
                    "past_seq_len": 0,
                    "total_seq_len": 1024
                 }
            ]
        }
    with open(f"{HOME}/model_lists/bert_{size}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
print("DONE")