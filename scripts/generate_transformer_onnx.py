import json
from onnxruntime.transformers import optimizer
from optimum.onnxruntime import ORTModelForQuestionAnswering
import argparse
import pathlib
import os
HOME = os.getenv("ONNXIM_HOME", default="../")

size_list = [1, 2, 4, 8, 16, 32]

parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--model', required=True, help="support gpt2, gpt2-medium, gpt2-large, gpt2-xl, bert")
args = parser.parse_args()
if "gpt2" in args.model:
    onnx_path = pathlib.Path(f"{args.model}.onnx")
    if not onnx_path.is_file():
        os.system(f"python3.8 -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m {args.model} --model_class GPT2LMHeadModel -t 1 -r 1 --output {args.model}.onnx -p fp32")
    if args.model == "gpt2":
        optimized_model = optimizer.optimize_model(f"{args.model}.onnx", model_type="gpt2", num_heads=12, hidden_size=768)
    elif args.model == "gpt2-medium":
        optimized_model = optimizer.optimize_model(f"{args.model}.onnx", model_type="gpt2", num_heads=16, hidden_size=1024)
    elif args.model == "gpt2-large":
        optimized_model = optimizer.optimize_model(f"{args.model}.onnx", model_type="gpt2", num_heads=20, hidden_size=1280)
    elif args.model == "gpt2-xl":
        optimized_model = optimizer.optimize_model(f"{args.model}.onnx", model_type="gpt2", num_heads=25, hidden_size=1600)
elif args.model == "bert":
    onnx_path = pathlib.Path(f"bert/model.onnx")
    if not onnx_path.is_file():
        model = ORTModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", export=True, provider="CPUExecutionProvider")
        model.save_pretrained("bert")
    optimized_model = optimizer.optimize_model(f"bert/model.onnx", model_type=args.model)
else:
    print("Only gpt2, bert are supported...!")
    exit(1)

# Create output folder
pathlib.Path(f'{HOME}/models/{args.model}/').mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)

# Check attention node
nodes = optimized_model.graph().node
node_types = [node.op_type for node in nodes]
if "Attention" not in node_types:
    print("Opimizing model failed...")
    exit(1)

# Save optimized onnx file
optimized_model.save_model_to_file(f'{HOME}/models/{args.model}/{args.model}.onnx', use_external_data_format=True)


# Generate model_list json file
if "gpt2" in args.model:
    for size in size_list:
        # GPT2 summarize json
        config = {
            "models": [
                    {
                        "name": f"{args.model}",
                        "batch_size": size,
                        "nr_atten": -1,
                        "sequence_length": 1024,
                        "seq_len": 1024,
                        "past_seq_len": 0,
                        "total_seq_len": 1024,
                        "output_seq_len": 1025,
                        "request_time": 0
                    }
                ]
            }
        with open(f"{HOME}/model_lists/{args.model}_s_{size}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)

        # GPT2 generation
        config = {
            "models": [
                    {
                        "name": f"{args.model}",
                        "batch_size": size,
                        "nr_atten": -1,
                        "sequence_length": 1,
                        "seq_len": 1,
                        "past_seq_len": 1024,
                        "total_seq_len": 1025,
                        "output_seq_len": 1125,
                        "request_time": 0
                    }
                ]
            }

        with open(f"{HOME}/model_lists/{args.model}_g_{size}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)

if "bert" in args.model:
    for size in size_list:
        # BERT
        config = {
            "models": [
                    {
                        "name": f"{args.model}",
                        "batch_size": size,
                        "nr_atten": -1,
                        "sequence_length": 1024,
                        "seq_len": 1024,
                        "past_seq_len": 0,
                        "total_seq_len": 1025
                    }
                ]
            }
        with open(f"{HOME}/model_lists/bert_{size}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
print("DONE")