import argparse
import pathlib
import os
import json

HOME = os.getenv("ONNXIM_HOME", default="../")
parser = argparse.ArgumentParser(prog = 'ONNX generator')
parser.add_argument('--resnet_ms', default=0)
parser.add_argument('--gpts_ms', default=0)
parser.add_argument('--gptg_ms', default=0)
parser.add_argument('--bert_ms', default=0)
parser.add_argument('--total_ms')

parser.add_argument('--resnet_p', default=0)
parser.add_argument('--gpts_p', default=0)
parser.add_argument('--gptg_p', default=0)
parser.add_argument('--bert_p', default=0)
args = parser.parse_args()

resnet_ms = int(args.resnet_ms)
gptg_ms = int(args.gptg_ms)
gpts_ms = int(args.gpts_ms)
bert_ms = int(args.bert_ms)
total_ms = int(args.total_ms)
resnet_partition = int(args.resnet_p)
gptg_partition = int(args.gptg_p)
gpts_partition = int(args.gpts_p)
bert_partition = int(args.bert_p)

pathlib.Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
config = {
  "models": [
  ]
}
 
if resnet_ms:
  for i in range(0, total_ms, resnet_ms):
    model_config = {
      "name": "resnet50",
      "batch_size": 1,
      "request_time": i,
      "partition_id": resnet_partition
    }
    config["models"].append(model_config)

if gptg_ms:
  for i in range(0, total_ms, gptg_ms):
    model_config = {
      "name": "gpt2",
      "batch_size": 1,
      "nr_atten": -1,
      "sequence_length": 1,
      "seq_len": 1,
      "past_seq_len": 1023,
      "total_seq_len": 1024,
      "request_time": i,
      "partition_id": gptg_partition
    }
    config["models"].append(model_config)

if gpts_ms:
  for i in range(0, total_ms, gpts_ms):
    model_config = {
      "name": "gpt2",
      "batch_size": 1,
      "nr_atten": -1,
      "sequence_length": 1024,
      "seq_len": 1024,
      "past_seq_len": 0,
      "total_seq_len": 1024,
      "request_time": i,
      "partition_id": gpts_partition
    }
    config["models"].append(model_config)

if bert_ms: 
  for i in range(0, total_ms, bert_ms):
    model_config = {
      "name": "bert",
      "batch_size": 1,
      "nr_atten": -1,
      "sequence_length": 1024,
      "seq_len": 1024,
      "past_seq_len": 0,
      "total_seq_len": 1024,
      "request_time": i,
      "partition_id": bert_partition
    }
    config["models"].append(model_config)

with open(f"{HOME}/model_lists/multi_{resnet_ms}_{gpts_ms}_{gptg_ms}_{bert_ms}_{total_ms}_{resnet_partition}{gpts_partition}{gptg_partition}{bert_partition}.json", "w") as json_file:
  json.dump(config, json_file, indent=4)
print("DONE")