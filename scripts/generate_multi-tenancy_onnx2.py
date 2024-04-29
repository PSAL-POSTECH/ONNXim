import argparse
import pathlib
import os
import json

HOME = os.getenv("ONNXIM_HOME", default="../")
parser = argparse.ArgumentParser(prog = 'ONNX generator')
# Interval of DNN inference request
parser.add_argument('--resnet_ms', default=0)
parser.add_argument('--gpts_ms', default=0)
parser.add_argument('--gptg_ms', default=0)
parser.add_argument('--bert_ms', default=0)
parser.add_argument('--total_ms')

# Partition id of DNN model
# Note: Use this option, when you want to allocate a specific DNN model to a specific core group (partition)
parser.add_argument('--resnet_p', default=0)
parser.add_argument('--gpts_p', default=0)
parser.add_argument('--gptg_p', default=0)
parser.add_argument('--bert_p', default=0)
parser.add_argument('--resnet_b', default=1)
parser.add_argument('--gpts_b', default=1)
parser.add_argument('--gptg_b', default=1)
parser.add_argument('--bert_b', default=1)

parser.add_argument('--gpt2_version', default="gpt2")
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
resnet_batch = int(args.resnet_b)
gptg_batch = int(args.gptg_b)
gpts_batch = int(args.gpts_b)
bert_batch = int(args.bert_b)
gpt2_version = args.gpt2_version

pathlib.Path(f"{HOME}/model_lists").mkdir(parents=True, exist_ok=True)
config = {
  "models": [
  ]
}
 
if resnet_ms:
  for i in range(0, total_ms, resnet_ms):
    model_config = {
      "name": "resnet50",
      "batch_size": resnet_batch,
      "request_time": i,
      "partition_id": resnet_partition
    }
    config["models"].append(model_config)

if gptg_ms:
  for i in range(0, total_ms, gptg_ms):
    model_config = {
      "name": f"{gpt2_version}",
      "batch_size": gptg_batch,
      "nr_atten": -1,   # Number of attention block to simulate. -1 means all layers
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
      "name": f"{gpt2_version}",
      "batch_size": gpts_batch,
      "nr_atten": -1,   # Number of attention block to simulate. -1 means all layers
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
      "batch_size": bert_batch,
      "nr_atten": -1,   # Number of attention block to simulate. -1 means all layers
      "sequence_length": 1024,
      "seq_len": 1024,
      "past_seq_len": 0,
      "total_seq_len": 1024,
      "request_time": i,
      "partition_id": bert_partition
    }
    config["models"].append(model_config)

with open(f"{HOME}/model_lists/multi_{resnet_ms}_{gpts_ms}_{gptg_ms}_{bert_ms}_{total_ms}_"\
          f"{resnet_batch}{gpts_batch}{gptg_batch}{bert_batch}_"
          f"{resnet_partition}{gpts_partition}{gptg_partition}{bert_partition}.json", "w") as json_file:
  json.dump(config, json_file, indent=4)
print("DONE")