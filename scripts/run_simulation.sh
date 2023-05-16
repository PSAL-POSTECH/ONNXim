#!/bin/bash
#Script for simulatioon
SCRIPT_DIR=`dirname $0`
SIMULTATOR=`realpath "${SCRIPT_DIR}/../build/bin/Simulator"`
# PRECOMMAND="gdb --args"
PRECOMMAND="srun -o log.out"
MODE=one_model

run_simulator() {
  pushd $1
  echo "Run Simulation"
  $PRECOMMAND $SIMULTATOR --config $CONFIG_PATH --model $MODEL_PATH --input_name "input" --log_level debug --mode $MODE &
  popd
}
#Argumenbt parser 
while (( "$#" )); do
  case "$1" in 
    -m| --model)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then 
        MODEL_PATH=$2
        shift 2
      else 
        echo "Error: Arcument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -c| --config)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then 
        CONFIG_PATH=$2
         shift 2
      else 
        echo "Error: Arcument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -t| --two)
      MODE=two_model
      shift 1
      ;;
    -h| --help) 
      echo "Usage:  $0 -i <input> [options]" >&2
          echo "        -m | --model  %  (set input model for simulation)" >&2
          echo "        -c | --config  %  (set configuration for simulation)" >&2
          exit 0
      ;;
  esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$CONFIG_PATH" ]; then
  echo "Error: --model and --config option must be set" >&2
  exit 1
fi
CONFIG_PATH=`realpath $CONFIG_PATH`

MODEL_TMP=$MODEL_PATH
unset MODEL_PATH
unset MODEL_NAME
for model in ${MODEL_TMP//,/$'\n'}; do 
  model=`realpath $model`
  model_name=`basename $model`
  model_name=${model_name%.*}
  if [[ -n $MODEL_PATH ]]; then
    MODEL_PATH="${MODEL_PATH},"
    MODEL_NAME="${MODEL_NAME}-"
  fi
  MODEL_PATH="${MODEL_PATH}${model}"
  MODEL_NAME="${MODEL_NAME}${model_name}"
done

#Make simulation workspace
CURRENTDATE=`date +"%Y-%m-%d"`
CURRENTTIME=`date +"%H-%M"`
CONFIG_NAME=`basename $CONFIG_PATH`
# MODEL_NAME=`basename $MODEL_PATH`
echo $MODEL_PATH
echo ./workspace/$CURRENTDATE/${MODEL_NAME%.*}/${CONFIG_NAME%.*}-$CURRENTTIME
WORKSPACE=./workspace/$CURRENTDATE/${MODEL_NAME%.*}/${CONFIG_NAME%.*}-$CURRENTTIME
mkdir -p $WORKSPACE
run_simulator $WORKSPACE