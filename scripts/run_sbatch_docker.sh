#!/bin/bash
CONFIG_FILE=$1
MODEL_FILE=$2
BATCH=$3

EXPR_DIR=results/$MODEL_FILE/$BATCH/$CONFIG_FILE
mkdir -p $EXPR_DIR
pushd $EXPR_DIR
echo "#!/bin/bash" > slurm.sh
echo "#SBATCH --job-name=$INPUDATA_WIDHOUT_EXT" >> slurm.sh
echo "#SBATCH --output=slurm-%j.out" >> slurm.sh
echo "#SBATCH --error=slurm-%j.err" >> slurm.sh
echo "#SBATCH --nodelist=n10" >> slurm.sh
echo "docker start onnxim-ubuntu" >> slurm.sh
echo "docker exec onnxim-ubuntu bash /home/workspace/scripts/run_docker_template.sh $CONFIG_FILE $MODEL_FILE" >> slurm.sh
sbatch slurm.sh
popd