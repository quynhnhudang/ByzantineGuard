#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=v10_cifar_PGD
#SBATCH --partition=Quick
#SBATCH --mem-per-cpu=16GB 
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --array=1

#module purge
# module add apps/cuda/11.3.1

source $HOME/.bashrc
# conda activate tf_21

conda activate torch_13 #Gaivi

#module add apps/cuda/11.3.1
# echo $DISPLAY

# export DISPLAY=$(hostname)$DISPLAY

# echo "DISPLAY=$DISPLAY"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"

# Specify the path to the config file
config=/storage2-mnt/data/longdang/FedShare/config_FedAvg_AT_Eval.txt

model=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

num_users=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

rounds=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

local_ep=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

filepath=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)

sampling=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)


if echo "$sampling" | grep -q '^(?!.*\?)[a-zA-Z0-9_]*$'; then
  echo "Sampling variable is valid: $sampling"
else
  echo "Sampling variable contains special characters, including '?': $sampling"
  # sampling=$(echo "$sampling" | sed 's/\\?//g')
  # sampling=$(echo "$sampling" | tr -d '?')
  # sampling=$(echo "$sampling" | tr -d '\?')
  sampling=$(echo "$sampling" | tr -dc '[:alnum:]_')
  echo "Sampling variable with '?' removed: $sampling"
fi

if echo "$sampling" | grep -q '^(?!.*\?)[a-zA-Z0-9_]*$'; then
  echo "Sampling variable is valid: $sampling"
else
  echo "2nd-Sampling variable contains special characters, including '?': $sampling"
  # exit 1
fi

fed='fedavg'
# rounds=1
# num_users=1 #1 user
frac=1
# local_ep=1 #increase next or K in the paper
local_bs=128
bs=100
lr=0.001
global_lr=1 #SCAFFOLD 
momentum=0.9
classwise=1000 #maximum sharing 10,000 images.
alpha=0.5
l2_lambda=0.0002
##########################
dataset='cifar'
# model=${model}
# sampling=${sampling}
num_classes=10
num_channels=3
gpu=0
verbose=False
seed=123
all_clients=True
sys_homo=True
debug=False
soft_label_clean=0.95
mean=0
sigma=0.1
rho=0.5
#PGD
eps=0.0314
nb_iter=7
eps_iter=0.00784
clip_min=0.0
clip_max=1.0
#FGSM
eps_FGSM=0.031
pretrained=False

filename=AT_out_${fed}_${eps_FGSM}_${dataset}_${model}_${rounds}_${local_ep}_nParties_${sampling}_${num_users}_eval_v11
error_filename=AT_error_${fed}_${eps_FGSM}_${dataset}_${model}_${rounds}_${local_ep}_nParties_${sampling}_${num_users}_eval_v11

echo "filename: ${filename}"
echo "error_filename: ${error_filename}"

if [ ! -f ${filename} ]; then
    echo "${filename} does not exist."
    echo "The results of experiments" > ${filename}
fi
echo "------------------ " >> ${filename}
echo "New experiment:" >> ${filename}
echo "Start time `date` " >> ${filename}

echo "rounds: ${rounds}" >> ${filename}
echo "num_users: ${num_users}" >> ${filename}
echo "fed: ${fed}" >> ${filename}
echo "model: ${model}" >> ${filename}
echo "local_ep: ${local_ep}" >> ${filename}
echo "local_bs: ${local_bs}" >> ${filename}
echo "alpha: ${alpha}" >> ${filename}
echo "sampling: ${sampling}" >> ${filename}
echo "eps_FGSM: ${eps_FGSM}" >> ${filename}
echo "classwise: ${classwise}" >> ${filename}
echo "alpha: ${alpha}" >> ${filename}
echo "pretrained: ${pretrained}" >> ${filename}
echo "filepath: ${filepath}" >> ${filename}

srun python FedShare_AT_v11_evaluation.py \
    --fed ${fed} \
    --rounds ${rounds} \
    --num_users ${num_users} \
    --frac 1 \
    --fed ${fed} \
    --local_ep ${local_ep} \
    --local_bs ${local_bs} \
    --bs ${bs} \
    --lr ${lr} \
    --momentum ${momentum} \
    --l2_lambda ${l2_lambda} \
    --classwise ${classwise} \
    --alpha ${alpha} \
    --model ${model} \
    --dataset cifar \
    --sampling ${sampling} \
    --num_classes 10 \
    --num_channels 3 \
    --gpu ${gpu} \
    --seed 123 \
    --alpha ${alpha} \
    --verbose \
    --all_clients \
    --sys_homo \
    --debug \
    --global_lr 1 \
    --soft_label_clean ${soft_label_clean} \
    --mean ${mean} \
    --sigma ${sigma} \
    --rho ${rho} \
    --eps ${eps} \
    --nb_iter ${nb_iter} \
    --eps_iter ${eps_iter} \
    --clip_min ${clip_min} \
    --clip_max ${clip_max} \
    --eps_FGSM ${eps_FGSM} \
    --filepath ${filepath} \
     2>> ${error_filename} \
              1>> ${filename}
    # --sys_homo \

conda deactivate


#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short
#SBATCH --partition=general #Gaivi