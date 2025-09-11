######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

export TORCH_NCCL_BLOCKING_WAIT=1

######## Set Experiment Configuration ###########
exp_config="$exp_dir/tascodec_6_25hz_16384_bsq.json"
exp_name="tascodec_6_25hz_16384_bsq"

######## Train Model ###########
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port 2001 \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    --resume \
    --dataloader_seed 1000000 \