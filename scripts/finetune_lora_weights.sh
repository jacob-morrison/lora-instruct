export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# DATASET_FILE=flan_v2/flan_v2_data.jsonl \
# DATASET=flan_v2

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=cot/cot_data.jsonl \
# DATASET=cot

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&


# DATASET_FILE=oasst1/oasst1_data.jsonl \
# DATASET=oasst1

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=lima/lima_data.jsonl \
# DATASET=lima

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=code_alpaca/code_alpaca_data.jsonl \
# DATASET=code_alpaca

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=sharegpt/sharegpt_data.jsonl \
# DATASET=sharegpt

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=wizardlm/wizardlm_data.jsonl \
# DATASET=wizardlm

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=open_orca/open_orca_data.jsonl \
# DATASET=open_orca

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/${DATASET}/ \
#     --logging_steps 1 # &&

# DATASET_FILE=tulu/tulu_v2_human_mix.jsonl
# DATASET=tulu

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/tulu_v2_human/ \
#     --logging_steps 1 # &&


# DATASET_FILE=tulu/tulu_v2_mix.jsonl
# DATASET=tulu

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/tulu_v2/ \
#     --logging_steps 1 # &&

DATASET_FILE=tulu/tulu_v1_human_mix.jsonl
DATASET=tulu

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_lora \
    --use_flash_attn \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file data/processed/${DATASET_FILE} \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/tulu_v1_human/ \
    --logging_steps 1 # &&

DATASET_FILE=tulu/tulu_v1_mix.jsonl
DATASET=tulu

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_lora \
    --use_flash_attn \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file data/processed/${DATASET_FILE} \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/tulu_v1/ \
    --logging_steps 1 # &&