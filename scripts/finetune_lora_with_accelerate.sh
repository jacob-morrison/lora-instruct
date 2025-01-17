export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/finetune.py \
#     --model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --use_flash_attn \
#     --use_lora \
#     --lora_rank 64 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     --tokenizer_name ../hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file oasst1_data.jsonl \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 16 \
#     --checkpointing_steps epoch \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 5 \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora/ \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1 &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_lora/ \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#     --save_tokenizer

# my code

# BATCH_SIZE_PER_GPU=2
# TOTAL_BATCH_SIZE=128
# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# DATASET_FILE=flan_v2/flan_v2_data.jsonl \
# DATASET=flan_v2

# DATASET_FILE=cot/cot_data.jsonl \
# DATASET=cot

# DATASET_FILE=oasst1/oasst1_data.jsonl \
# DATASET=oasst1

# DATASET_FILE=lima/lima_data.jsonl \
# DATASET=lima

# DATASET_FILE=code_alpaca/code_alpaca_data.jsonl \
# DATASET=code_alpaca

# DATASET_FILE=sharegpt/sharegpt_data.jsonl \
# DATASET=sharegpt

# DATASET_FILE=wizardlm/wizardlm_data.jsonl \
# DATASET=wizardlm

# DATASET_FILE=open_orca/open_orca_data.jsonl \
# DATASET=open_orca

# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE"

# # Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/finetune.py \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_lora \
#     --use_flash_attn \
#     --seed 13034431 \
#     --lora_rank 256 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FILE} \
#     --max_seq_length 4096 \
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
#     # --save_merged_lora_model \
#     # --with_tracking \
#     # --report_to tensorboard \

# code_alpaca_filtered
# flan_v2_filtered
# hard_coded_filtered
# oasst1_filtered
# science_filtered
# wizardlm_filtered
# cot_filtered
# gpt4_alpaca_filtered
# lima_filtered
# open_orca_filtered
# sharegpt_filtered
# tulu_v2

# for DATASET in hard_coded_filtered oasst1_filtered science_filtered wizardlm_filtered cot_filtered gpt4_alpaca_filtered lima_filtered open_orca_filtered sharegpt_filtered tulu_v2
for DATASET in code_alpaca_filtered-flan_v2_filtered      gpt4_alpaca_filtered-sharegpt_filtered    lima_filtered-code_alpaca_filtered    oasst1_filtered-sharegpt_filtered        science_filtered-sharegpt_filtered cot_filtered-code_alpaca_filtered          hard_coded_filtered-code_alpaca_filtered  lima_filtered-flan_v2_filtered        oasst1_filtered-wizardlm_filtered        science_filtered-wizardlm_filtered cot_filtered-flan_v2_filtered              hard_coded_filtered-cot_filtered          lima_filtered-open_orca_filtered      open_orca_filtered-code_alpaca_filtered  sharegpt_filtered-code_alpaca_filtered cot_filtered-gpt4_alpaca_filtered          hard_coded_filtered-flan_v2_filtered      lima_filtered-sharegpt_filtered       open_orca_filtered-flan_v2_filtered      sharegpt_filtered-flan_v2_filtered cot_filtered-lima_filtered                 hard_coded_filtered-gpt4_alpaca_filtered  oasst1_filtered-code_alpaca_filtered  open_orca_filtered-sharegpt_filtered     wizardlm_filtered-code_alpaca_filtered cot_filtered-open_orca_filtered            hard_coded_filtered-lima_filtered         oasst1_filtered-cot_filtered          science_filtered-code_alpaca_filtered    wizardlm_filtered-cot_filtered cot_filtered-sharegpt_filtered             hard_coded_filtered-oasst1_filtered       oasst1_filtered-flan_v2_filtered      science_filtered-cot_filtered            wizardlm_filtered-flan_v2_filtered gpt4_alpaca_filtered-code_alpaca_filtered  hard_coded_filtered-open_orca_filtered    oasst1_filtered-gpt4_alpaca_filtered  science_filtered-flan_v2_filtered        wizardlm_filtered-gpt4_alpaca_filtered gpt4_alpaca_filtered-flan_v2_filtered      hard_coded_filtered-science_filtered      oasst1_filtered-lima_filtered         science_filtered-gpt4_alpaca_filtered    wizardlm_filtered-lima_filtered gpt4_alpaca_filtered-lima_filtered         hard_coded_filtered-sharegpt_filtered     oasst1_filtered-open_orca_filtered    science_filtered-lima_filtered           wizardlm_filtered-open_orca_filtered gpt4_alpaca_filtered-open_orca_filtered    hard_coded_filtered-wizardlm_filtered     oasst1_filtered-science_filtered      science_filtered-open_orca_filtered      wizardlm_filtered-sharegpt_filtered
do
    echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps, on $DATASET using $DATASET_FILE" &&
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        open_instruct/finetune.py \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
        --use_flash_attn \
        --use_lora \
        --seed 13034431 \
        --lora_rank 256 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
        --use_slow_tokenizer \
        --train_file /net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/merged_datasets/${DATASET}.jsonl \
        --max_seq_length 2048 \
        --preprocessing_num_workers 16 \
        --checkpointing_steps epoch \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
        --learning_rate 2e-4 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0. \
        --num_train_epochs 3 \
        --output_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_${MODEL_SIZE}_lora_exp/pairwise_experts/${DATASET}/ \
        --logging_steps 1
done

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_${MODEL_SIZE}_lora_exp/ \
#     --output_dir output/tulu_${MODEL_SIZE}_lora_merged_exp/

#    --use_deepspeed \
#    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
