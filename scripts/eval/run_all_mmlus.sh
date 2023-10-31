for DATASET in code_alpaca_filtered  cot_filtered  flan_v2_filtered  gpt4_alpaca_filtered  hard_coded_filtered  lima_filtered  oasst1_filtered  open_orca_filtered science_filtered  sharegpt_filtered wizardlm_filtered
do
    for SHOTS in 0 5
    do
        python -m eval.mmlu.run_eval \
            --ntrain ${SHOTS} \
            --data_dir /tulu-eval-data/mmlu/ \
            --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/mmlu-${SHOTS}-shot \
            --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
            --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
            --lora_weight_path /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/${DATASET}/ \
            --eval_batch_size 2 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
    done
done