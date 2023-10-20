# example scripts for toxigen

# evaluate an open-instruct model with chat format
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir tulu_65b \
    --model_name_or_path tulu_65b/ \
    --eval_batch_size 32 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# evaluate a base model without chat format
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir tulu_65b \
    --model_name_or_path tulu_65b/ \
    --eval_batch_size 32 \
    --use_vllm


# evaluate chatGPT
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results_chatgpt \
    --openai_engine gpt-3.5-turbo-0301

# mine
python -m eval.toxigen.run_eval \
    --data_dir /tulu-eval-data/toxigen/ \
    --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/toxigen \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
    --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
    --lora_weight_path /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/${DATASET}/ \
    --eval_batch_size 16 \
    --use_vllm \
    --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
