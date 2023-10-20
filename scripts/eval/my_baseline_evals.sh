# for DATASET in tulu_v2
for DATASET in llama-2-7b
do
    # toxigen
    python -m eval.toxigen.run_eval \
        --data_dir /tulu-eval-data/toxigen/ \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/toxigen \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --eval_batch_size 16 \
        --use_vllm \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # MMLU
    python -m eval.mmlu.run_eval \
        --ntrain ${SHOTS} \
        --data_dir /tulu-eval-data/mmlu/ \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/mmlu-${SHOTS}-shot \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --eval_batch_size 2 \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # GSM CoT (WIP)
    python -m eval.gsm.run_eval \
        --data_dir /tulu-eval-data/gsm/ \
        --max_num_examples 200 \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/gsm/cot/ \
        --use_vllm \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --n_shot 8 \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # GSM Direct (WIP)
    python -m eval.gsm.run_eval \
        --data_dir /tulu-eval-data/gsm/ \
        --max_num_examples 200 \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/gsm/direct/ \
        --use_vllm \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --n_shot 8 \
        --no_cot \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # TydiQA No Context (WIP)
    python -m eval.tydiqa.run_eval \
        --data_dir /tulu-eval-data/tydiqa/ \
        --no_context \
        --n_shot 1 \
        --max_num_examples_per_lang 100 \
        --max_context_length 512 \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/tydiqa/no-context/ \
        --use_vllm \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # TydiQA GoldP (WIP)
    python -m eval.tydiqa.run_eval \
        --data_dir /tulu-eval-data/tydiqa/ \
        --n_shot 1 \
        --max_num_examples_per_lang 100 \
        --max_context_length 512 \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/tydiqa/goldp/ \
        --use_vllm \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # BBH CoT (WIP)
    python -m eval.bbh.run_eval \
        --data_dir /tulu-eval-data/bbh \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/bbh/cot/ \
        --use_vllm \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --max_num_examples_per_task 40 \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

    # BBH Direct (WIP)
    python -m eval.bbh.run_eval \
        --data_dir /tulu-eval-data/bbh \
        --save_dir /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/${DATASET}/bbh/direct/ \
        --use_vllm \
        --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --tokenizer_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/ \
        --max_num_examples_per_task 40 \
        --no_cot \
        --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
done