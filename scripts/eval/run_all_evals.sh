# alpacafarm

# example script for alpacafarm
export OPENAI_API_KEY=YOUR_API_KEY

python eval/alpaca_farm_eval.py --model tulu_65b --batch_size 8

# bbh

# cot with gpt-4
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/gpt4-cot/ \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40


# direct answer with gpt-4
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/gpt4-no-cot/ \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40 \
    --no_cot

# codex humaneval

# export CUDA_VISIBLE_DEVICES=0

# # evaluating huggingface models

# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 \
#     --unbiased_sampling_size_n 5 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/llama_7B \
#     --model ../hf_llama_models/7B/ \
#     --tokenizer ../hf_llama_models/7B/ \
#     --eval_batch_size 32 \
#     --load_in_8bit

# creative tasks ppl

python -m eval.creative_tasks.perplexity_eval \
    --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
    --output_field_name reference \
    --save_dir results/creative_tasks/reference_ppl_alpaca_65B \
    --model ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --tokenizer ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --load_in_8bit 


python -m eval.creative_tasks.perplexity_eval \
    --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
    --output_field_name gpt-4-0314_output \
    --save_dir results/creative_tasks/gpt4_ppl_alpaca_65B \
    --model ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --tokenizer ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --load_in_8bit

# gsm

# cot evaluation with gpt4
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/gpt4-cot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --n_shot 8 


# no cot evaluation with gpt4
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/gpt4-no-cot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --n_shot 8 \
    --no_cot

# mgsm

# export CUDA_VISIBLE_DEVICES=0

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-7B-2shot \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --eval_batch_size 4 \
    --n_shot 2 \
    --load_in_8bit

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-7B-6shot \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --eval_batch_size 4 \
    --n_shot 6 \
    --load_in_8bit

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-65B-2shot \
    --model ../hf_llama_models/65B \
    --tokenizer ../hf_llama_models/65B \
    --eval_batch_size 1 \
    --n_shot 2 \
    --load_in_8bit

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-65B-6shot \
    --model ../hf_llama_models/65B \
    --tokenizer ../hf_llama_models/65B \
    --eval_batch_size 1 \
    --n_shot 6 \
    --load_in_8bit

# mmlu

python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/alpaca-7B-0shot/ \
    --model_name_or_path ../checkpoints/llama_7B_large_mix/ \
    --tokenizer_name_or_path ../checkpoints/llama_7B_large_mix/ \
    --eval_batch_size 2 \
    --load_in_8bit \
    --use_chat_format

# supernatural instructions

python -m eval.superni.run_eval \
    --data_dir data/eval/superni/splits/default/ \
    --task_dir data/eval/tasks/ \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task 10 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --num_pos_examples 0 \
    --add_task_definition True \
    --output_dir results/superni/llama-7B-superni-def-only-batch-gen/ \
    --model /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/superni_7B_def_only \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/superni_7B_def_only \
    --eval_batch_size 8

# toxigen

# example scripts for toxigen

# evaluate an open-instruct model with chat format
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir tulu_65b \
    --model_name_or_path tulu_65b/ \
    --eval_batch_size 32 \
    --use_vllm \
    --use_chat_format

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

# truthful QA

python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/llama_7B_large_mix \
    --model_name_or_path ../checkpoints/llama_7B_large_mix/ \
    --tokenizer_name_or_path ../checkpoints/llama_7B_large_mix/ \
    --metrics judge info mc \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 32 \
    --load_in_8bit \
    --use_chat_format

# tydiqa

# with gold passage, using gpt4
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 20 \
    --max_context_length 512 \
    --save_dir results/tydiqa/gpt4-goldp-1shot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20

# xorqa

python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 5 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-7B-5shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --eval_batch_size 16 \
    --load_in_8bit


python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 5 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-65B-5shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --eval_batch_size 2 \
    --load_in_8bit


python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 0 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-7B-0shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --eval_batch_size 16 \
    --load_in_8bit


python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 0 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-65B-0shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --eval_batch_size 2 \
    --load_in_8bit

