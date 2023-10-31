from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from peft import PeftModel, PeftConfig
from collections import OrderedDict
import argparse
from peft import LoraConfig, PeftModel, PeftConfig, TaskType, get_peft_model, set_peft_model_state_dict
import os

base_dir = '/net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/'

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_model", help = "Which base model", type = str)
parser.add_argument("-l", "--base_lora_path", help = "Which base model", type = str)
parser.add_argument('--target_lora_modules', nargs='+')
parser.add_argument('--results_dir', nargs='?', default='/results')
args = parser.parse_args()

if args.base_lora_path:
    base_dir = args.base_lora_path

print(int(len(args.target_lora_modules)))
print(args.target_lora_modules)

# # load the base model
# base_state_dict = OrderedDict()
# base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
# tokenizer = AutoTokenizer.from_pretrained(args.base_model)
# base_model.resize_token_embeddings(len(tokenizer))

# for key in base_model.state_dict():
#     base_state_dict[key] = base_model.state_dict()[key]

# new_state_dict = OrderedDict()

# lora_base_models = []
# lora_models = []
# model_weights = []

# for lora_module in args.target_lora_modules:
#     lora_base_models.append(AutoModelForCausalLM.from_pretrained(args.base_model))
#     lora_base_models[-1].resize_token_embeddings(len(tokenizer))
#     lora_models.append(PeftModel.from_pretrained(lora_base_models[-1], os.path.join(base_dir, lora_module)))
#     model_weights.append(len(args.target_lora_modules))

# for i in range(len(lora_base_models)):
#     lora_model = lora_models[i]
#     for key in lora_model.state_dict():
#         if key in base_state_dict:
#             new_state_dict[key] = lora_model.state_dict()[key]
#         else:
#             if key not in new_state_dict:
#                 new_state_dict[key] = torch.div(lora_model.state_dict()[key], model_weights[i])
#             else:
#                 new_state_dict[key] += torch.div(lora_model.state_dict()[key], model_weights[i])

# # load a single model to test
# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM, 
#     inference_mode=False, 
#     r=256, 
#     lora_alpha=256, 
#     lora_dropout=0.05,
#     target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
# )
# base_lora_model = AutoModelForCausalLM.from_pretrained(args.base_model)
# base_lora_model.resize_token_embeddings(len(tokenizer))
# final_model = get_peft_model(base_lora_model, peft_config)
# # print(new_state_dict.keys())
# # print(final_model.state_dict().keys())
# # merged_model = set_peft_model_state_dict(model, new_state_dict)
# merge_result = final_model.load_state_dict(new_state_dict)
# print(merge_result)

# if len(merge_result.missing_keys) == 0 and len(merge_result.unexpected_keys) == 0:
#     print("This worked!")

#     path_to_write = args.base_model.replace('/', '-')
#     out_dir = os.path.join(args.results_dir, 'merged-lora-weights/')
#     print("Writing to " + out_dir)
#     final_model.save_pretrained(out_dir)
#     tokenizer.save_pretrained(out_dir)

import sys

print(sys.path)

# now call eval on out_dir
from eval.mmlu.run_eval import main

# delete this later
out_dir = os.path.join(args.results_dir, 'merged-lora-weights/')

MMLU_0_shot_args = [
    "--ntrain", 0,
    "--data_dir", "/tulu-eval-data/mmlu/",
    "--save_dir", os.path.join(args.results_dir, "0-shot-results"),
    "--model_name_or_path", "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/",
    "--tokenizer_name_or_path", "/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/"
    "--eval_batch_size", 2,
    "--use_chat_format",
    "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
    "--lora_weight_path", out_dir,
]

mmlu_0_shot_parser = argparse.ArgumentParser()
mmlu_0_shot_parser.add_argument(
    "--ntrain",
    type=int,
    default=5
)
mmlu_0_shot_parser.add_argument(
    "--data_dir",
    type=str,
    default="data/mmlu"
)
mmlu_0_shot_parser.add_argument(
    "--save_dir",
    type=str,
    default="results/mmlu/llama-7B/"
)
mmlu_0_shot_parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=None,
    help="if specified, we will load the model to generate the predictions."
)
mmlu_0_shot_parser.add_argument(
    "--tokenizer_name_or_path",
    type=str,
    default=None,
    help="if specified, we will load the tokenizer from here."
)
mmlu_0_shot_parser.add_argument(
    "--use_slow_tokenizer",
    action="store_true",
    help="If given, we will use the slow tokenizer."
)
mmlu_0_shot_parser.add_argument(
    "--openai_engine",
    type=str,
    default=None,
    help="if specified, we will use the OpenAI API to generate the predictions."
)
mmlu_0_shot_parser.add_argument(
    "--subjects",
    nargs="*",
    help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
)
mmlu_0_shot_parser.add_argument(
    "--n_instances",
    type=int,
    help="if specified, a maximum of n_instances per subject will be used for the evaluation."
)
mmlu_0_shot_parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=1,
    help="batch size for evaluation."
)
mmlu_0_shot_parser.add_argument(
    "--load_in_8bit",
    action="store_true",
    help="load model in 8bit mode, which will reduce memory and speed up inference."
)
mmlu_0_shot_parser.add_argument(
    "--gptq",
    action="store_true",
    help="If given, we're evaluating a 4-bit quantized GPTQ model."
)
mmlu_0_shot_parser.add_argument(
    "--use_chat_format", 
    action="store_true", 
    help="If given, we will use the chat format for the prompts."
)
mmlu_0_shot_parser.add_argument(
    "--chat_formatting_function", 
    type=str, 
    default="eval.templates.create_prompt_with_tulu_chat_format", 
    help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
)
mmlu_0_shot_parser.add_argument(
    "--lora_weight_path",
    help="If given, we load lora weights."
)


parsed_mmlu_0_shot_args = mmlu_0_shot_parser.parse_args(MMLU_0_shot_args)
main(parsed_mmlu_0_shot_args)
