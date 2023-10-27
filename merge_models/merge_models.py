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
args = parser.parse_args()

if args.base_lora_path:
    base_dir = args.base_lora_path

print(int(len(args.target_lora_modules)))
print(args.target_lora_modules)

# load the base model
base_state_dict = OrderedDict()
base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
base_model.resize_token_embeddings(len(tokenizer))

for key in base_model.state_dict():
    base_state_dict[key] = base_model.state_dict()[key]

new_state_dict = OrderedDict()

lora_base_models = []
lora_models = []
model_weights = []

for lora_module in args.target_lora_modules:
    lora_base_models.append(AutoModelForCausalLM.from_pretrained(args.base_model))
    lora_base_models[-1].resize_token_embeddings(len(tokenizer))
    lora_models.append(PeftModel.from_pretrained(lora_base_models[-1], os.path.join(base_dir, lora_module)))
    model_weights.append(len(args.target_lora_modules))

for i in range(len(lora_base_models)):
    lora_model = lora_models[i]
    for key in lora_model.state_dict():
        if key not in base_state_dict:
            new_state_dict[key] = lora_model.state_dict()[key]
        else:
            if key not in new_state_dict:
                new_state_dict[key] = torch.div(lora_model.state_dict()[key], model_weights[i])
            else:
                new_state_dict[key] += torch.div(lora_model.state_dict()[key], model_weights[i])

# load a single model to test
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=True, 
    r=256, 
    lora_alpha=256, 
    lora_dropout=0.05,
    # target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
)
base_lora_model = AutoModelForCausalLM.from_pretrained(args.base_model)
base_lora_model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(base_lora_model, peft_config)
merged_model = set_peft_model_state_dict(base_model, new_state_dict)

path_to_write = args.base_model.replace('/', '-')
out_dir = f'/results/merged-lora-weights/'
merged_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

