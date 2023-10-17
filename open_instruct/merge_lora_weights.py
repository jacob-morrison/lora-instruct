
# Time this whole operation:
# general algorithm:
    # 1. load base peft model with default LoRa settings
    # 2. create new state dict and save non-lora weights
    # 3. save lora weights in new state dicts for each set of lora weights
    # 4. in the new state dict from step # 2, add in merged lora weights to the dictionary (because non-lora weights won't change)
    # 5. re-load base peft model with set_peft_model_state_dict(base_model, new_state_dict)

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from peft import LoraConfig, PeftModel, PeftConfig, TaskType, get_peft_model, set_peft_model_state_dict
from collections import OrderedDict
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-c", "--checkpoint", help = "Which checkpoint", type = int)
# parser.add_argument("-b", "--base_model", help = "Which base model", type = str)
# parser.add_argument("-l", "--lora", help = "Use lora or not", type = bool)
# args = parser.parse_args()


# load the base model
base_state_dict = OrderedDict()
base_model_path = ''
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model.resize_token_embeddings(len(tokenizer))

for key in base_model.state_dict():
    base_state_dict[key] = base_model.state_dict()[key]

# load the lora modules
lora_paths = {
    'orca': ''
}

lora_state_dicts = {}

for dataset in lora_paths:
    lora_base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    lora_base_model.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(lora_base_model, lora_paths[dataset])
    lora_state_dicts[dataset] = OrderedDict()
    for key in lora_model.state_dict():
        if key not in base_state_dict:
            lora_state_dicts[dataset][key] = lora_model.state_dict()[key]
        else:
            if lora_model.state_dict()[key].shape == base_state_dict[key].shape:
                lora_state_dicts[dataset][key] = lora_model.state_dict()[key]
            else:
                print('The base shapes don\'t match??')
                print(lora_model.state_dict()[key].shape)
                print(base_state_dict[key].shape)

# load a single model to test
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=True, 
    r=256, 
    lora_alpha=256, 
    lora_dropout=0.05
    target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
)
base_lora_model = AutoModelForCausalLM.from_pretrained(base_model_path)
base_lora_model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(base_lora_model, peft_config)
orca_model = set_peft_model_state_dict(base_model, lora_state_dicts['orca'])
