
# Time this whole operation:
# general algorithm:
    # 1. load base peft model with default LoRa settings
    # 2. create new state dict and save non-lora weights
    # 3. save lora weights in new state dicts for each set of lora weights
    # 4. in the new state dict from step # 2, add in merged lora weights to the dictionary (because non-lora weights won't change)
    # 5. re-load base peft model with set_peft_model_state_dict(base_model, new_state_dict)
