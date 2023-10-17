import json
import os

def should_be_filtered(example):
    # we filter out conversations that contain some specific strings
    filter_strings = [
        "OpenAI",
        "Open AI",
        "ChatGPT",
        "Chat GPT",
        "GPT-3",
        "GPT3",
        "GPT 3",
        "GPT-4",
        "GPT4",
        "GPT 4",
        "GPT-3.5",
        "GPT3.5",
        "GPT 3.5",
        "BingChat",
        "Bing Chat",
        "BARD",
        "Palm",
        "Anthropic",
        "Claude",
        "LAION",
        "Open Assistant",
        "OpenAssistant", 
    ]
    for message in example["messages"]:
        if any([filter_string.lower() in message["content"].lower() for filter_string in filter_strings]):
            return True
    return False

all_subsets = [f for f in os.listdir(os.path.join('/tulu-data/', "tulu_v2")) if f.endswith("_subset")]
with open("/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/tulu_v2_data.jsonl", "w") as fout, \
    open("/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/tulu_v2_filtered_data.jsonl", "w") as fout_filtered:
    for subset in all_subsets:
        print(subset)
        dataset_name = subset[:-len("_subset")]
        with open(os.path.join('/tulu-data/', "tulu_v2", subset, f"{dataset_name}_data.jsonl"), "r") as fin, \
            open(f"/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/{dataset_name}_filtered_data.jsonl", "w") as fout_filtered:
            for line in fin:
                example = json.loads(line)
                if subset not in ["hard_coded_subset"] and should_be_filtered(example):
                    fout_filtered.write(line)
                else:
                    fout.write(line)