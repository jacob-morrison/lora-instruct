import json
import os
from glob import glob
from pprint import pprint

root_dir = '/net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/results/'
subdirectories = glob(root_dir + '/*/', recursive=False)
pprint(subdirectories)
final_data = {}
for subdir in subdirectories:
    print(subdir)
    dataset = subdir.split('/')[-2]
    print(dataset)
    # bbh
        # cot
    with open(os.path.join(subdir, '/bbh/cot/', 'metrics.json')) as f:
        bbh_cot_data = json.loads(f.read())
        # direct
    with open(os.path.join(subdir, '/bbh/direct/', 'metrics.json')) as f:
        bbh_direct_data = json.loads(f.read())

    # gsm
        # cot
    with open(os.path.join(subdir, '/gsm/cot/', 'metrics.json')) as f:
        gsm_cot_data = json.loads(f.read())
        # direct
    with open(os.path.join(subdir, '/gsm/direct/', 'metrics.json')) as f:
        gsm_direct_data = json.loads(f.read())

    # mmlu-0-shot
    with open(os.path.join(subdir, '/mmlu-0-shot/', 'metrics.json')) as f:
        mmlu_0_shot_data = json.loads(f.read())

    # mmlu-5-shot
    with open(os.path.join(subdir, '/mmlu-5-shot/', 'metrics.json')) as f:
        mmlu_5_shot_data = json.loads(f.read())

    # toxigen
    with open(os.path.join(subdir, '/toxigen/', 'metrics.json')) as f:
        toxigen_data = json.loads(f.read())

    # tydiqa
        # goldp
    with open(os.path.join(subdir, '/tydiqa/goldp/', 'metrics.json')) as f:
        tydiqa_goldp_data = json.loads(f.read())
        # no-context
    with open(os.path.join(subdir, '/tydiqa/no-context/', 'metrics.json')) as f:
        tydiqa_no_context_data = json.loads(f.read())

    final_data[dataset] = {
        'bbh_cot_data': bbh_cot_data,
        'bbh_direct_data': bbh_direct_data,
        'gsm_cot_data': gsm_cot_data,
        'gsm_direct_data': gsm_direct_data,
        'mmlu_0_shot_data': mmlu_0_shot_data,
        'mmlu_5_shot_data': mmlu_5_shot_data,
        'toxigen_data': toxigen_data,
        'tydiqa_goldp_data': tydiqa_goldp_data,
        'tydiqa_no_context_data': tydiqa_no_context_data,
    }

pprint(final_data)
