import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pandas as pd

data = []
unmerged_data = []
unmerged_data_maps = []
# build sort order
with open('results/LoRA Soups - MMLU 0 Shot Pairwise Merge.csv') as f:
    i = 0
    for line in f.readlines():
        if i == 0:
            columns = line.split(',')[:-2]
            i += 1
        else:
            tokens = line.split(',')
            for i in range(1, len(columns)):
                if columns[i] == 'Average' or tokens[0] == '':
                    break
                new_data = {}
                if tokens[0] == columns[i]:
                    unmerged_data.append((tokens[0], float(tokens[i].replace('%', ''))))

# dataset size:
# keys_by_size = [
#     'Hard Coded',
#     'Lima',
#     'Code Alpaca',
#     'Open Assistant 1',
#     'GPT4 Alpaca',
#     'Open Orca',
#     'Science',
#     'CoT',
#     'WizardLM',
#     'Flan V2',
#     'ShareGPT',
# ]

# number of examples:
keys_by_size = [
    'Hard Coded',
    'Lima',
    'Open Assistant 1',
    'Science',
    'GPT4 Alpaca',
    'Code Alpaca',
    'Open Orca',
    'WizardLM',
    'Flan V2',
    'CoT',
    'ShareGPT',
]

unmerged_data.sort(key = lambda x: x[1])
ranks = {}
ranks_by_size = {}
for i in range(len(unmerged_data)):
    ranks[unmerged_data[i][0]] = i
    ranks_by_size[unmerged_data[i][0]] = keys_by_size.index(unmerged_data[i][0])

with open('results/LoRA Soups - MMLU 0 Shot Pairwise Merge.csv') as f:
    i = 0
    for line in f.readlines():
        if i == 0:
            columns = line.split(',')[:-2]
            i += 1
        else:
            tokens = line.split(',')
            for i in range(1, len(columns)):
                if columns[i] == 'Average' or tokens[0] == '':
                    break
                new_data = {}
                if tokens[0] != columns[i]:
                    new_data = {
                        'Dataset': tokens[0],
                        'Other Dataset': columns[i],
                        'Score': float(tokens[i].replace('%', '')),
                        'Sort Order': ranks[columns[i]],
                        'Sort Order By Size': ranks_by_size[columns[i]]
                    }
                else:
                    new_data = {
                        'Dataset': 'Unmerged',
                        'Other Dataset': columns[i],
                        'Score': float(tokens[i].replace('%', '')),
                        'Sort Order': ranks[columns[i]],
                        'Sort Order By Size': ranks_by_size[columns[i]]
                    }
                    unmerged_data_maps.append({
                        'Dataset': tokens[0],
                        'Other Dataset': columns[i],
                        'Score': float(tokens[i].replace('%', '')),
                        'Sort Order': ranks[columns[i]],
                        'Sort Order By Size': ranks_by_size[columns[i]]
                    })
                data.append(new_data)

unmerged_data.sort(key = lambda x: x[1])
keys = []
for dataset, _ in unmerged_data:
    keys.append(dataset)
# pprint(data)
# pprint(keys)

for dataset in keys:
    plt.figure()
    df = pd.DataFrame(data)
    df = df.sort_values('Sort Order')
    # df = df.sort_values('Sort Order By Size')
    df = df[df['Dataset'].isin([dataset, 'Unmerged'])].dropna()
    # df = df[df['Dataset'] == dataset].dropna()
    print(df.to_string())
    fig, ax = plt.subplots(figsize=(16,12))
    sns.set(rc={'figure.figsize':(16,12)})
    ax.set_ylim(30, 55)
    sns.scatterplot(data=df, x = 'Other Dataset', y = 'Score', hue = 'Dataset')
    plt.savefig(f"./mmlu_0_shot_pairwise_merge/{dataset}.png")
    # plt.show()

# df = pd.DataFrame(unmerged_data_maps)
# print(df.to_string())
# df = df.sort_values('Sort Order')
# df = df.sort_values('Sort Order By Size')
# print(df.to_string())
# fig, ax = plt.subplots(figsize=(16,12))
# scatterplot = sns.scatterplot(data=df, ax=ax, x = 'Other Dataset', y = 'Score', hue = 'Dataset', hue_order = keys)
# fig = scatterplot.get_figure()
# sns.set(rc={'figure.figsize':(16,12)})
# ax.set_ylim(30, 55)
# fig.savefig("mmlu_0_shot-by-size.png") 
# plt.show()