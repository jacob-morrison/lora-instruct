import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import spearmanr
import os

files = os.listdir('results/')
print(files)

# build sort order
# file_name = 'LoRA Soups - MMLU 0 Shot Pairwise Merge.csv'
interval_counts = {}
interval_counts_by_expert = {}
seen_intervals = set()
with open('./rank-correlations.csv', 'w') as f_out:
    f_out.write('dataset,correlation,pvalue,task\n')
    for file_name in files:
        data = []
        unmerged_data = []
        unmerged_data_maps = []
        print(file_name)
        task = file_name.split('-')[1].strip().split('.')[0]
        interval_counts[task] = {
            'Above': 0,
            'Below': 0,
            'In Between': 0
        }
        interval_counts_by_expert[task] = {}
        unmerged_scores = {}
        with open('results/' + file_name) as f:
            i = 0
            for line in f.readlines():
                if i == 0:
                    columns = line.split(',')[:-2]
                else:
                    tokens = line.split(',')
                    for i in range(1, len(columns)):
                        if columns[i] == 'Average' or columns[i-1] == 'WizardLM' or tokens[0] == '':
                            break
                        new_data = {}
                        if tokens[0] == columns[i]:
                            unmerged_data.append((tokens[0], float(tokens[i].replace('%', ''))))
                            unmerged_scores[tokens[0]] =  float(tokens[i].replace('%', ''))
                i += 1
                if i == 11:
                    break

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

        for key in keys_by_size:
            interval_counts_by_expert[task][key] = {
                'Above': 0,
                'Below': 0,
                'In Between': 0
            }

        unmerged_data.sort(key = lambda x: x[1])
        ranks = {}
        ranks_by_size = {}
        for i in range(len(unmerged_data)):
            ranks[unmerged_data[i][0]] = i
            ranks_by_size[unmerged_data[i][0]] = keys_by_size.index(unmerged_data[i][0])

        with open('results/' + file_name) as f:
            i = 0
            for line in f.readlines():
                if i == 0:
                    columns = line.split(',')[:-2]
                else:
                    tokens = line.split(',')
                    for i in range(1, len(columns)):
                        if columns[i] == 'Average' or columns[i-1] == 'WizardLM' or tokens[0] == '':
                            break
                        new_data = {}
                        if tokens[0] != columns[i]:
                            score = float(tokens[i].replace('%', ''))
                            new_data = {
                                'Dataset': tokens[0],
                                'Other Dataset': columns[i],
                                'Score': score,
                                'Sort Order': ranks[columns[i]],
                                'Sort Order By Size': ranks_by_size[columns[i]]
                            }
                            tup = (task, tokens[0], columns[i])
                            if tup not in seen_intervals:
                                if score > unmerged_scores[tokens[0]] and score > unmerged_scores[columns[i]]:
                                    interval_counts[task]['Above'] += 1
                                    interval_counts_by_expert[task][tokens[0]]['Above'] += 1
                                elif score < unmerged_scores[tokens[0]] and score < unmerged_scores[columns[i]]:
                                    interval_counts[task]['Below'] += 1
                                    interval_counts_by_expert[task][tokens[0]]['Below'] += 1
                                else:
                                    interval_counts[task]['In Between'] += 1
                                    interval_counts_by_expert[task][tokens[0]]['In Between'] += 1
                                seen_intervals.add(tup)
                        # else:
                        #     new_data = {
                        #         'Dataset': tokens[0],
                        #         'Other Dataset': columns[i],
                        #         'Score': 0.0,
                        #         'Sort Order': ranks[columns[i]],
                        #         'Sort Order By Size': ranks_by_size[columns[i]]
                        #     }
                        #     unmerged_data_maps.append({
                        #         'Dataset': tokens[0],
                        #         'Other Dataset': columns[i],
                        #         'Score': float(tokens[i].replace('%', '')),
                        #         'Sort Order': ranks[columns[i]],
                        #         'Sort Order By Size': ranks_by_size[columns[i]]
                        #     })
                            data.append(new_data)
                i += 1
                if i == 11:
                    break

        unmerged_data.sort(key = lambda x: x[1])
        keys = []
        for dataset, _ in unmerged_data:
            keys.append(dataset)

        for dataset in keys:
            unmerged_df = pd.DataFrame(unmerged_data)
            unmerged_df.columns =['Dataset', 'Score']
            df = pd.DataFrame(data)
            df = df.sort_values('Sort Order')
            df = df[df['Dataset'] == dataset].dropna()
            unmerged_df = unmerged_df[unmerged_df['Dataset'] != dataset].dropna()
            # print(df)
            # print(unmerged_df)
            # print(df['Score'].tolist())
            # print(unmerged_df['Score'].tolist())
            result = spearmanr(df['Score'].tolist(), unmerged_df['Score'].tolist())
            # print(f'Rank correlation for {dataset}: correlation: {str(result.statistic)}, pvalue: {str(result.pvalue)}')
            f_out.write(f'{dataset},{str(result.statistic)},{str(result.pvalue)},{task}\n')

with open('interval-counts.csv', 'w') as f_out:
    f_out.write('task,above,in between, below\n')
    for task in interval_counts:
        f_out.write(f'{task},{interval_counts[task]["Above"]},{interval_counts[task]["In Between"]},{interval_counts[task]["Below"]}\n')
with open('interval-counts-by-task.csv', 'w') as f_out:
    f_out.write('task,expert,above,in between, below\n')
    for task in interval_counts_by_expert:
        for expert in interval_counts_by_expert[task]:
            f_out.write(f'{task},{expert},{interval_counts_by_expert[task][expert]["Above"]},{interval_counts_by_expert[task][expert]["In Between"]},{interval_counts_by_expert[task][expert]["Below"]}\n')
pprint(interval_counts_by_expert)