from copy import deepcopy
from glob import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

log_dir = 'logs/fd/lmcut/autoscale'

domains = os.listdir(log_dir)

empty_timings_dict = {
    "scoping":[],
    "plan_scoped_time":[],
    "total_scoped_time":[],
    "total_unscoped_time":[],
    "plan_unscoped_generated_nodes": [],
    "plan_unscoped_node_expansions": [],
    "plan_scoped_generated_nodes": [],
    "plan_scoped_node_expansions": [],
    "encoding_size": [],
    "scoping_exit_code": [],
    "plan_scoped_exit_code": [],
    "plan_unscoped_exit_code": [],
    "domain": [],
    "problem": [],
    "seed": [],
}

timings_dict = deepcopy(empty_timings_dict)
for domain in domains:
    domain_dir = log_dir + '/' + domain
    if not os.path.isdir(domain_dir):
        continue
    problems = os.listdir(domain_dir)
    for problem in problems:
        problem_dir = domain_dir + '/' + problem
        if not os.path.isdir(problem_dir):
            continue
        results_paths = glob(problem_dir + '/*/times.json')
        for timings_path in results_paths:
            seed = timings_path.split('/')[-2]
            with open(timings_path, "r") as f:
                loaded_timings = json.load(f)
            for key in loaded_timings.keys():
                value = np.nan if loaded_timings[key] == [] else loaded_timings[key][0]
                timings_dict[key].append(value)
            timings_dict['domain'].append(domain)
            timings_dict['problem'].append(problem)
            timings_dict['seed'].append(seed)

df = pd.DataFrame(data=timings_dict)
df.to_csv('fdr-autoscale-results.csv')

#%% Check if scoping helps solve rate

solved_scoped_df = df[(df['plan_scoped_exit_code'] == 0) & pd.to_numeric(df['total_scoped_time'], errors='coerce').notnull()]

solved_unscoped_df = df[(df['plan_unscoped_exit_code'] == 0) & pd.to_numeric(df['total_unscoped_time'], errors='coerce').notnull()]

solved_scoped_stats = solved_scoped_df.groupby('domain')['problem'].nunique().reset_index(name='n_solved_scoped')
solved_unscoped_stats = solved_unscoped_df.groupby('domain')['problem'].nunique().reset_index(name='n_solved_unscoped')
solved_total_stats = df.groupby('domain')['problem'].nunique().reset_index(name='n_problems')

# Joining all three sets of stats into a single table
merged_solve_stats = pd.merge(solved_total_stats, solved_scoped_stats, on='domain', how='left')
merged_solve_stats = pd.merge(merged_solve_stats, solved_unscoped_stats, on='domain', how='left')

merged_solve_stats.fillna(0, inplace=True)  # Replacing NaN values with 0

#%% Check times

# Calculating the mean and standard error for specified metrics across all seeds for each domain and problem pair
filtered_df = df.dropna(subset=['plan_scoped_time', 'total_scoped_time'])
filtered_merged_df = filtered_df.groupby(filtered_df['domain'].str.split('-').str[0]).agg({
    'scoping': ['mean', 'sem'],
    'plan_scoped_time': ['mean', 'sem'],
    'total_scoped_time': ['mean', 'sem'],
    'total_unscoped_time': ['mean', 'sem'],
})
