from copy import deepcopy
from glob import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

log_dir = 'logs/fd/lmcut/structural'

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
df.to_csv('fdr-results.csv')

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
    'plan_unscoped_generated_nodes': 'mean',
    'plan_unscoped_node_expansions': 'mean',
    'plan_scoped_generated_nodes': 'mean',
    'plan_scoped_node_expansions': 'mean',
})

metrics = ['scoping', 'plan_scoped_time', 'total_scoped_time', 'total_unscoped_time']
stats_df = filtered_merged_df.groupby(['domain', 'problem'])[metrics].agg(['mean', 'sem']).reset_index()

# Renaming columns for clarity
stats_df.columns = [' '.join(col).strip() for col in stats_df.columns.values]

#%% Summarize stats for each domain, filtering out NaNs


# Filtering out rows where either plan_scoped_time or total_scoped_time is NaN
# Also removing corresponding scoping time entries to maintain consistency

# Comparing the recalculated total scoped time with the reported total scoped time
# Displaying the first few rows for a preview


# Averaging the non-NaN time values for each time column over the problems in each domain
# This is done for all four time columns

# Filtering out rows where all time values are NaN
filtered_stats_df = stats_df.dropna(subset=['scoping mean', 'scoping sem', 'plan_scoped_time mean', 'plan_scoped_time sem', 'total_scoped_time mean', 'total_scoped_time sem', 'total_unscoped_time mean', 'total_unscoped_time sem'], how='all')

# Grouping by domain and calculating the mean for each time column, considering only non-NaN values
domain_averages = filtered_stats_df.groupby('domain').agg({
    'scoping mean': 'mean',
    'scoping sem': 'sem',
    'plan_scoped_time mean': 'mean',
    'plan_scoped_time sem': 'sem',
    'total_scoped_time mean': 'mean',
    'total_scoped_time sem': 'sem',
    'total_unscoped_time mean': 'mean',
    'total_unscoped_time sem': 'sem'
}).reset_index()

domain_averages

#%%
# Identifying domains where the total_scoped_time mean is less than the total_unscoped_time mean
domains_with_lower_scoped_time = domain_averages[domain_averages['total_scoped_time mean'] < domain_averages['total_unscoped_time mean']]

domains_with_lower_scoped_time[['domain', 'total_scoped_time mean', 'total_unscoped_time mean']]

#%%

# Identifying domains where the plan_scoped_time mean is less than the total_unscoped_time mean
domains_with_lower_plan_scoped_time = domain_averages[domain_averages['plan_scoped_time mean'] < domain_averages['total_unscoped_time mean']]

domains_with_lower_plan_scoped_time[['domain', 'plan_scoped_time mean', 'total_unscoped_time mean']]


#%%

# Function to perform Mann-Whitney U test for comparing scoped and unscoped times for each domain
def perform_mannwhitneyu_test_by_domain(df, scoped_col, unscoped_col):
    test_results = {}
    grouped = df.groupby('domain')

    for name, group in grouped:
        scoped_data = group[scoped_col].dropna()
        unscoped_data = group[unscoped_col].dropna()

        # Ensure there are enough data points for comparison
        if len(scoped_data) >= 3 and len(unscoped_data) >= 3:
            stat, p_value = mannwhitneyu(scoped_data, unscoped_data, alternative='two-sided')
            test_results[name] = p_value

    return test_results

# Performing Mann-Whitney U tests for comparing scoped and unscoped times by domain
test_results_scoped_vs_unscoped_total_time_by_domain = perform_mannwhitneyu_test_by_domain(df, 'total_scoped_time', 'total_unscoped_time')
test_results_scoped_vs_unscoped_plan_time_by_domain = perform_mannwhitneyu_test_by_domain(df, 'plan_scoped_time', 'total_unscoped_time')

# Converting the results to a DataFrame for better readability
test_results_total_time_df = pd.DataFrame(list(test_results_scoped_vs_unscoped_total_time_by_domain.items()), columns=['Domain', 'Total Time p-value'])
test_results_plan_time_df = pd.DataFrame(list(test_results_scoped_vs_unscoped_plan_time_by_domain.items()), columns=['Domain', 'Plan Time p-value'])

# Merging the results into a single DataFrame
merged_test_results_df = pd.merge(test_results_total_time_df, test_results_plan_time_df, on='Domain')
merged_test_results_df.round(5)  # Displaying the first few rows for a preview


#%%

# Merging the statistical test results with the domain average times
final_merged_df = pd.merge(domain_averages, merged_test_results_df, left_on='domain', right_on='Domain')
final_merged_df = pd.merge(final_merged_df, merged_solve_stats, left_on='domain', right_on='domain')

# Dropping the extra 'Domain' column as it's redundant
final_merged_df.drop('Domain', axis=1, inplace=True)

final_merged_df
