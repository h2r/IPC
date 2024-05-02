from glob import glob

print('FDR - autoscale')
for DOMAIN in glob('fdr-generator/benchmarks/autoscale/*'):
  for PROBLEM in glob(DOMAIN+"/*[0-9].sas"):
    for RUN_ID in range(1):
      print(f"python ./experiments/sas_experiment.py 1 {PROBLEM} ../downward/fast-downward.py ./logs --force_clear_log_dir --plan_type lmcut --run_id {RUN_ID}")
#%%
print('----------------------------------------')
print('FDR - structural')
for DOMAIN in glob('fdr-generator/benchmarks/structural/*'):
  for PROBLEM in glob(DOMAIN+"/*[0-9].sas"):
    print(f"python ./experiments/sas_experiment.py --n_runs 10 --fd_path ../downward/fast-downward.py --log_dir ./logs --plan_type lmcut --sas_file {PROBLEM} --run_id -1")
