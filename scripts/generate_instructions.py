print('FD - IPC Domains')
for DOMAIN, PROBLEMS in [
    ('driverlog', ['15', '16', '17']),
    ('logistics', ['15', '20', '25']),
    ('satellite', ['05', '06', '07']),
    ('zenotravel', ['10', '12', '14']),
]:
  for PROBLEM in PROBLEMS:
    for RUN_ID in range(10):
      print(f"python ./experiments/fd_experiment.py 1 examples/IPC_domains_propositional/{DOMAIN}/domain.pddl examples/IPC_domains_propositional/{DOMAIN}/prob{PROBLEM}.pddl ../downward/fast-downward.py ./logs --force_clear_log_dir --plan_type lmcut --run_id {RUN_ID}")


print('Playroom')
for PROBLEM in ['1', '3', '5', '7', '9']:
  for RUN_ID in range(10):
    print(f"python ./experiments/enhsp_experiment.py 1 examples/multi_monkeys_playroom/multi_monkeys_playroom.pddl examples/multi_monkeys_playroom/prob0{PROBLEM}.pddl ./logs --force_clear_log_dir --planner opt --run_id {RUN_ID}")


print('IPC Composite')
for PROBLEM in ['1', '2', '3', '4', '5']:
  for RUN_ID in range(10):
    print(f"python ./experiments/enhsp_experiment.py 1 scoping/examples/domains/IPC_Domains/CompositeIPC/ipc_composite.pddl scoping/examples/domains/IPC_Domains/CompositeIPC/prob-0{PROBLEM}.pddl ./logs --force_clear_log_dir --planner opt --run_id {RUN_ID}")

print('Minecraft')
for PROBLEM in ['make_wooden_planks', 'get_dyed_wool', 'make_bed']:
  for RUN_ID in range(10):
    print(f"python ./experiments/enhsp_experiment.py 1 scoping/examples/domains/minecraft3/minecraft-contrived3.pddl scoping/examples/domains/minecraft3/prob_{PROBLEM}_irrel.pddl ./logs --force_clear_log_dir --planner opt --run_id {RUN_ID}")

print('Toy Example')
print(f"python ./experiments/enhsp_experiment.py 1 scoping/examples/domains/toy-example-numeric/toy-example.pddl scoping/examples/domains/toy-example-numeric/example-1.pddl ./logs --force_clear_log_dir --planner opt")
