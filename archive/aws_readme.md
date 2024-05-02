# AWS Readme

These are instructions for running scoping experiments on aws EC2. Note that the results in the paper are from experiments run on a laptop using a different set of scripts, NOT in EC2 using the scripts described here. You can find a tutorial on using AWS EC2 [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html).

## Instance parameters
These instructions were verified to work using an instance with the following configuration:

- Type: t2.large

- AMI
  - AMI ID: ami-0ac5a3737d540e82a
  - AMI Name: Deep Learning Base AMI (Ubuntu 16.04) Version 40.0
  - AMI Location: amazon/Deep Learning Base AMI (Ubuntu 16.04) Version 40.0

- Network 
## Usage

Copy the scoping_supplement.zip to your ec2 instance with scp:

    scp -i <path2key_file> scoping_supplement.zip ubuntu@<ec2_instance_address>:~/

SSH into the ec2 instance

    ssh -i <path2key_file> ubuntu@<ec2_instance_address>


Unzip the supplement

    unzip scoping_supplement.zip

Mount the planutils docker image

    source scoping_supplement/scoping/ubuntu_scripts/get_enhsp_docker.sh

Change directory to /scoping_supplement

    cd /scoping_supplement

Run config_all.sh to setup conda, fast downward

    source scoping/ubuntu_scripts/config_all.sh

Run ENHSP-2020 experiments using the experiments/enhsp_experiment.py script. The arguments are:

1. Number of runs in this experiment
2. Path to domain file
3. Path to problem file
4. Path to directory the experiment logs will be saved in
5. (optional) `--force_clear_log_dir` if you want to forcibly clear the passed experiment log directory. If you pass an already existing directory as the experiment log directory and don't use the `--force_clear_log_dir` option, the script will raise an exception.

Example command:

    python ./experiments/enhsp_experiment.py 2 examples/multi_monkeys_playroom/multi_monkeys_playroom.pddl examples/multi_monkeys_playroom/prob01.pddl ./experiment_logs/multi_monkeys_playroom_01


Run Fast Downward experiments using the experiments/fd_experiment.py script. The arguments are the same as for the enhsp script. Example command:

    python experiments/fd_experiment.py 1 examples/IPC_domains_propositional/logistics00/domain.pddl examples/IPC_domains_propositional/logistics00/prob15.pddl ../downward/fast-downward.py experiment_logs/logistics15

To move results out of the ec2 isntance, you can use scp from your local machine:

    scp -i <path2key_file> ubuntu@<ec2_instance_address>:<path2experiments_log_dir> scoping_experiment_logs