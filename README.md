# Task Scoping

**This code was mostly written before the authors became competent engineers.**

## Installation
Make sure to have python 3.7.9 installed (we recommend a virtual or conda environment!). Then, from the root of this repo's directory, run:

```pip install -e .```

## Usage
You will need to run a different command depending on whether your domain and problem PDDL files are propositional or numeric.

### Propositional Domains
For these domains, we first convert the problem into the SAS+ representation, then scope the domain and problem.

```python scoping/downward_translate/translate_and_scope.py <domain_path> <problem_path> --sas-file <desired_sas_file_name> --scope True```

This will generate 2 files in the current directory: 1 `.sas` file with the name `<desired_sas_file_name>` and 1 with `_scoped` inserted into the name. The latter is the file after scoping. Each of these files can then directly be used for planning with Fast Downward or any other planner that uses SAS+ files.

Note: this leverages a PDDL to SAS+ translator taken directly from the [Fast Downward codebase](https://github.com/aibasel/downward).


Example:

```python scoping/downward_translate/translate_and_scope.py domains/propositional/toy-minecraft/toy-example.pddl domains/propositional/toy-minecraft/example-1.pddl --sas-file toy-example.sas --scope True```


You can also run this from within a python script by calling

```python
from scoping.downward_translate.translate_and_scope import main_from_other_script
main_from_other_script(**kwargs)
```

The kwargs to pass are specified in [scoping/downward_translate/options.py](scoping/downward_translate/options.py). The main kwargs are `domain`, `task`, `--scope`, and `sas_file`.

### Numeric Domains
For these domains, we first ground all variables and operators, then run task scoping on the grounded problem and use this to return "scoped" domain and problem PDDL files.

```python scoping/pddl_scoper.py --domain <path_to_domain_file> --prob <path_to_problem_file>```

Note that this assumes the domain and problem file are within the same directory.

The scoped domain and problem will be placed in the same directories as the input domain and problem. Use the scoped problem that ends with "with_cl". The file that says "sans_cl" may remove some causally-linked objects that, in principle, can be ignored, but maybe unsafe to remove (this discrepancy arises because task scoping identifies *grounded* state-variables and operators to remove, whereas PDDL files are lifted by default).

Example:

```python scoping/pddl_scoper.py --domain examples/multi_monkeys_playroom/multi_monkeys_playroom.pddl --prob examples/multi_monkeys_playroom/prob01.pddl```

## Running Experiments from paper

Experiments can be run on aws using the instructions in [aws_readme.md](aws_readme.md). Note that the experimental results shown in the paper were run on a laptop using different scripts.

## Common problems / bugs
1. Task Scoping commented out everything in my problem and domain files!!!
    1. Check if your goal is already solved in the initial state
    1. If not, there is likely some kind of contradiction in your initial state definition in your problem file. A contradiction causes our algorithm to think that the goal is already satisfied in the initial state by the [Principle of Explosion](https://en.wikipedia.org/wiki/Principle_of_explosion). These contradictions might be very subtle and are most often just re-declarations of the value of a variable (e.g. saying (= (agent-x steve) 4) on one line and (= (agent-x steve) 3) on another line in your problem file)
