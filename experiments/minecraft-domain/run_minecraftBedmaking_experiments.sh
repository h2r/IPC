#!/bin/bash

# Usage: (from repository root, run) ./experiments/minecraft-domain/run_minecraftBedMaking_experiments.sh <num_trials> <optional scope_true_or_false>
# If the second (optional) arg is not provided, only planning commands will be run
# if the second arg is the string "scope_true" (or really anything that's not null), this script will run scoping num_trials times and report times
# Example: ./experiments/minecraft-domain/run_minecraftBedMaking_experiments.sh 5 scope_true
TIMEFORMAT=%R
user_arg="$1"

if [ -n "$2" ]
then
    declare -a sruntimes
    # The counter needs to be 1 + num_trials you want for the experiment, because the first trial is slightly slower due to needing to execute
    # some of the setup bash commands. Thus, take all readings except the first!
    COUNTER="$user_arg + 1"
    until [ $COUNTER -lt 1 ]; do
        echo RUN_NUMBER $COUNTER
        exec 3>&1 4>&2
        foo=$( { time python scoping/scope_and_writeback_pddl.py --domain scoping/examples/domains/minecraft3/minecraft-contrived3.pddl --prob scoping/examples/domains/minecraft3/prob_make_bed_irrel.pddl 1>&3 2>&4; } 2>&1 )  # Captures time only.
        exec 3>&- 4>&-
        sruntimes+=("$foo")
        let COUNTER-=1
    done
fi

declare -a enhsp_runtimes
COUNTER="$user_arg + 1"
until [ $COUNTER -lt 1 ]; do
    echo RUN_NUMBER $COUNTER
    exec 3>&1 4>&2
    foo=$( { time enhsp-2020 -o scoping/examples/domains/minecraft3/minecraft-contrived3.pddl -f scoping/examples/domains/minecraft3/prob_make_bed_irrel.pddl -planner opt-hmax 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    enhsp_runtimes+=("$foo")
    let COUNTER-=1
done

declare -a senhsp_runtimes
COUNTER="$user_arg + 1"
until [ $COUNTER -lt 1 ]; do
    echo RUN_NUMBER $COUNTER
    exec 3>&1 4>&2
    foo=$( { time enhsp-2020 -o scoping/examples/domains/minecraft3/minecraft-contrived3_scoped_prob_make_bed_irrel.pddl -f scoping/examples/domains/minecraft3/prob_make_bed_irrel_scoped.pddl -planner opt-hmax 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    senhsp_runtimes+=("$foo")
    let COUNTER-=1
done

if [ -n "$2" ]
then
    echo
    echo TIME_TAKEN_FOR_TRANSLATING_AND_SCOPING_EACH_TRIAL
    # Print the times taken for scoping valid trials
    echo "${sruntimes[@]:1}"
fi
echo
echo TIME_TAKEN_FOR_RAW_PLANNING_FOR_EACH_TRIAL
# Print the times taken for raw planning on all valid trials
echo "${enhsp_runtimes[@]:1}"
echo
echo TIME_TAKEN_FOR_SCOPED_PLANNING_FOR_EACH_TRIAL
echo "${senhsp_runtimes[@]:1}"