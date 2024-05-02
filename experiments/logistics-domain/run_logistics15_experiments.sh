#!/bin/bash

# Usage instructions: see README.md
TIMEFORMAT=%R
# array of translation runtimes
declare -a truntimes
user_arg="$1"
echo ""

# The counter needs to be 1 + num_trials you want for the experiment, because the first trial is slightly slower due to needing to execute
# some of the setup bash commands. Thus, take all readings except the first!
let COUNTER=$user_arg+1
until [ $COUNTER -lt 1 ]; do
    echo RUN_NUMBER $COUNTER
    # This exec is redirecting file descriptor 3 to  standard out and 4 to standard error
    exec 3>&1 4>&2
    # This will print out only the time of this command. We do this because otherwise, stdout gets full of other information, making it hard to find time
    foo=$( { time python scoping/downward_translate/translate_and_scope.py examples/IPC_domains_propositional/logistics00/domain.pddl examples/IPC_domains_propositional/logistics00/prob15.pddl --sas-file logistics15.sas 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    truntimes+=("$foo")
    let COUNTER-=1
done

declare -a sruntimes
# Array of scoping runtimes
let COUNTER=$user_arg+1
until [ $COUNTER -lt 1 ]; do
    echo RUN_NUMBER $COUNTER
    exec 3>&1 4>&2
    foo=$( { time python scoping/downward_translate/translate_and_scope.py examples/IPC_domains_propositional/logistics00/domain.pddl examples/IPC_domains_propositional/logistics00/prob15.pddl --sas-file logistics15.sas --scope True 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    sruntimes+=("$foo")
    let COUNTER-=1
done

# Array of planning on unscoped domain runtimes
declare -a fd_runtimes
let COUNTER=$user_arg+1
until [ $COUNTER -lt 1 ]; do
    echo RUN_NUMBER $COUNTER
    exec 3>&1 4>&2
    foo=$( { time python $2 --alias seq-opt-lmcut logistics15.sas 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    fd_runtimes+=("$foo")
    let COUNTER-=1
done

# Array of planning on scoped domain runtimes
declare -a sfd_runtimes
let COUNTER=$user_arg+1
until [ $COUNTER -lt 1 ]; do
    echo RUN_NUMBER $COUNTER
    exec 3>&1 4>&2
    foo=$( { time python $2 --alias seq-opt-lmcut logistics15_scoped.sas 1>&3 2>&4; } 2>&1 )  # Captures time only.
    exec 3>&- 4>&-
    sfd_runtimes+=("$foo")
    let COUNTER-=1
done

echo
echo TIME_TAKEN_FOR_TRANSLATING_FOR_EACH_TRIAL
# Print the times taken for scoping valid trials
echo "${truntimes[@]:1}"
echo
echo TIME_TAKEN_FOR_TRANSLATING_AND_SCOPING_EACH_TRIAL
# Print the times taken for scoping valid trials
echo "${sruntimes[@]:1}"
echo
echo TIME_TAKEN_FOR_RAW_PLANNING_FOR_EACH_TRIAL
# Print the times taken for raw planning on all valid trials
echo "${fd_runtimes[@]:1}"
echo
echo TIME_TAKEN_FOR_SCOPED_PLANNING_FOR_EACH_TRIAL
echo "${sfd_runtimes[@]:1}"
