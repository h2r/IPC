#!/bin/bash

# This is a bash script to showcase a simple example of how Task Scoping is able to scope some causally-unconnected irrelevance that
# FD alone cannot.
# Running this from the root of the directory will produce two .sas files: one ('gripper_painting.sas') that is pruned only by
# FD and another ('gripper-painting_scoped.sas') that is pruned by both FD and Task Scoping. Notice how the latter file has less operators:
# these correspond to the operators that change the color (a causally-unconnected state-variable) of the different balls.
# Thus, Task Scoping is able to prune operators corresponding to the causally-unconnected ball-color state-variable, while FD cannot.
python scoping/downward_translate/translate_and_scope.py examples/gripper-painting-domain/domain.pddl examples/gripper-painting-domain/prob04.pddl --sas-file gripper-painting.sas --scope True
