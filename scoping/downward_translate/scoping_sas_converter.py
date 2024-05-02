from __future__ import annotations
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable

import z3, re

from oo_scoping.skill_classes import EffectTypePDDL, SkillPDDL
from oo_scoping.utils import (
    product_dict,
    nested_list_replace,
    get_atoms,
    get_unique_z3_vars,
)
from oo_scoping.action import Action
from oo_scoping.PDDLz3 import compile_expression
from oo_scoping.PDDL import PDDL_Parser, Action
from oo_scoping.downward_translate import sas_tasks

"""
General Notes on current implementation of SAS+ -> z3 for scoping purposes.

- All pvars are treated as ints
- Actions only have "positive effects" as a result

"""

# TODO:
#  Write type-sig and function specification for all the functions below.


def make_str2var_dict(sas_vars: sas_tasks.SASVariables):
    """
    Returns a dict of str(var_names) -> z3Var(var_names)

    :param sas_vars - a SASVariables Object corresponding to all the variables in this particular problem
    Output - a new dictionary mapping var_name strings to z3 vars that are created for each variable

    Note: Currently, all variables are treated as z3.Ints. This is because SAS+ essentially treats everything as an enum.
    We could get bools by simply uncommenting the commented out block below, but will need to deal with this downstream.

    """
    str2var = OrderedDict()
    for i in range(len(sas_vars.ranges)):
        #     if sas_vars.ranges[i] == 2:
        #         z3_func = z3.Bool
        #     elif sas_vars.ranges[i] > 2:
        #         z3_func = z3.Int
        #     else:
        #         print("There's some SAS+ variable with a domain of 0 or 1. We probably can handle this by just treating this as a Bool, but \
        #             I'm setting a breakpoint here just in case so you're aware of this variable and there's no silent failure...")
        #         from IPython import embed; embed()
        #         exit(1)
        z3_func = z3.Int
        var_str = "var" + str(i) + "()"
        z3_var = z3_func(var_str)
        str2var[var_str] = z3_var

    return str2var


def make_str_grounded_actions(sas_ops):
    """
    Returns a list oo_scoping.action.Action variables

    :param sas_ops - a list of SASOperator objects corresponding to all the actions in this particular problem
    Output - a non-nested list of oo_scoping.action.Action variables representing these operators

    TODO: Currently, we're not grouping the actions in any way (and it's unclear if we should), but this list
    might be a good place to do it if we wanted
    """
    list_of_actions = []
    for op in sas_ops:
        # First, parse out all the preconditions (including the one mentioned in the effect line)
        op_precond_list = op.get_applicability_conditions()
        action_precond_list = []
        for precond in op_precond_list:
            var_num, var_val = precond
            action_precond_list.append(["=", ["var" + str(var_num)], str(var_val)])
        # Next, extract the effect
        pre_post_conds_list = op.pre_post
        action_effect_list = []
        for pre_post_cond in pre_post_conds_list:
            var_num, _, var_val, cond_list = pre_post_cond
            if len(cond_list) > 0:
                print(
                    "The pre_post cond of this operator has some cond list, and I haven't thought about \
                    what to do in the case where this is non-empty. Worth looking into"
                )
                from IPython import embed

                embed()
                exit(1)
            else:
                action_effect_list.append(
                    ["assign", ["var" + str(var_num)], str(var_val)]
                )
        # Finally construct an Action and append it to the list to be returned
        list_of_actions.append(
            Action(op.name, [], action_precond_list, [], action_effect_list, [])
        )
    return list_of_actions


def make_init_cond_list(sas_init_vals, str2var_dict):
    """
    Returns a list of z3 exprs representing initial conditions

    :param sas_init_vals - a list of ints. The len of this list should be the number of state-variables in the
    SAS+ problem, and each value in the list should be the initial state value of the state-variable corresponding to
    that particular index in the list. This can be obtained from an instantiated SASInit.values
    :param str2var_dict - a dict of str(var_names) -> z3Var(var_names)

    Output - a list of z3 exprs representing the initial state
    """
    str_init_cond_list = []
    for i in range(len(sas_init_vals)):
        str_init_cond_list.append(["=", ["var" + str(i)], str(sas_init_vals[i])])
    z3_init_conds = [
        compile_expression(init_cond, str2var_dict) for init_cond in str_init_cond_list
    ]
    return z3_init_conds


def make_goal_cond(sas_goal_pairs, str2var_dict):
    """
    Returns a z3 And expr representing goal conditions

    :param sas_init_vals - a list of tuples corresponding to the goal conditions. This can be obtained from an instantiated SASGoal.pairs
    :param str2var_dict - a dict of str(var_names) -> z3Var(var_names)

    Output - a z3 And expr representing the goal conds
    """
    str_goal_list = []
    for goal_cond in sas_goal_pairs:
        var_num, var_val = goal_cond
        str_goal_list.append(["=", ["var" + str(var_num)], str(var_val)])
    goal_cond_list = [
        compile_expression(goal_cond_str, str2var_dict)
        for goal_cond_str in str_goal_list
    ]
    goal_cond = z3.And(*goal_cond_list)
    return goal_cond


# Note: Function taken directly from PDDLz3.py
def list2var_str(x):
    return x[0] + "(" + ", ".join(x[1:]) + ")"


# Note: Function taken directly from PDDLz3.py
def action2precondition(a, str_var_dict):
    clauses = [
        compile_expression(p, str_var_dict) for p in a.positive_preconditions
    ] + [z3.Not(compile_expression(p, str_var_dict)) for p in a.negative_preconditions]
    return z3.And(*clauses)


# Note: Function taken directly from PDDLz3.py
def action2effect_types(a, str_var_dict, parser=None):
    effect_types = []
    for eff in a.add_effects:
        # Check for complex effects like 'increase'
        if eff[0] in ["increase", "decrease", "assign"]:
            effected_var = str_var_dict[list2var_str(eff[1])]
            effect_details = compile_expression(eff, str_var_dict, parser)
            params = tuple(
                [
                    x
                    for x in get_atoms(effect_details)
                    if not z3_identical(x, effected_var)
                ]
            )
            effect_type = EffectTypePDDL(effected_var, effect_details, params=params)
            effect_types.append(effect_type)
        else:
            effected_var = compile_expression(eff, str_var_dict, parser)
            # This may accidentally cause different EfectTypes to be identified bc True == 1.
            effect_details = True
            effect_type = EffectTypePDDL(effected_var, effect_details)
            effect_types.append(effect_type)
    # Assume for now that del_effects only sets bools to false
    for eff in a.del_effects:
        effected_var = compile_expression(eff, str_var_dict, parser)
        # This may accidentally cause different EfectTypes to be identified bc True == 1.
        effect_details = False
        effect_type = EffectTypePDDL(effected_var, effect_details)
        effect_types.append(effect_type)
    return tuple(effect_types)


# This function makes all the actual CAE triples
def str_grounded_actions2skills(str_grounded_actions, str2var_dict):
    skill_list = []
    for grounded_action in str_grounded_actions:
        precond = action2precondition(grounded_action, str2var_dict)
        effect_types = action2effect_types(grounded_action, str2var_dict)
        skill = SkillPDDL(precond, grounded_action.name, effect_types)
        skill_list.append(skill)
    return skill_list
