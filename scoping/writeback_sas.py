# NOTE: Right now, even though Scoping recognizes various state-variables are irrelevant, they cannot be deleted from the problem
# This is because the action section, etc. refer to the particular number of the variable (i.e 1 -> var1), and if we delete some variable
# this numbering will change (i.e, deleting var0 from the problem would make var1 now var0, which will confuse everything else...)
# Thus, we also cannot prune the initial state


def prune_goal_conds(i_line, sas_domain_lines, rel_pvars):
    goal_cond_list = []
    while "end_goal" not in sas_domain_lines[i_line]:
        goal_vars_and_vals = sas_domain_lines[i_line].split(" ")
        if "var" + goal_vars_and_vals[0] in rel_pvars:
            goal_cond_list.append(sas_domain_lines[i_line])
        i_line += 1
    return goal_cond_list, i_line


def writeback_scoped_sas(rel_ops, rel_pvars, problem_file_path, scoped_path=None):
    """
    Function that writes out a scoped SAS+ problem file

    :param rel_ops - a set of strings of the names of all the relevant operators
    :param rel_pvars - a set of strings of all the relevant pvars
    :param problem_file_path - a string representing the path to the written SAS+ file

    Output: (void), just writes out the file

    """
    scoped_domain_lines = []
    skip_lines = False
    first_operator_encountered = False
    # inside_init_state = False
    # lines_inside_init_state_counter = 0

    with open(problem_file_path, "r") as f:
        sas_domain_lines = f.readlines()

        i_line = 0
        while i_line < len(sas_domain_lines):
            if "begin_operator" in sas_domain_lines[i_line]:
                # Replace the count of operators.
                # We do this here because the operator count is the line before the first operator.
                if not first_operator_encountered:
                    scoped_domain_lines.pop(-1)
                    scoped_domain_lines.append(str(len(rel_ops)) + "\n")
                    first_operator_encountered = True
                # If the next line after a begin_operator statement is
                # not in the rel_ops set, then just keep skipping lines until
                # you hit the next end_line
                # Note: we stop at -1 to exclude the '\n' character
                # This may cause odd bugs depending on OS or changes to the SAS+ writer. It may be safer to use .strip()
                if sas_domain_lines[i_line + 1][:-1] not in rel_ops:
                    skip_lines = True
                else:
                    scoped_domain_lines.append(sas_domain_lines[i_line])
                i_line += 1
            elif "end_operator" in sas_domain_lines[i_line] and skip_lines:
                # If skip_lines is true and you see 'end_operator', start appending
                # from the next line onwards
                skip_lines = False
                i_line += 1

            # # The line below 'end_metric' is the number of state variables in the
            # # problem, so write this out there.
            # elif "end_metric" in sas_domain_lines[i_line]:
            #     scoped_domain_lines.append(sas_domain_lines[i_line])
            #     scoped_domain_lines.append(str(len(rel_pvars))+'\n')
            #     i_line += 2
            # elif "begin_variable" in sas_domain_lines[i_line]:
            #     # If the next line after a begin_variable statement is
            #     # not in the rel_pvars set, then just keep skipping lines until
            #     # you hit the next end_line
            #     if sas_domain_lines[i_line + 1][:-1] not in rel_pvars:
            #         skip_lines = True
            #     else:
            #         scoped_domain_lines.append(sas_domain_lines[i_line])
            #     i_line += 1
            # elif "end_variable" in sas_domain_lines[i_line] and skip_lines:
            #     # If skip_lines is true and you see 'end_variable', start appending
            #     # from the next line onwards
            #     skip_lines = False
            #     i_line += 1
            # elif "begin_state" in sas_domain_lines[i_line]:
            #     # If you see 'begin_state', then set the inside_init_state bool flag to True
            #     inside_init_state = True
            #     scoped_domain_lines.append(sas_domain_lines[i_line])
            #     i_line += 1
            # elif inside_init_state and "end_state" not in sas_domain_lines[i_line]:
            #     # Once inside the init state, delete lines that don't correspond to relevant pvars
            #     if ('var'+str(lines_inside_init_state_counter)) in rel_pvars:
            #         scoped_domain_lines.append(sas_domain_lines[i_line])
            #     i_line += 1
            #     lines_inside_init_state_counter += 1
            # elif "end_state" in sas_domain_lines[i_line] and inside_init_state:
            #     # Set the 'inside_init_state' bool flag to false
            #     inside_init_state = False
            #     scoped_domain_lines.append(sas_domain_lines[i_line])
            #     i_line += 1

            elif "begin_goal" in sas_domain_lines[i_line]:
                # Append the begin_goal statement
                scoped_domain_lines.append(sas_domain_lines[i_line])
                i_line += 1
                # Get a list of all the relevant goal conditions
                rel_goal_conds, i_line = prune_goal_conds(
                    i_line, sas_domain_lines, rel_pvars
                )
                # Add the new number of goal conditions to scoped_domain_lines
                scoped_domain_lines.append(str(len(rel_goal_conds)) + "\n")
                # Add all the goal conds
                scoped_domain_lines.extend(rel_goal_conds)
                # Add the end_goal statement to scoped_domain_lines
                scoped_domain_lines.append(sas_domain_lines[i_line])
                i_line += 1

            else:
                if not skip_lines:
                    scoped_domain_lines.append(sas_domain_lines[i_line])
                i_line += 1

    # The filepath will always be <file_name>.sas at the end, so create a file
    # named <file_name>_scoped.sas
    if scoped_path is None:
        scoped_path = problem_file_path[:-4] + "_scoped" + problem_file_path[-4:]
    with open(scoped_path, "w+") as f_scoped:
        f_scoped.writelines(scoped_domain_lines)
