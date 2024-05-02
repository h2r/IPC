import re, os, copy
from typing import Optional, List


def get_scoped_problem_path(p: str, suffix: Optional[str] = None) -> str:
    p_split = p.split(".")
    base = ".".join(p_split[:-1])
    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + suffix
    return base + "_scoped" + suffix + "." + p_split[-1]


def get_scoped_domain_path(d: str, p: str, suffix: Optional[str] = None) -> str:
    d_split = d.split(".")
    base = ".".join(d_split[:-1])
    # p_id = re.search("([0-9]+)",p).group()
    p_name = p.split("/")[-1]
    p_base = ".".join(p_name.split(".")[:-1])

    if suffix is None:
        suffix = ""
    else:
        suffix = "_" + suffix
    d_new = base + "_" + "scoped" + "_" + p_base + suffix + "." + d_split[-1]
    return d_new


def find_closing_paren(s: str, start: int) -> int:
    lefts_left = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == ")":
            lefts_left -= 1
            if lefts_left == 0:
                return i
        elif c == "(":
            lefts_left += 1


def writeback_domain(input_path: str, output_path: str, actions: List[str]) -> None:
    with open(input_path, "r") as f:
        domain = f.read()
    # TODO find any of a list of action names
    for action in actions:
        action_prefix = "\(:action " + action
        action_start = re.search(action_prefix, domain).start()
        action_end = find_closing_paren(domain, action_start) + 1
        action_lines = domain[action_start:action_end].splitlines()
        action_lines = [";" + l for l in action_lines]
        action_commented = "\n".join(action_lines)
        domain = domain[:action_start] + action_commented + domain[action_end:]
    with open(output_path, "w") as f:
        f.write(domain)


# TODO clean this up, maybe make less janky
def writeback_problem(input_path: str, output_path: str, objects: List[str]) -> None:
    with open(input_path, "r") as f:
        instance_lines = f.read().splitlines()
    scoped_lines = []
    in_objects_flag = False
    for l in instance_lines:
        if in_objects_flag == True:
            l_stripped = l.replace("\t", "").replace(" ", "")
            if len(l_stripped) > 0 and l_stripped[0] == ")":
                in_objects_flag = False
        elif "(:objects" in l:
            in_objects_flag = True

        tokens = re.split("[ (),]", l)
        tokens[0] = tokens[0].strip("\t")

        if not any(o in tokens for o in objects):
            scoped_lines.append(l)

        else:
            # from IPython import embed; embed()
            if in_objects_flag:
                split_l = l.split(" ")
                obj_type = split_l[-1]
                objs = [o.replace("\t", "") for o in split_l[:-2]]
                objs_kept = []
                objs_removed = []
                # objs = [o for o in objs if o not in objects and o != '']
                for o in objs:
                    if o in objects:
                        objs_removed.append(o)
                    else:
                        objs_kept.append(o)
                objs_kept = [i for i in objs_kept if i]
                if len(objs_kept) > 0:
                    l_new = "\t" + " ".join(objs_kept) + " - " + obj_type
                    scoped_lines.append(l_new)
                if len(objs_removed) > 0:
                    l_new = ";\t" + " ".join(objs_removed) + " - " + obj_type
                    scoped_lines.append(l_new)
            else:
                # print(l)
                scoped_lines.append(";" + l)

    with open(output_path, "w") as f:
        f.write("\n".join(scoped_lines))
