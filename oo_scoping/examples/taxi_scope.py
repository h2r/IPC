import time
import sys, pprint
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable
import re, copy
import itertools
import z3
from oo_scoping.skill_classes import EffectTypePDDL, SkillPDDL
from oo_scoping.utils import product_dict, nested_list_replace, get_atoms, get_all_objects, condition_str2objects
from oo_scoping.scoping import scope
from oo_scoping.PDDLz3 import PDDL_Parser_z3
from oo_scoping.action import Action


def remove_objects(input_path, output_path, objects):
    with open(input_path, "r") as f:
        instance_lines = f.read().splitlines()
    scoped_lines = []
    in_objects_flag = False
    for l in instance_lines:
        if in_objects_flag == True:
            if len(l) > 0 and l[0] == ")":
                in_objects_flag = False
        if "(:objects" in l:
            in_objects_flag = True
        
        if not any(o in l for o in objects):
            scoped_lines.append(l)
        else:
            if in_objects_flag:
                split_l = l.split(" ")
                obj_type = split_l[-1]
                objs = [o for o in split_l[:-2] if o not in objects and o != '']
                if len(objs) > 0:
                    l_new = " ".join(objs) + " - " + obj_type
                    scoped_lines.append(l_new)
            else:
                scoped_lines.append(";" + l)

    with open(output_path, "w")  as f:
        f.write("\n".join(scoped_lines))

def get_scoped_path(p):
    p_split = p.split(".")
    base = ".".join(p_split[:-1])
    return base + "_scoped." + p_split[-1]

if __name__ == '__main__':
    # zeno_dom = "domains/zeno/zeno.pddl"
    # zeno_prob = "domains/zeno/pb1.pddl"
    # domain, problem = zeno_dom, zeno_prob

    # taxi_dom = "domains/infinite-taxi-numeric/taxi-domain.pddl"
    # taxi_prob = "domains/infinite-taxi-numeric/prob02.pddl"

    start_time = time.time()
    domain =  "domains/existential-taxi/taxi-domain.pddl"
    problem =  "domains/existential-taxi/prob-02.pddl"

    parser = PDDL_Parser_z3()
    parser.parse_domain(domain)
    parser.parse_problem(problem)

    skill_list = parser.get_skills()

    # This below block converts all the domain's goals to z3
    goal_cond = parser.get_goal_cond()

    # This below block converts all the domain's initial conditions to z3
    init_cond_list = parser.get_init_cond_list()

    # Run the scoper on the constructed goal, skills and initial condition
    rel_pvars, rel_skills = scope(goals=goal_cond, skills=skill_list, start_condition=init_cond_list)
      
    print("~~~~~Relevant skills~~~~~")
    print("\n\n".join(map(str,rel_skills)))
    print("~~~~~Relevant pvars~~~~~")
    for p in rel_pvars:
        print(p)
 
    # print(rel_pvars)
    # print(rel_skills)

    def pvars2objects(pvars):
        objs = condition_str2objects(map(str,pvars))
        objs = [s.strip() for s in objs]
        objs = sorted(list(set(objs)))
        return objs
    all_pvars = []
    for s in skill_list:
        all_pvars.extend(get_atoms(s.precondition))
        all_pvars.extend(s.params)
    all_objects = pvars2objects(all_pvars)
    rel_objects = pvars2objects(rel_pvars)
    irrel_objects = [x for x in all_objects if x not in rel_objects]

    print(f"Relevant objects:")
    print("\n".join(rel_objects))
    remove_objects(problem, get_scoped_path(problem), irrel_objects)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")