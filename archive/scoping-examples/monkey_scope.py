
import sys, pprint, time, itertools, re, copy
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable

import z3

from scoping.action import Action
from scoping.PDDLz3 import PDDL_Parser_z3
from scoping.skill_classes import EffectTypePDDL, SkillPDDL
from scoping.utils import product_dict, nested_list_replace, get_atoms, get_all_objects\
    , condition_str2objects, pvars2objects, get_unique_z3_vars
from scoping.writeback_pddl import writeback_problem, writeback_domain, \
    get_scoped_problem_path, get_scoped_domain_path,
from scoping.scoping import scope

# For profiling
import cProfile
import pstats
from pstats import SortKey




def scope_pddl(domain, problem):
    parser = PDDL_Parser_z3()
    parser.parse_domain(domain)
    parser.parse_problem(problem)

    skill_list = parser.get_skills()

    # This below block converts all the domain's goals to z3
    goal_cond = parser.get_goal_cond()

    # This below block converts all the domain's initial conditions to z3
    init_cond_list = parser.get_init_cond_list()
   
    

    # Run the scoper on the constructed goal, skills and initial condition
    rel_pvars, cl_pvars, rel_skills = scope(goals=goal_cond, skills=skill_list, start_condition=init_cond_list)
    # rel_pvars = rel_pvar  s + cl_pvars #TODO don't just merge them
    # print("~~~~~Relevant skills~~~~~")
    # print("\n\n".join(map(str,rel_skills)))
    # print("~~~~~Relevant pvars~~~~~")
    # for p in rel_pvars:
    #     print(p)
 
    # print(rel_pvars)
    # print(rel_skills)

    
    all_pvars = []
    for s in skill_list:
        all_pvars.extend(get_atoms(s.precondition))
        all_pvars.extend(s.params)
    def get_unique_pvars_in_scope_pddl(all_pvars):
        """
        Wrapper used to view this call in a profiler
        """
        return get_unique_z3_vars(all_pvars)
    all_pvars = map(str,all_pvars)
    all_pvars = sorted(list(set(all_pvars)))
    # all_pvars = get_unique_pvars_in_scope_pddl(all_pvars)
    irrel_pvars = [p for p in map(str,all_pvars) if p not in map(str,rel_pvars)]
    all_objects = pvars2objects(all_pvars)
    rel_objects = pvars2objects(rel_pvars)
    irrel_objects = [x for x in all_objects if x not in rel_objects]

    # print(f"Irrelevant objects:")
    # print("\n".join(irrel_objects))
    scoped_problem_path = get_scoped_problem_path(problem)
    writeback_problem(problem, scoped_problem_path, irrel_objects)

    all_actions = sorted(list(set([a.name for a in parser.actions])))
    relevant_actions = []
    for s in rel_skills:
        if isinstance(s.action,str):
            relevant_actions.append(s.action)
        else:
            relevant_actions.extend(s.action)
    relevant_actions = sorted(list(set(relevant_actions)))
    irrel_actions = [a for a in all_actions if a not in relevant_actions]
    # print("~~~~~~~Irrel actions~~~~~~~")
    # for a in irrel_actions: print(a)
    scoped_domain_path = get_scoped_domain_path(domain, problem)
    writeback_domain(domain, scoped_domain_path, irrel_actions)

if __name__ == '__main__':
    start_time = time.time()
    prob_num = '07'
    domain, problem = f"domains/multi_monkeys_playroom/multi_monkeys_playroom.pddl", f"domains/multi_monkeys_playroom/prob-{prob_num}.pddl"
    # Path to save entire, non-human-readable profile object
    profile_path = f'time_profiles/monkey_{prob_num}'
    # Path to save human-readable profile stats
    profile_path_txt = profile_path + ".txt"
    # Run scoper and profile it, saving the profile to profile_path
    cProfile.run('scope_pddl(domain, problem)', profile_path)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    # Get the file object for outputting human-readable profile stats
    stats_readable_file = open(profile_path_txt,"w")
    # Get stats object from saves profile, and set it to stream output to stats_readable_file
    p = pstats.Stats(profile_path, stream = stats_readable_file)
    # Save output to stats_reasable_file
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
