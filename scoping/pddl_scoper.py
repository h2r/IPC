from oo_scoping.utils import get_atoms, get_unique_z3_vars, pvars2objects
from oo_scoping.writeback_pddl import (
    get_scoped_domain_path,
    get_scoped_problem_path,
    writeback_problem,
    writeback_domain,
)
from oo_scoping.scoping import scope
from oo_scoping.PDDLz3 import PDDL_Parser_z3
import argparse


def scope_pddl(domain: str, problem: str, **kwargs: dict) -> None:
    """
    :param domain: Path of domain file
    :param problem: Path of problem file
    """
    parser = PDDL_Parser_z3()
    parser.parse_domain(domain)
    parser.parse_problem(problem)

    skill_list = parser.get_skills()

    # This below block converts all the domain's goals to z3
    goal_cond = parser.get_goal_cond()

    # This below block converts all the domain's initial conditions to z3
    init_cond_list = parser.get_init_cond_list()

    # Run the scoper on the constructed goal, skills and initial condition
    rel_pvars, cl_pvars, rel_skills = scope(
        goals=goal_cond, skills=skill_list, start_condition=init_cond_list, **kwargs
    )

    all_pvars = []
    for s in skill_list:
        all_pvars.extend(get_atoms(s.precondition))
        all_pvars.extend(s.params)
    # Remove duplicates from all_pvars
    all_pvars = get_unique_z3_vars(all_pvars)

    all_objects = pvars2objects(all_pvars)
    rel_objects = pvars2objects(rel_pvars)
    cl_objects = pvars2objects(cl_pvars)
    objects2remove_keep_cl = [
        x for x in all_objects if x not in (rel_objects + cl_objects)
    ]
    objects2remove_remove_cl = [x for x in all_objects if x not in rel_objects]
    # Keep CL
    scoped_problem_path = get_scoped_problem_path(problem, suffix="with_cl")
    writeback_problem(problem, scoped_problem_path, objects2remove_keep_cl)
    # Remove CL
    scoped_problem_path = get_scoped_problem_path(problem, suffix="sans_cl")
    writeback_problem(problem, scoped_problem_path, objects2remove_remove_cl)

    all_actions = sorted(list(set([a.name for a in parser.actions])))
    relevant_actions = []
    for s in rel_skills:
        if isinstance(s.action, str):
            relevant_actions.append(s.action)
        else:
            relevant_actions.extend(s.action)
    relevant_actions = sorted(list(set(relevant_actions)))
    irrel_actions = [a for a in all_actions if a not in relevant_actions]

    scoped_domain_path = get_scoped_domain_path(domain, problem)
    writeback_domain(domain, scoped_domain_path, irrel_actions)
    print('All actions:', len(all_actions))
    print('Relevant actions:', len(relevant_actions))
    print('All objects:', len(all_objects))
    print('Relevant (+CL) objects:', len(all_objects) - len(objects2remove_keep_cl))
    print('All grounded actions:', len(skill_list))
    print('Relevant grounded actions:', len(rel_skills))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str, help="Path location of the PDDL domain file"
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Path location of the PDDL problem file corresponding to the specified domain",
    )
    args = parser.parse_args()
    scope_pddl(args.domain, args.problem, **{"verbose": 1})
