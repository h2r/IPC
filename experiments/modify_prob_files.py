from scoping.PDDL import PDDL_Parser
import argparse
import itertools
import os
import glob

def find_new_irrel_goal_clauses(domain, problem, **kwargs):
    """
    :param domain: Path of domain file
    :param problem: Path of problem file
    """
    parser = PDDL_Parser()
    parser.parse_domain(domain)
    parser.parse_problem(problem)

    # First, find the predicates that appear in the goal.
    goal_pred_names = set()
    for g_clause in parser.positive_goals + parser.negative_goals:
        goal_pred_names.add(g_clause[0])

    # Next, find all initial conditions that use these predicates.
    init_conds_with_goal_pred = []
    for init_clause in parser.state:
        if init_clause[0] in goal_pred_names:
            init_conds_with_goal_pred.append(init_clause)

    # Finally, find the initial conditions that are not already
    # goal atoms.
    conditional_irrel_candidates = []
    for init_clause in init_conds_with_goal_pred:
        if init_clause not in parser.positive_goals + parser.negative_goals:
            conditional_irrel_candidates.append(init_clause)

    # Yield all possible combos of conditionally-irrelevant candidates
    # such that the combo length is the number of candidates minus 1.
    # (arbitrary choice on length)
    # for combo in itertools.combinations(conditional_irrel_candidates, len(conditional_irrel_candidates) - 1):
    for combo in itertools.combinations(conditional_irrel_candidates, 1):
        yield combo


def write_irrel_prob_files(domain, problem, **kwargs):
    domain_name = domain.split('/')[-2]
    problem_name = problem.split('/')[-1]
    save_folder = f"randomly_generated_prob_files/{domain_name}/{problem_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    current_problem_files = glob.glob(save_folder+ "/*")
    i = 0
    # find an i to continue from
    for i in range(10000):
        output_filename = f"{save_folder}/irrel{str(i)}.pddl"
        if output_filename not in current_problem_files:
            break
    for goal_conds in find_new_irrel_goal_clauses(domain, problem):
        output_filename = f"{save_folder}/irrel{str(i)}.pddl"
        wf = open(output_filename, "w+")
        lines_to_be_written = []
        with open(problem, 'r') as rf:
            lines = rf.readlines()
            in_goal = False
            for line in lines:
                if in_goal:
                    for goal_cond in goal_conds:
                        new_goal_cond_str = '\t(' + ' '.join(goal_cond) + ')\n'
                        lines_to_be_written.append(new_goal_cond_str)
                    in_goal = False
                if "goal" in line.strip():
                    in_goal = True
                lines_to_be_written.append(line)
        wf.writelines(lines_to_be_written)
        wf.seek(0)
        wf.close()
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",type=str,help="Path location of the PDDL domain file")
    parser.add_argument("--problem",type=str,help="Path location of the PDDL problem file corresponding to the specified domain")
    args = parser.parse_args()
    write_irrel_prob_files(args.domain, args.problem, **{'verbose': 1})
