"""This script was used to get state var counts for PDDL domains for the IJCAI 2023 response."""
from scoping.PDDLz3 import PDDL_Parser
from dataclasses import dataclass
from typing import List


@dataclass
class StateVarCount:
    experiment_name: str
    n_unscoped: int
    n_scoped: int

    def __repr__(self) -> str:
        return f"{self.experiment_name}: {self.n_unscoped} -> {self.n_scoped}"


def pddl_paths_to_state_variable_count(domain: str, problem: str) -> int:
    """Get the count of state variables from domain and problem path"""
    parser = PDDL_Parser()
    parser.parse_domain(domain)
    parser.parse_problem(problem)
    return parser.get_state_variable_count()


def get_ipc_composite_stat_var_counts() -> List[StateVarCount]:
    ipc_dir = "scoping/examples/domains/IPC_Domains/CompositeIPC"
    ipc_composite_domain_path = f"{ipc_dir}/ipc_composite.pddl"
    ipc_names = {
        "depot": "prob-01",
        "satellite": "prob-02",
        "driverlog": "prob-03",
        "zenotravel": "prob-04",
    }
    counts = []
    for experiment_name, problem_name in ipc_names.items():
        unscoped_problem = f"{ipc_dir}/{problem_name}.pddl"
        # scoped_problem = f"{ipc_dir}/{problem_name}_scoped_sans_cl.pddl"
        scoped_problem = f"{ipc_dir}/{problem_name}_scoped.pddl"
        scoped_domain = f"{ipc_dir}/ipc_composite_scoped_{problem_name}.pddl"
        n_unscoped = pddl_paths_to_state_variable_count(
            ipc_composite_domain_path, unscoped_problem
        )
        n_scoped = pddl_paths_to_state_variable_count(scoped_domain, scoped_problem)
        counts.append(StateVarCount(experiment_name, n_unscoped, n_scoped))
    return counts


def get_playroom_state_var_counts() -> List[StateVarCount]:
    """Get state var counts for Monkey Playroom domains"""
    dir = "examples/multi_monkeys_playroom"
    unscoped_domain_path = f"{dir}/multi_monkeys_playroom.pddl"
    problem_numbers = [1, 3, 5, 7, 9]
    counts = []
    for problem_number in problem_numbers:
        unscoped_problem = f"{dir}/prob{problem_number:02}.pddl"
        scoped_problem = f"{dir}/prob{problem_number:02}_scoped.pddl"
        scoped_domain = (
            f"{dir}/multi_monkeys_playroom_scoped_prob{problem_number:02}.pddl"
        )
        counts.append(
            StateVarCount(
                f"Playroom {problem_number}",
                pddl_paths_to_state_variable_count(
                    unscoped_domain_path, unscoped_problem
                ),
                pddl_paths_to_state_variable_count(scoped_domain, scoped_problem),
            )
        )
    return counts


def get_minecraft_state_var_counts() -> List[StateVarCount]:
    dir = "scoping/examples/domains/minecraft3/"
    unscoped_domain = f"{dir}/minecraft-contrived3.pddl"
    problem_names = ["make_wooden_planks", "get_dyed_wool", "make_bed"]
    counts = []
    for problem_name in problem_names:
        unscoped_problem = f"{dir}/prob_{problem_name}_irrel.pddl"
        scoped_problem = f"{dir}/prob_{problem_name}_irrel_scoped.pddl"
        scoped_domain = (
            f"{dir}/minecraft-contrived3_scoped_prob_{problem_name}_irrel.pddl"
        )
        counts.append(
            StateVarCount(
                f"Minecraft ({problem_name})",
                pddl_paths_to_state_variable_count(unscoped_domain, unscoped_problem),
                pddl_paths_to_state_variable_count(scoped_domain, scoped_problem),
            )
        )
    return counts


def get_all_counts() -> List[StateVarCount]:
    counts = (
        get_playroom_state_var_counts()
        + get_ipc_composite_stat_var_counts()
        + get_minecraft_state_var_counts()
    )
    return counts


if __name__ == "__main__":
    for x in get_all_counts():
        print(x)
