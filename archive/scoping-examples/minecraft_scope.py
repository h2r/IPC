import sys, pprint
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable
import re, copy
import time
import pathlib

from oo_scoping.scoping import scope
from oo_scoping.pddl_scoper import scope_pddl
from oo_scoping.PDDLz3 import PDDL_Parser_z3

# For profiling
import cProfile
import pstats
from pstats import SortKey

profile_dir = pathlib.Path(__file__).parent.parent.parent.absolute() / "time_profiles"

if __name__ == "__main__":
    start_time = time.time()
    domain = "domains/minecraft2/minecraft-contrived2.pddl"
    # problems = ["domains/minecraft2/prob_obsidian_with_pick.pddl", "domains/minecraft2/prob_obsidian_without_pick.pddl"]
    problems = [
        # "domains/minecraft2/prob_irrel_obsidian_with_pick.pddl"
        # "domains/minecraft2/prob_irrel_flint_with_pick.pddl"
        "domains/minecraft2/prob_irrel_nether_with_pick.pddl"
    ]

    for problem in problems:
        # Path to save entire, non-human-readable profile object
        profile_path = f"time_profiles/minecraft_old"
        # Path to save human-readable profile stats
        profile_path_txt = profile_path + ".txt"
        cProfile.run(
            "scope_pddl(domain, problem, old_var_uniquify = True)", profile_path
        )
        # Get the file object for outputting human-readable profile stats
        stats_readable_file = open(profile_path_txt, "w")
        # Get stats object from saves profile, and set it to stream output to stats_readable_file
        p = pstats.Stats(profile_path, stream=stats_readable_file)
        # Save output to stats_reasable_file
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
        # scope_pddl(domain, problem)

        # Path to save entire, non-human-readable profile object
        profile_path = f"time_profiles/minecraft_new"
        # Path to save human-readable profile stats
        profile_path_txt = profile_path + ".txt"
        cProfile.run(
            "scope_pddl(domain, problem, old_var_uniquify = False)", profile_path
        )
        # Get the file object for outputting human-readable profile stats
        stats_readable_file = open(profile_path_txt, "w")
        # Get stats object from saves profile, and set it to stream output to stats_readable_file
        p = pstats.Stats(profile_path, stream=stats_readable_file)
        # Save output to stats_reasable_file
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
        # scope_pddl(domain, problem)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
