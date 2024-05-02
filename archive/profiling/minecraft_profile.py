import sys, pprint
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable
import re, copy
import time
import pathlib

from oo_scoping.pddl_scoper import scope_pddl
from oo_scoping.examples import domains_dir
from oo_scoping.utils import make_dir

# For profiling
import cProfile
import pstats
from pstats import SortKey

profile_dir = pathlib.Path(__file__).parent.absolute() / 'time_profiles'
print(str(profile_dir))

def get_path_sans_extension(p):
    return ".".join(str(p).split(".")[:-1])

def get_profile_path(domain, problem, suffix=""):
    """
    :param domain: Path of the domain file
    :param problem: Path of the problem file
    Returns path of the profile file for this domain and path
    """
    domain, problem = str(domain).split(str(domains_dir))[1], str(problem).split(str(domains_dir))[1]
    output_dir = str(profile_dir) + get_path_sans_extension(domain)
    output_path = output_dir + "/" + get_path_sans_extension(problem) + suffix
    return output_path

if __name__ == '__main__':
    start_time = time.time()
    domain = f"{domains_dir}/minecraft2/minecraft-contrived2.pddl"
    # problems = ["domains/minecraft2/prob_obsidian_with_pick.pddl", "domains/minecraft2/prob_obsidian_without_pick.pddl"]
    problem = f"{domains_dir}/minecraft2/prob_irrel_nether_with_pick.pddl"

    print(f"Domain: {domain}")
    print(f"Problem: {problem}")
    # Path to save entire, non-human-readable profile object
    # profile_path = f'time_profiles/minecraft_old'
    profile_path = get_profile_path(domain, problem, "_nonverbose")
    # Path to save human-readable profile stats
    profile_path_txt = profile_path + ".txt"
    print(f"Human-readable profile output: {profile_path_txt}")
    # Create folder containing profile path, if it does not yet exist
    make_dir(profile_path, is_file=True)
    # Run the profiler and save the full, non-human-readable results
    cProfile.run('scope_pddl(domain, problem, verbose=-1)', profile_path)
    # Get the file object for outputting human-readable profile stats
    stats_readable_file = open(profile_path_txt,"w")
    # Get stats object from saves profile, and set it to stream output to stats_readable_file
    p = pstats.Stats(profile_path, stream = stats_readable_file)
    # Save output to stats_reasable_file
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
    # scope_pddl(domain, problem)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")