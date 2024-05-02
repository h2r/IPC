import os, time, argparse, subprocess, json, shutil, glob, re, enum
from typing import Iterable


import pandas as pd
import numpy as np

# Example command:
# python experiments/fd_experiment.py 3 examples/IPC_domains_propositional/driverlog/domain.pddl examples/IPC_domains_propositional/driverlog/prob15.pddl ~/Documents/GitHub/downward/fast-downward.py ./logs --problems_dir randomly_generated_prob_files/driverlog/
"""
TODO:
Get state-visited counts
Use https://gist.github.com/nawatts/e2cdca610463200c12eac2a14efc0bfb to print output
"""
repo_root = os.path.dirname(os.path.dirname(__file__))

# Helper functions
def get_scoped_file_path(unscoped_file):
    return add_path_suffix(unscoped_file, "_scoped")

def add_path_suffix(p, s):
    basename, ext = os.path.splitext(p)
    return basename + s + ext


# Main function
def run_experiment(n_runs, domain, problem, fd_path, log_dir, plan_type: str, force_clear=False, run_id=None):
    log_dir = log_dir + "/fd/" + plan_type + "/" + problem.split('.')[-2]
    start_time_exp = time.time()

    if run_id is None or run_id == -1:
        run_id_start = 0
    else:
        run_id_start = run_id

    if run_id != -1:
        # Clear log dir if force_clear is True
        # if force_clear and os.path.exists(log_dir):
        #     shutil.rmtree(log_dir)
        # Make the log directory. Throws an error if the directory already exists
        os.makedirs(log_dir, exist_ok=True)

    # Save arguments to log_dir
    args_dict = {
        "n_runs":n_runs,
        "domain":domain,
        "problem":problem,
        "fd_path":fd_path,
        "log_dir":log_dir
    }
    with open(f"{log_dir}/args.json", "w") as f:
        json.dump({k: str(v) for k, v in args_dict.items()}, f)

    timings_dict = {
        "translate":[],
        "scoping":[],
        "translate_and_scope":[],
        "plan_unscoped_time":[],
        "plan_scoped_time":[],
        "total_unscoped_time":[],
        "total_scoped_time":[],
        "plan_unscoped_generated_nodes": [],
        "plan_unscoped_node_expansions": [],
        "plan_scoped_generated_nodes": [],
        "plan_scoped_node_expansions": [],
        "encoding_size": [],
        "scoping_exit_code": [],
        "plan_scoped_exit_code": [],
        "plan_unscoped_exit_code": [],
    }

    # This would be more precise if we recorded time for multiple iterations of each portion, then divided. TODO consider doing this.
    for i_run in range(run_id_start, run_id_start + n_runs):
        print(f"Run {i_run}")
        log_dir_this_run = f"{log_dir}/{i_run}"
        timings_path = f"{log_dir_this_run}/times.json"

        if run_id != -1:
            if force_clear and os.path.exists(log_dir_this_run):
                shutil.rmtree(log_dir_this_run)
            os.makedirs(log_dir_this_run, exist_ok=False)

            # Translation
            print("Translating")
            sas_path = f"{log_dir_this_run}/unscoped.sas"

            # new_irrel_prob_file_name = f"{log_dir_this_run}/irrel.pddl"
            # add_irrel_goals_to_prob_file(domain, problem, new_irrel_prob_file_name)
            # problem = new_irrel_prob_file_name

            translate_start = time.time()
            trans_cmd_output = translate(domain, problem, sas_path)
            translate_end = time.time()
            if trans_cmd_output.returncode != 0:
                        raise ValueError(f"Translation failed with returncode {trans_cmd_output.returncode}\nstderr: {trans_cmd_output.stderr}\nstdout: {trans_cmd_output.stdout}")

            translate_time = translate_end - translate_start
            timings_dict["translate"].append(translate_time)
            timings_dict["encoding_size"].append(int(re.search(r"(Translator task size:) \d*", trans_cmd_output.stdout.decode()).group().split(' ')[3]))
            with open(timings_path, "w") as f:
                json.dump(timings_dict, f)
            save_cmd_output(trans_cmd_output, f"{log_dir_this_run}/translate")

            # Scoping
            print("Scoping")
            sas_2_path = f"{log_dir_this_run}/unscoped2.sas"
            sas_scoped_path = get_scoped_file_path(sas_2_path)
            translate_and_scope_start = time.time()
            scope_cmd_output = translate_and_scope(domain, problem, sas_2_path)
            translate_and_scope_end = time.time()
            translate_and_scope_time = translate_and_scope_end - translate_and_scope_start
            scoping_time = translate_and_scope_time - translate_time
            timings_dict["scoping"].append(scoping_time)
            timings_dict["translate_and_scope"].append(translate_and_scope_time)
            timings_dict["scoping_exit_code"].append(scope_cmd_output.returncode)
            with open(timings_path, "w") as f:
                json.dump(timings_dict, f)
            save_cmd_output(scope_cmd_output, f"{log_dir_this_run}/translate_and_scope")
            if scope_cmd_output.returncode != 0:
                raise ValueError(f"Scoping failed with returncode {scope_cmd_output.returncode}\nstderr: {scope_cmd_output.stderr}\nstdout: {scope_cmd_output.stdout}")

            # Planning on scoped
            print("Planning on scoped")
            plan_scoped_start_time = time.time()
            plan_scoped_cmd_output = plan(sas_scoped_path, fd_path, plan_type=plan_type)
            plan_scoped_end_time = time.time()
            plan_scoped_time = plan_scoped_end_time - plan_scoped_start_time
            timings_dict["plan_scoped_time"].append(plan_scoped_time)
            timings_dict["total_scoped_time"].append(translate_time + scoping_time + plan_scoped_time)
            timings_dict["plan_scoped_exit_code"].append(plan_scoped_cmd_output.returncode)
            if plan_scoped_cmd_output.returncode == 0:
                timings_dict["plan_scoped_generated_nodes"].append(int(re.search(r"(Generated) \d*", plan_scoped_cmd_output.stdout.decode()).group(0).split(' ')[1]))
                timings_dict["plan_scoped_node_expansions"].append(int(re.search(r"(Expanded) \d*", plan_scoped_cmd_output.stdout.decode()).group(0).split(' ')[1]))
            with open(timings_path, "w") as f:
                json.dump(timings_dict, f)
            save_cmd_output(plan_scoped_cmd_output, f"{log_dir_this_run}/plan_scoped")

            # Planning on unscoped
            print("Planning on unscoped")
            plan_unscoped_start_time = time.time()
            plan_unscoped_cmd_output = plan(sas_path, fd_path, plan_type=plan_type)
            # plan_unscoped_cmd_output = plan(sas_2_path, fd_path, plan_type=plan_type)
            plan_unscoped_end_time = time.time()
            plan_unscoped_time = plan_unscoped_end_time - plan_unscoped_start_time
            timings_dict["plan_unscoped_time"].append(plan_unscoped_time)
            timings_dict["total_unscoped_time"].append(translate_time + plan_unscoped_time)
            timings_dict["plan_unscoped_exit_code"].append(plan_unscoped_cmd_output.returncode)
            if plan_unscoped_cmd_output.returncode == 0:
                timings_dict["plan_unscoped_generated_nodes"].append(int(re.search(r"(Generated) \d*", plan_unscoped_cmd_output.stdout.decode()).group(0).split(' ')[1]))
                timings_dict["plan_unscoped_node_expansions"].append(int(re.search(r"(Expanded) \d*", plan_unscoped_cmd_output.stdout.decode()).group(0).split(' ')[1]))
            with open(timings_path, "w") as f:
                json.dump(timings_dict, f)
            save_cmd_output(plan_unscoped_cmd_output, f"{log_dir_this_run}/plan_unscoped")
        else:
            print("Loading results")
            with open(timings_path, "r") as f:
                loaded_timings = json.load(f)
                for key in loaded_timings.keys():
                    value = np.nan if loaded_timings[key] == [] else loaded_timings[key][0]
                    timings_dict[key].append(value)

    end_time_exp = time.time()
    experiment_duration = end_time_exp - start_time_exp
    print(f"Finished experiment")
    print(f"Ran {n_runs} trials for a total duration of {experiment_duration}")
    print(timings_dict)
    df_times = pd.DataFrame(data=timings_dict)
    s_times_avg = df_times.mean()
    s_times_avg.name = 'avg'
    s_times_std = df_times.std()
    s_times_std.name = 'std'
    s_times_cv = s_times_std / s_times_avg
    s_times_cv.name = "cv"
    df_time_summary = pd.concat([s_times_avg, s_times_std, s_times_cv], axis=1)
    print(f"Timing summary:")
    print(df_time_summary)
    if n_runs > 1:
        df_time_summary.to_csv(f"{log_dir}/timing_summary.csv", index=True)
    return timings_dict


def run_experiments_on_folder(n_runs, domain, problem_folder, fd_path, log_dir,plan_type: str, force_clear=False):
    total_timings_dict = {}
    num_solved_problems = 0
    problem_files = glob.glob(problem_folder + "/*")

    for problem in problem_files:
        log_dir_this_problem = log_dir + "/" + problem.split(".")[-2]

        try:
            curr_timings_dict = run_experiment(n_runs, domain, problem, fd_path, log_dir_this_problem, plan_type=plan_type, force_clear=force_clear)
        except ValueError:
            # In this case, the randomly-generated problem was impossible to solve.
            # Simply skip and move on.
            print(f"Problem {problem} is impossible to solve.")
            continue

        num_solved_problems += 1
        if len(total_timings_dict) == 0:
            total_timings_dict = curr_timings_dict
        else:
            for key in total_timings_dict.keys():
                total_timings_dict[key] += curr_timings_dict[key]

    # Convert timings dict to dataframe for easy processing (code mostly
    # copied from above method).
    df_times = pd.DataFrame(data=total_timings_dict)
    s_times_avg = df_times.mean()
    s_times_avg.name = 'avg'
    s_times_std = df_times.std()
    s_times_std.name = 'std'
    s_times_cv = s_times_std / s_times_avg
    s_times_cv.name = "cv"
    df_time_summary = pd.concat([s_times_avg, s_times_std, s_times_cv], axis=1)

    print(f"Finished experiments; {num_solved_problems} problems out of {len(problem_files)} were solvable.")
    print(f"Aggregate Timing summary:")
    print(df_time_summary)

    log_dir_this_domain = '/'.join(log_dir_this_problem.split('/')[:-1])
    os.makedirs(log_dir_this_domain, exist_ok=True)
    df_time_summary.to_csv(f"{log_dir_this_domain}/timing_summary.csv", index=True)


def save_cmd_output(cmd_output, save_dir):
    os.makedirs(save_dir, exist_ok=False)
    outpaths = {
        "args":f"{save_dir}/args.txt",
        "stdout":f"{save_dir}/stdout.txt",
        "stderr":f"{save_dir}/stderr.txt",
        "returncode":f"{save_dir}/returncode.txt"
    }

    with open(outpaths["args"], "w") as f:
        # TODO would be better to store as a csv perhaps
        f.write(str(cmd_output.args))

    with open(outpaths["stdout"], "w") as f:
        f.write(cmd_output.stdout.decode('UTF-8'))

    with open(outpaths["stderr"], "w") as f:
        f.write(cmd_output.stderr.decode('UTF-8'))

    with open(outpaths["returncode"], "w") as f:
        f.write(str(cmd_output.returncode))

    return outpaths


def translate(domain, problem, sas_path):
    cmd_pieces = ["python", f"{repo_root}/scoping/downward_translate/translate_and_scope.py", domain, problem, "--sas-file", sas_path]
    cmd_output = subprocess.run(cmd_pieces, capture_output=True, shell=False)
    return cmd_output


def translate_and_scope(domain, problem, unscoped_sas_path):
    cmd_pieces = ["python", f"{repo_root}/scoping/downward_translate/translate_and_scope.py", domain, problem, "--sas-file", unscoped_sas_path, "--scope", "True"]
    cmd_output = subprocess.run(cmd_pieces, capture_output=True, shell=False)
    return cmd_output


SEARCH_CONFIGS = {
    "lmcut":"astar(lmcut())",
    "ms":"astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(greedy=false),merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1))",
    "h2":"astar(hm(m=2, verbosity=normal, transform=no_transform(), cache_estimates=true))"
    # h2 is incredibly slow in FD. Don't use it.
}

def plan(sas_path, fd_path, plan_type: str = "lmcut"):
    search_config = SEARCH_CONFIGS[plan_type]
    # Note: we don't call "python" at the beginning
    # Note: "--" separates the file path arg from the remaining args
    cmd_pieces = [fd_path, sas_path, "--", "--search", search_config]
    cmd_output = subprocess.run(cmd_pieces, capture_output=True, shell=False)
    return cmd_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_runs" ,type=int)
    parser.add_argument("domain", type=str)
    parser.add_argument("problem", type=str)
    parser.add_argument("fd_path", type=str)
    parser.add_argument("log_dir", type=str)
    parser.add_argument("--plan_type", type=str, default="lmcut", choices=list(SEARCH_CONFIGS.keys()), help="Plan techniques to use.")
    parser.add_argument("--force_clear_log_dir", default=False, action='store_true')
    parser.add_argument("--problems_dir", type=str, required=False, default=None)
    parser.add_argument("--run_id", type=int, default=None)

    args = parser.parse_args()

    if args.problems_dir is None:
        run_experiment(args.n_runs, args.domain, args.problem, args.fd_path, args.log_dir, plan_type=args.plan_type, force_clear=args.force_clear_log_dir, run_id=args.run_id)
    else:
        run_experiments_on_folder(args.n_runs, args.domain, args.problems_dir, args.fd_path, args.log_dir, plan_type=args.plan_type, force_clear=args.force_clear_log_dir)
