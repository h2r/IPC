import os, re, json

import pandas as pd

from oo_scoping.paths import (
    repo_dir,
    get_effectively_relevant_fluents_file_path,
    get_scoped_file_path,
    replace_extension,
    add_path_suffix,
)
from oo_scoping.downward_translate.translate_and_scope import main_from_other_script

# This file is a mess. It could be much cleaner - it's conceptually pretty simple.


def get_var_sizes(sas_file=None, sas_str=None):
    # Read file if it is passed instead of string
    if sas_str is None:
        with open(sas_file, "r") as f:
            sas_str = f.read()
    var2size = dict()
    matches = re.findall("begin_variable\n(.*)\n.*\n(.*)", sas_str)
    for (nm, n) in matches:
        var2size[nm] = int(n)
    return var2size


# print("\n".join(cmd_list))

# def filtered_dict_prod(d, keys):
#     """
#     :param d: dict mapping key to number
#     :param keys: keys we are taking product over
#     """
#     return math.product([d[k] for k in keys])
#     x = 1
#     for k in keys:
#         x *= d[k]
#     return x
def prod(iter):
    """
    Python does not have prod() function until 3.8, so we implement one here.
    """
    x = 1
    for i in iter:
        x *= i
    return x


def cmd2kwargs(cmd_s):
    """
    Returns kwargs for translate_and_scope.main_from_other_script() based on the equivalent cl command
    :param cmd_s: Command line string
    """
    parts = cmd_s.split(" ")
    tas_kwargs = {}
    # Assign positional args
    tas_kwargs["domain"] = parts[2]
    tas_kwargs["task"] = parts[3]
    # Assign named kwargs
    for i in range(4, len(parts), 2):
        name = parts[i]
        # Check that this i
        assert name[:2] == "--"
        name = name[2:].replace("-", "_")
        val = parts[i + 1]
        # Convert string to python type if needed
        if val == "True":
            val = True
        elif val == "False":
            val = False
        elif re.match("^[-+]?[0-9]+$", val) is not None:
            val = int(val)
        # Warning! This matches non floats like ..92..0. Should be good enough for now though.
        elif re.match("^[-+]?[0-9\.]+$", val) is not None:
            val = float(val)
        tas_kwargs[name] = val
    return tas_kwargs


def calculate_statespaces():
    cmds_path = f"{repo_dir}/downward_translate/scope_cmds.txt"
    with open(cmds_path, "r") as f:
        cmds = f.read().splitlines()
    for cmd_str in cmds:
        tas_kwargs = cmd2kwargs(cmd_str)
        print(cmd_str)
        # Change output path to a subdirectory for cleanliness
        tas_kwargs["sas_file"] = (
            repo_dir + "/examples/sas_generated/" + tas_kwargs["sas_file"]
        )
        tas_kwargs["write_erfs"] = True
        print(tas_kwargs)

        main_from_other_script(**tas_kwargs)
        # Get effectively relevant fluents, both scoped and unscoped
        erf_path = get_effectively_relevant_fluents_file_path(tas_kwargs["sas_file"])
        erf_path_scoped = get_effectively_relevant_fluents_file_path(
            get_scoped_file_path(tas_kwargs["sas_file"])
        )

        with open(erf_path, "r") as f:
            erf = f.read().splitlines()
        with open(erf_path_scoped, "r") as f:
            erf_scoped = f.read().splitlines()

        # print("erf_scoped:\n" + "\n".join(erf_scoped))
        var2size = get_var_sizes(tas_kwargs["sas_file"])
        unscoped_size = prod([var2size[k] for k in erf])
        scoped_size = prod([var2size[k] for k in erf_scoped])
        full_size = prod(var2size.values())
        print(f"Full size: {full_size}")
        print(f"Unscoped size: {unscoped_size}")
        print(f"Scoped size: {scoped_size}")
        size_path = replace_extension(
            add_path_suffix(tas_kwargs["sas_file"], "_sizes"), "json"
        )
        # Should have written to csv instead of json. Oh well
        sizes_dict = {
            "full": full_size,
            "effectively_relevant_unscoped": unscoped_size,
            "effectively_relevant_scoped": scoped_size,
        }
        with open(size_path, "w") as f:
            json.dump(sizes_dict, f, indent=4)


def merge_statespace_jsons():
    cmds_path = f"{repo_dir}/downward_translate/scope_cmds.txt"
    with open(cmds_path, "r") as f:
        cmds = f.read().splitlines()

    df_sizes = pd.DataFrame()
    df_sizes.index.name = "task"
    for cmd_str in cmds:
        tas_kwargs = cmd2kwargs(cmd_str)
        print(cmd_str)
        # Change output path to a subdirectory for cleanliness
        tas_kwargs["sas_file"] = (
            repo_dir + "/examples/sas_generated/" + tas_kwargs["sas_file"]
        )
        tas_kwargs["write_erfs"] = True
        print(tas_kwargs)
        taskname, _ = os.path.splitext(os.path.split(tas_kwargs["sas_file"])[1])

        # Get effectively relevant fluents, both scoped and unscoped
        size_path = replace_extension(
            add_path_suffix(tas_kwargs["sas_file"], "_sizes"), "json"
        )
        with open(size_path, "r") as f:
            size_dict = json.load(f)
        for k, v in size_dict.items():
            df_sizes.loc[taskname, k] = v

    df_sizes.to_csv(f"{repo_dir}/examples/sas_generated/all_sizes.csv", index=True)
    print(df_sizes)
    # big_size_path = f"{repo_dir}/examples/sas_generated/all_sizes.json"
    # with open(size_path, "w") as f:
    # json.dump(sizes_dict, f, indent=4)


def get_operator_counts():
    cmds_path = f"{repo_dir}/downward_translate/scope_cmds.txt"
    with open(cmds_path, "r") as f:
        cmds = f.read().splitlines()

    df_sizes = pd.read_csv(
        f"{repo_dir}/examples/sas_generated/all_sizes.csv", index_col="task"
    )
    for cmd_str in cmds:
        tas_kwargs = cmd2kwargs(cmd_str)
        print(cmd_str)
        # Change output path to a subdirectory for cleanliness
        tas_kwargs["sas_file"] = (
            repo_dir + "/examples/sas_generated/" + tas_kwargs["sas_file"]
        )
        tas_kwargs["write_erfs"] = True
        print(tas_kwargs)
        taskname, _ = os.path.splitext(os.path.split(tas_kwargs["sas_file"])[1])
        unscoped_operators = get_operator_count(tas_kwargs["sas_file"])
        scoped_operators = get_operator_count(
            get_scoped_file_path(tas_kwargs["sas_file"])
        )
        print(f"Unscoped operators: {unscoped_operators}")
        print(f"Scoped operators: {scoped_operators}")
        df_sizes.loc[taskname, "unscoped_operators"] = unscoped_operators
        df_sizes.loc[taskname, "scoped_operators"] = scoped_operators
        # # Get effectively relevant fluents, both scoped and unscoped
        # size_path = replace_extension(add_path_suffix(tas_kwargs["sas_file"], "_sizes"), "json")
        # with open(size_path, "r") as f:
        #     size_dict = json.load(f)
        # for k, v in size_dict.items():
        #     df_sizes.loc[taskname, k] = v
    df_sizes.to_csv(f"{repo_dir}/examples/sas_generated/all_sizes.csv", index=True)


def get_operator_count(sas_file):
    with open(sas_file, "r") as f:
        sas_str = f.read()
    return int(re.search("([0-9]+)\nbegin_operator", sas_str).groups()[0])


if __name__ == "__main__":
    # merge_statespace_jsons()
    get_operator_counts()
