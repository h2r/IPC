import math
import re
import os
from collections import defaultdict, OrderedDict
import glob
import json

# Functions for counting state space as function of the objects in problem
# Predicates
def rbutton_on_count(typecounts):
    return math.pow(2, typecounts["redbutton"])


def gbutton_on_count(typecounts):
    return math.pow(2, typecounts["greenbutton"])


def light_on_count(typecounts):
    return math.pow(2, typecounts["lightswitch"])


def monkey_screaming_count(typecounts):
    return math.pow(2, typecounts["monkey"])


def monkey_watching_bell_count(typecounts):
    return math.pow(2, typecounts["monkey"] * typecounts["bell"])

def connected_buttons_count(typecounts):
    # Buttons are either connected or not - cannot be changed!
    return 1

# Numerics
def xy_count(typecounts):
    xy_types = ["eye", "hand", "marker"]
    xy_types_sum = sum([typecounts[t] for t in xy_types])
    return math.pow(typecounts["x"] * typecounts["y"], xy_types_sum)


def calc_state_space(typecounts):
    count_funcs = [rbutton_on_count, gbutton_on_count, light_on_count, monkey_screaming_count, monkey_watching_bell_count, connected_buttons_count]
    state_space_size = 1
    for fn in count_funcs:
        state_space_size *= fn(typecounts)
    return state_space_size

def get_typecounts(s = None, file_path = None):
    # Get problem string
    if s is None and file_path is None:
        raise ValueError()
    if s is None:
        with open(file_path, "r") as f:
            s = f.read()
    objects_block = re.findall("\(:objects(.*?)\)", s, re.DOTALL)[0]
    objects_block = objects_block.strip()
    print(objects_block)
    typecounts = defaultdict(int)
    objects_block = objects_block.splitlines()
    for l in objects_block:
        obj_nms, type_nm = l.split("-")
        obj_nms = obj_nms.replace("  ", " ").strip()
        n_objs = len(obj_nms.split(" "))
        type_nm = type_nm.strip()
        typecounts[type_nm] += n_objs
    return typecounts

def file2state_space(file_path):
    return calc_state_space(get_typecounts(file_path=file_path))

if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    # file_path = f"{this_dir}/prob01.pddl"
    # typecounts = get_typecounts(file_path=file_path)
    # print(typecounts)
    # state_space = calc_state_space(typecounts)
    # print(state_space)
    state_space_dict = OrderedDict()
    for fp in glob.glob(f"{this_dir}/prob*.pddl"):
        print(fp)
        fn = os.path.basename(fp)
        state_space_dict[fn] = file2state_space(fp)
    print(state_space_dict)
    with open(f"{this_dir}/state_space_sizes.json", "w") as f:
        json.dump(state_space_dict, f, indent=4)