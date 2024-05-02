from collections import OrderedDict
from itertools import product
import math
import operator as op
from functools import reduce
import copy

item_types = ["wool","diamond", "stick", "diamond-pickaxe", "apple", "potato"
    , "rabbit", "orchid-flower", "daisy-flower", "flint", "coal", "iron-ore", "iron-ingot", "netherportal",
    "flint-and-steel"]

def ncr(n, r):
    # https://stackoverflow.com/a/4941932
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2
def conventional_state_space_count(x_min, x_max, y_min, y_max, z_min, z_max, n_obsidian_total, item_counts):
    plane_size = (x_max - x_min + 1) * (y_max - y_min + 1)
    item_counts = copy.copy(item_counts)
    item_counts["agent"] = 1
    item_counts["obsidian-block"] = n_obsidian_total

    # xy position
    x_y_types = list(set(item_types + ["agent", "obsidian-block"]))
    n_x_y_objects = count_key_values(item_counts, x_y_types)
    x_y_states = math.pow(plane_size,n_x_y_objects)

    # z position
    z_types = ["obsidian-block"]
    n_x_y_objects = count_key_values(item_counts, z_types)
    z_states = math.pow(2, n_x_y_objects)

    # present
    present_types = list(set(item_types + ["obsidian-block"]))
    n_present_objects = count_key_values(item_counts, present_types)
    present_states= math.pow(2, n_present_objects)

    # item-count
    item_count_types = list(set(item_types))
    item_count_states = 1
    for t in item_count_types:
        item_count_states *= item_counts[t] + 1
    # n_item_count_objects = count_key_values(item_counts, item_count_types)
    # item_count_states = math.pow()

    # alive
    alive_types = ["agent"]
    n_alive_objects = count_key_values(item_counts, alive_types)
    alive_states = math.pow(2,n_alive_objects)

    n_states = x_y_states * z_states * present_states * item_count_states * alive_states
    return n_states

def count_key_values(d, keys):
    n = 0
    for t in keys:
        if t not in d.keys():
            print(t,"no count")
        else:
            n += d.get(t)
    return n

if __name__ == "__main__":
    item_types = ["wool","diamond", "stick", "diamond-pickaxe", "apple", "potato"
    , "rabbit", "orchid-flower", "daisy-flower", "flint", "coal", "iron-ore", "iron-ingot", "netherportal",
    "flint-and-steel"]

    x_min, x_max = 0, 8
    y_min, y_max = 0, 11
    z_min, z_max = 0, 1

    object_names = OrderedDict()
    object_names["obsidian-block"] = ["obsidian0", "obsidian1"]
    object_names["agent"] = ["steve"]
    object_names["flint"] = ["flint0", "flint1", "flint2"]
    object_names["iron-ore"] = ["iron-ore0"]
    object_names["coal"] = ["coal0"]
    object_names["iron-ingot"] = ["iron-ingot0"]
    object_names["netherportal"] = ["netherportal0"]
    object_names["flint-and-steel"] = ["flint-and-steel0"]

    item_counts = []
    for item_type in item_types:
        item_counts.append(len(object_names.get(item_type,[])))
    item_counts_dict = OrderedDict()
    for item_type in item_types:
        item_counts_dict[item_type] = len(object_names.get(item_type,[]))

    state_space_conventional = conventional_state_space_count(x_min, x_max, y_min, y_max, z_min, z_max
        , n_obsidian_total=len(object_names["obsidian-block"]),item_counts=item_counts_dict)

    print(state_space_conventional)