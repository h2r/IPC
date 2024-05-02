from functools import reduce
import operator as op


def ncr(n, r):
    # https://stackoverflow.com/a/4941932
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def underestimate_state_space(x_min, x_max, y_min, y_max, z_min, z_max, n_obsidian_total, item_counts):
    """
    Underestimate state space
    Sources of underestimation:
        When asigning item locations, assume all obsidian blocks are present (easy to correct)
        Assume that all items that are not present are in the agent's inventory (harder to correct)
        Assume the agent is alive
    """
    n_locations = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
    obsidian_states = 0
    # Choose how many obsidians are present
    for n_obsidian_present in range(n_obsidian_total):
        # Choose which obsidian are present (unordered)
        which_obsidian = ncr(n_obsidian_total, n_obsidian_present)
        # Choose obsidian locations (ordered)
        obsidian_placements = 1
        for i in range(n_obsidian_present):
            obsidian_placements *= (n_locations - i)
        obsidian_states += obsidian_placements * which_obsidian

    # Items
    item_states = 1
    # Iterate over item types
    for this_item_type_total in  item_counts:
        if this_item_type_total == 0: continue
        # Choose a lower-bound for the locations items can be in by assuming all obsidian is present
        n_available_locations = n_locations - n_obsidian_total
        states_this_item_type = 0
        # Choose how many of these items are present
        for n_item_present in range(this_item_type_total):
            # Choose which items are present
            which_items = ncr(this_item_type_total, n_item_present)
            item_placements = 1
            # Choose item locations. Items can be in any location not occupied by a block
            for i in range(n_item_present):
                # items can be in the same location
                item_placements *= n_available_locations
            states_this_item_type += which_items * item_placements
        item_states *= states_this_item_type

    agent_states = n_locations
    total_states = obsidian_states * item_states * agent_states
    return total_states

if __name__ == "__main__":
    # Obsidian Task Scoped
    print(underestimate_state_space(0, 11, 0, 8, 0, 0, 2, []))
    # Flint Task Scoped
    print(underestimate_state_space(0, 11, 0, 8, 0, 0, 2, [3,1,1,1,1]))
    # Nether Task Scoped
    print(underestimate_state_space(0, 11, 0, 8, 0, 0, 2, [3,1,1,1,1,1]))
