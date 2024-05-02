from collections import OrderedDict
from itertools import product
import math
import operator as op
from functools import reduce
import copy
from oo_scoping.domain_writers.malmo_writer import make_malmo_domain

item_types = [
    "wool",
    "diamond",
    "stick",
    "diamond-pickaxe",
    "apple",
    "potato",
    "rabbit",
    "flint",
    "coal",
    "iron-ore",
    "iron-ingot",
    "netherportal",
    "flint-and-steel",
]
destructible_item_types = ["orchid-flower", "daisy-flower", "red-tulip"]


def ncr(n, r):
    # https://stackoverflow.com/a/4941932
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom  # or / in Python 2


def conventional_state_space_count(
    x_min, x_max, y_min, y_max, z_min, z_max, n_obsidian_total, item_counts
):
    plane_size = (x_max - x_min + 1) * (y_max - y_min + 1)
    item_counts = copy.copy(item_counts)
    item_counts["agent"] = 1
    item_counts["obsidian-block"] = n_obsidian_total

    # xy position
    x_y_types = list(set(item_types + ["agent", "obsidian-block"]))
    n_x_y_objects = count_key_values(item_counts, x_y_types)
    x_y_states = math.pow(plane_size, n_x_y_objects)

    # z position
    z_types = ["obsidian-block"]
    n_x_y_objects = count_key_values(item_counts, z_types)
    z_states = math.pow(2, n_x_y_objects)

    # present
    present_types = list(set(item_types + ["obsidian-block"]))
    n_present_objects = count_key_values(item_counts, present_types)
    present_states = math.pow(2, n_present_objects)

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
    alive_states = math.pow(2, n_alive_objects)

    n_states = x_y_states * z_states * present_states * item_count_states * alive_states
    return n_states


def count_key_values(d, keys):
    n = 0
    for t in keys:
        if t not in d.keys():
            print(t, "no count")
        else:
            n += d.get(t)
    return n


def underestimate_state_space(
    x_min, x_max, y_min, y_max, z_min, z_max, n_obsidian_total, item_counts
):
    """
    Underestimate state space
    Sources of underestimation:
        When asigning item locations, assume all obsidian blocks are present (easy to correct)
        Assume that all items that are not present are in the agent's inventory (harder to correct)
        Assume the agent is alive
    NOT AN UNDERESTIMATE - ONLY OBSIDIANS CAN MOVE Z, AND ITS BINARY
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
            obsidian_placements *= n_locations - i
        obsidian_states += obsidian_placements * which_obsidian

    # Items
    item_states = 1
    # Iterate over item types
    for this_item_type_total in item_counts:
        if this_item_type_total == 0:
            continue
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


def count_state_space(x_min, x_max, y_min, y_max, z_min, z_max, n_obsidian_total):
    # TODO finish
    # Each destructible block can be in any location, or not-present
    # Each item can be
    # obsidian_states = 0
    n_locations = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
    # Choose how many obsidians are present
    for n_obsidian_present in range(n_obsidian_total):
        # Choose which obsidian are present (unordered)
        which_obsidian = ncr(n_obsidian_total, n_obsidian_present)
        # Choose obsidian locations (ordered)
        obsidian_placements = 1
        for i in range(n_obsidian_present):
            obsidian_placements *= n_locations - i
        # obsidian_states += obsidian_placements

        # Choose how many items are present
    raise NotImplementedError()


# type2name = {
#     "apple":"ap",
#     "potato":"tot",
#     "rabbit":"rab",
#     "diamond-axe":"ax",
#     "orchid-flower":"orc",
#     "daisy-flower":"daisy",
#     "bedrock":"bed"
# }

# def set_objects(item_counts):
#     s_prefix = "(:objects"
#     s_suffix = ")"
#     type_lines = []
#     type_lines.append("steve - agent")
#     for item_type, n in item_counts.items():
#         name_prefix = type2name[item_type]
#         obj_names = [f"{name_prefix}{i}" for i in range(n)]
#         this_line = " ".join(obj_names) + " - " + item_type
#         type_lines.append(this_line)
#     return s_prefix + "\n\t" + "\n\t".join(type_lines) + "\n" + s_suffix


def get_object_declarations(objects):
    prefix = "(:objects\n\t"
    suffix = "\n)"
    lines = []
    for type_name, object_names in objects.items():
        lines.append(" ".join(object_names) + " - " + type_name)
    return prefix + "\n\t".join(lines) + suffix


def get_init_location_conds(pos, object_name):
    x, y, z = pos
    init_conds = []
    init_conds.append(f"(= (x {object_name}) {x})")
    init_conds.append(f"(= (y {object_name}) {y})")
    init_conds.append(f"(= (z {object_name}) {z})")
    return init_conds


def get_boundary_positions(x_min, x_max, y_min, y_max, z_min, z_max):
    positions = []
    for x, y in product(range(x_min - 1, x_max + 2), range(y_min - 1, y_max + 2)):
        # Ceiling
        positions.append((x, y, z_max + 1))
        # Floor
        positions.append((x, y, z_min - 1))

    for x, z in product(range(x_min - 1, x_max + 2), range(z_min - 1, z_max + 2)):
        # Front wall
        positions.append((x, y_min - 1, z))
        # Back wall
        positions.append((x, y_max + 1, z))
    for z, y in product(range(z_min - 1, z_max + 2), range(y_min - 1, y_max + 2)):
        # Left wall
        positions.append((x_min - 1, y, z))
        # Right wall
        positions.append((x_max + 1, y, z))
    return positions


def make_init_conds_str(init_conds):
    s_prefix = "(:init"
    s_suffix = ")"
    return s_prefix + "\n\t" + "\n\t".join(init_conds) + "\n" + s_suffix


def get_inventory_funcs(item_types):
    inventory_count_vars = []
    for t in item_types:
        if t != "netherportal":
            inventory_count_vars.append(f"(agent-num-{t} ?ag - agent)")
    return inventory_count_vars


def get_pickup_actions(item_types):
    pass


def invert_dict(d):
    d_new = OrderedDict()
    for k, v in d.items():
        if v not in d_new.keys():
            d_new[v] = []
        d_new[v].append(k)
    return d_new


def get_functions_str(functions):
    prefix = "(:functions"
    suffix = ")"
    lines = ["\t" + f for f in functions]
    body = "\n".join(lines)
    return prefix + "\n" + body + "\n" + suffix


def get_predicates_str(predicates):
    prefix = "(:predicates"
    suffix = ")"
    lines = ["\t " + f for f in predicates]
    body = "\n".join(lines)
    return prefix + "\n" + body + "\n" + suffix


def get_move_actions():
    # TODO block can't be at same z or higher z
    s = "(:action move-north\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (x ?ag))\n                                                    (= (y ?bl) (+ (y ?ag) 1))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (increase (y ?ag) 1))\n)\n\n(:action move-south\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (x ?ag))\n                                                    (= (y ?bl) (- (y ?ag) 1))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (decrease (y ?ag) 1))\n)\n\n(:action move-east\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (+ (x ?ag) 1))\n                                                    (= (y ?bl) (y ?ag))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (increase (x ?ag) 1))\n)\n\n(:action move-west\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (- (x ?ag) 1))\n                                                    (= (y ?bl) (y ?ag))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (decrease (x ?ag) 1))\n)"
    return s


def make_pickup_actions(item_types):
    action_template = """(:action pickup-{t}
 :parameters (?ag - agent ?i - {t})
 :precondition (and (present ?i)
                    (= (x ?i) (x ?ag))
                    (= (y ?i) (y ?ag))
                    (= (z ?i) (z ?ag)))
 :effect (and (increase (agent-num-{t} ?ag) 1)
              (not (present ?i)))
)
"""
    actions = []
    for t in item_types:
        if t != "netherportal":
            actions.append(action_template.format(t=t))
    return actions


def make_drop_actions(item_types, item_or_block=True):
    if item_or_block:
        action_template = """(:action drop-{t}
 :parameters (?ag - agent ?i - {t})
 :precondition (and (>= (agent-num-{t} ?ag) 1)
                    (not (present ?i)))
 :effect (and (present ?i)
              (assign (x ?i) (x ?ag))
              (assign (y ?i) (+ (y ?ag) 1))
              (assign (z ?i) (z ?ag))
              (decrease (agent-num-{t} ?ag) 1)
         )
)
"""
    else:
        action_template = """(:action drop-{t}
 :parameters (?ag - agent ?b - {t})
 :precondition (and (>= (agent-num-{t} ?ag) 1)
                    (not (block-present ?b)))
 :effect (and (block-present ?b)
              (assign (x ?b) (x ?ag))
              (assign (y ?b) (+ (y ?ag) 1))
              (assign (z ?b) (z ?ag))
              (decrease (agent-num-{t} ?ag) 1)
         )
)
"""

    actions = []
    for t in item_types:
        if t != "netherportal" and t != "diamond-pickaxe":
            actions.append(action_template.format(t=t))
    return actions


def make_netherportal_action():
    # No restriction on obdisian being in portal shape?
    action = """(:action open-netherportal
 :parameters (?ag - agent ?ob - obsidian-block ?np - netherportal)
 :precondition (and (>= (agent-num-flint-and-steel ?ag) 1)
                    (= (y ?ob) (+ (y ?ag) 1))
                    (= (z ?ob) (z ?ag))
                    (= (x ?ob) (x ?ag))
                    (block-present ?ob)
                    (not (present ?np))
                )
 :effect (and (present ?np)
              (assign (x ?np) (x ?ob))
              (assign (y ?np) (y ?ob))
              (assign (z ?np) (z ?ob))
         )
)
"""
    return action


def get_destructible_block_action(block_type, needed_tool=None):
    # TODO either set x,y,z to far away, or check for block existence in movement actions
    if needed_tool is None:
        tool_precond = ""
    else:
        tool_precond = (
            f"\n                        ( >= ( agent-num-{needed_tool} ?ag ) 1 )"
        )
    hit_s = f"""(:action hit-{block_type}
    :parameters (?ag - agent ?b - {block_type})
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (block-present ?b)
                        (< (block-hits ?b) 4){tool_precond})
    :effect (and (increase (block-hits ?b) 1))
    )"""
    destroy_s = f"""(:action destroy-{block_type}
    :parameters (?ag - agent ?b - {block_type})
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (block-present ?b)
                        (= (block-hits ?b) 3){tool_precond})
    :effect (and (not (block-present ?b))
                 (increase (agent-num-{block_type} ?ag) 1)
            )
    )"""
    return [hit_s, destroy_s]


def get_destructible_item_action(item_type, needed_tool=None):
    # TODO either set x,y,z to far away, or check for block existence in movement actions
    if needed_tool is None:
        tool_precond = ""
    else:
        tool_precond = (
            f"\n                        ( >= ( agent-num-{needed_tool} ?ag ) 1 )"
        )

    destroy_s = f"""(:action destroy-{item_type}
    :parameters (?ag - agent ?b - {item_type})
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (present ?b)
                        (= (item-hits ?b) 0){tool_precond})
    :effect (and (not (present ?b))
                 (increase (agent-num-{item_type} ?ag) 1)
            )
    )"""
    return [destroy_s]


def make_domain():
    sections = []
    header = "(define (domain minecraft-contrived)\n(:requirements :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)"
    footer = ")"
    sections.append(header)
    type_hierarchy = OrderedDict()
    # locatables have position
    # items have agent count and present, in addition to location
    type_hierarchy["object"] = None
    type_hierarchy["locatable"] = "object"
    type_hierarchy["agent"] = "locatable"
    type_hierarchy["item"] = "locatable"
    type_hierarchy["block"] = "locatable"
    type_hierarchy["bedrock"] = "block"
    type_hierarchy["destructible-block"] = "block"
    type_hierarchy["obsidian-block"] = "destructible-block"
    type_hierarchy["destructible-item"] = "item"

    for i in item_types:
        type_hierarchy[i] = "item"

    for i in destructible_item_types:
        type_hierarchy[i] = "destructible-item"

    inverse_type_hierarchy = invert_dict(type_hierarchy)
    types_s = make_types_declaration(type_hierarchy)
    sections.append(types_s)

    predicates = []
    predicates.append("(present ?i - item)")
    predicates.append("(block-present ?b - block)")
    predicates.append("(agent-alive ?ag - agent)")
    functions = []
    functions.append("(block-hits ?b - destructible-block)")
    functions.extend(get_inventory_funcs(inverse_type_hierarchy["item"]))
    functions.extend(get_inventory_funcs(["obsidian-block"]))
    for d in ["x", "y", "z"]:
        functions.append(f"({d} ?l - locatable)")

    predicates_s = get_predicates_str(predicates)
    sections.append(predicates_s)
    functions_s = get_functions_str(functions)
    sections.append(functions_s)

    actions = []
    actions.append(get_move_actions())
    actions.extend(make_pickup_actions(inverse_type_hierarchy["item"]))
    actions.extend(make_drop_actions(inverse_type_hierarchy["item"], True))
    actions.extend(
        make_drop_actions(inverse_type_hierarchy["destructible-block"], False)
    )

    diamond_pick_inputs = OrderedDict([("stick", 2), ("diamond", 3)])
    diamond_pick_outputs = OrderedDict([("diamond-pickaxe", 1)])
    craft_diamond_pickaxe = get_crafting_action(
        "craft-diamond-pickaxe", diamond_pick_inputs, diamond_pick_outputs
    )
    actions.append(craft_diamond_pickaxe)

    iron_ingot_inputs = OrderedDict([("iron-ore", 1), ("coal", 1)])
    iron_ingot_outputs = OrderedDict([("iron-ingot", 1)])
    craft_iron_ingot = get_crafting_action(
        "craft-iron-ingot", iron_ingot_inputs, iron_ingot_outputs
    )
    actions.append(craft_iron_ingot)

    flint_and_steel_inputs = OrderedDict([("iron-ingot", 1), ("flint", 1)])
    flint_and_steel_outputs = OrderedDict([("flint-and-steel", 1)])
    craft_flint_and_steel = get_crafting_action(
        "craft-flint-and-steel", flint_and_steel_inputs, flint_and_steel_outputs
    )
    actions.append(craft_flint_and_steel)

    red_dye_inputs = OrderedDict([("red-tulip", 1)])
    red_dye_outputs = OrderedDict([("red-dye", 1)])
    craft_red_dye = get_crafting_action(
        "craft-red-dye", red_dye_inputs, red_dye_outputs
    )
    actions.append(craft_red_dye)

    blue_dye_inputs = OrderedDict([("orchid-flower", 1)])
    blue_dye_outputs = OrderedDict([("blue-dye", 1)])
    craft_blue_dye = get_crafting_action(
        "craft-blue-dye", blue_dye_inputs, blue_dye_outputs
    )
    actions.append(craft_blue_dye)

    white_dye_inputs = OrderedDict([("daisy-flower", 1)])
    white_dye_outputs = OrderedDict([("white-dye", 1)])
    craft_white_dye = get_crafting_action(
        "craft-white-dye", white_dye_inputs, white_dye_outputs
    )
    actions.append(craft_white_dye)

    for block_type in inverse_type_hierarchy["destructible-block"]:
        actions.extend(
            get_destructible_block_action(block_type, needed_tool="diamond-pickaxe")
        )

    for item_type in inverse_type_hierarchy["destructible-item"]:
        actions.extend(
            get_destructible_item_action(block_type, needed_tool="diamond-pickaxe")
        )

    actions.append(make_netherportal_action())

    sections.extend(actions)
    sections.append(footer)
    domain_s = "\n\n".join(sections)
    print(domain_s)
    return domain_s


def get_crafting_action(name, inputs, outputs, extra_preconditions=tuple()):
    """
    input: Dict[item_type] -> item_count
    output: Dict[item_type] -> item_count
    """
    prefix = f"""(:action {name}
    :parameters ( ?ag - agent )"""
    suffix = "\n)"

    precondition_prefix = "    :precondition ( and\n                      "
    precondition_suffix = "\n                  )"

    preconds = []
    for item_type, item_count in inputs.items():
        preconds.append(f"( >= (agent-num-{item_type} ?ag) {item_count} )")

    preconds.extend(extra_preconditions)
    precond_body = "\n                      ".join(preconds)
    precond_s = precondition_prefix + precond_body + precondition_suffix

    effects_prefix = "    :effect (and "
    effects_suffix = ")"
    effects = []
    for item_type, item_count in outputs.items():
        effects.append(f"(increase (agent-num-{item_type} ?ag) {item_count})")
    for item_type, item_count in inputs.items():
        effects.append(f"(decrease (agent-num-{item_type} ?ag) {item_count})")
    effects_body = "\n        ".join(effects)
    effects_s = effects_prefix + effects_body + effects_suffix

    return "\n".join([prefix, precond_s, effects_s, suffix])


def make_instance(
    start_with_pick=True,
    use_bedrock_boundaries=False,
    add_irrel_items=False,
    goal_var="",
):
    object_names = OrderedDict()
    # object_names["obsidian-block"] = ["obsidian0", "obsidian1", "obsidian2", "obsidian3", "obsidian4", "obsidian5", "obsidian6", "obsidian7", "obsidian8", "obsidian9"]
    object_names["obsidian-block"] = ["obsidian0", "obsidian1"]
    object_names["agent"] = ["steve"]
    object_names["diamond-pickaxe"] = ["old-pointy"]
    object_names["diamond"] = ["dmd0", "dmd1", "dmd2"]
    object_names["stick"] = ["stick0", "stick1"]
    object_names["flint"] = ["flint0", "flint1", "flint2"]
    object_names["iron-ore"] = ["iron-ore0"]
    object_names["coal"] = ["coal0"]
    object_names["iron-ingot"] = ["iron-ingot0"]
    object_names["netherportal"] = ["netherportal0"]
    object_names["flint-and-steel"] = ["flint-and-steel0"]
    object_names["red-tulip"] = [
        "rt0",
        "rt1",
        "rt2",
        "rt3",
        "rt4",
        "rt5",
        "rt6",
        "rt7",
        "rt8",
    ]
    object_names["daisy-flower"] = [
        "df0",
        "df1",
        "df2",
        "df3",
        "df4",
        "df5",
        "df6",
        "df7",
        "df8",
    ]
    object_names["orchid-flower"] = [
        "of0",
        "of1",
        "of2",
        "of3",
        "of4",
        "of5",
        "of6",
        "of7",
        "of8",
    ]

    if add_irrel_items:
        object_names["apple"] = ["apple1", "apple2", "apple3"]
        object_names["potato"] = ["potato1", "potato2", "potato3", "potato4", "potato5"]
        object_names["rabbit"] = ["rabbit1", "rabbit2", "rabbit3", "rabbit4", "rabbit5"]

    tgt_obsidian = object_names["obsidian-block"][0]
    tgt_netherpotal = object_names["netherportal"][0]
    agent_name = "steve"

    header = """(define (problem MINECRAFTCONTRIVED-1)
    (:domain minecraft-contrived)"""

    if goal_var == "break_obsidian":
        goal = f"""(:goal (and
                (not (block-present {tgt_obsidian} ))
                (= (x {agent_name}) 3)
                (= (y {agent_name}) 4)
                (= (z {agent_name}) 0)
            )
        )
        """
    elif goal_var == "make_netherportal":
        goal = f"""(:goal (and 
                            (present {tgt_netherpotal})
                            (= (x {tgt_netherpotal}) 11)
                            (= (y {tgt_netherpotal}) 7)
                            (= (z {tgt_netherpotal}) 0)
                          )
        )
        """
    elif goal_var == "make_flint_and_steel":
        goal = f"""(:goal (and
                ( = ( agent-num-flint-and-steel {agent_name} ) 1 )
                (= (x {agent_name}) 11)
                (= (y {agent_name}) 2)
                (= (z {agent_name}) 0)
                )
            )"""

    x_min, x_max = 0, 12
    y_min, y_max = 0, 12
    z_min, z_max = 0, 1

    # item_counts = OrderedDict([("apple",2),("potato",1)])
    # item_counts = OrderedDict([("apple",1)])

    # Setting up initial conditions block
    init_conds = [f"(agent-alive {agent_name})"]
    agent_start_pos = (0, 0, 0)
    init_conds.extend(get_init_location_conds(agent_start_pos, agent_name))
    inventory_count = OrderedDict()
    for item_type in item_types:
        inventory_count[item_type] = 0
    if start_with_pick:
        inventory_count["diamond-pickaxe"] = 1
    for item_type, item_count in inventory_count.items():
        if item_type != "netherportal":
            init_conds.append(
                f"( = ( agent-num-{item_type} {agent_name} ) {item_count} )"
            )
    # init_conds.append(f"( = ( agent-num-diamond-pickaxe {agent_name} ) 1 )")
    block_locations = OrderedDict()
    block_locations["obsidian-block"] = [(11, 8, 1), (10, 8, 0)]
    init_conds.extend(
        get_init_location_conds(block_locations["obsidian-block"][0], tgt_obsidian)
    )  # We got current excel results on (11, 7, 1)
    init_conds.extend(
        get_init_location_conds(
            block_locations["obsidian-block"][1], object_names["obsidian-block"][1]
        )
    )
    # init_conds.extend(get_init_location_conds((12,7,0),object_names["obsidian-block"][2]))
    # init_conds.extend(get_init_location_conds((10,7,1),object_names["obsidian-block"][3]))
    # init_conds.extend(get_init_location_conds((12,7,1),object_names["obsidian-block"][4]))
    # init_conds.extend(get_init_location_conds((10,7,2),object_names["obsidian-block"][5]))
    # init_conds.extend(get_init_location_conds((12,7,2),object_names["obsidian-block"][6]))
    # init_conds.extend(get_init_location_conds((10,7,3),object_names["obsidian-block"][7]))
    # init_conds.extend(get_init_location_conds((12,7,3),object_names["obsidian-block"][8]))
    # init_conds.extend(get_init_location_conds((11,7,3),object_names["obsidian-block"][9]))

    for s in object_names["obsidian-block"]:
        init_conds.append(f"( = ( block-hits {s} ) 0 )")
    init_conds.append("(= (agent-num-obsidian-block steve) 0)")

    for s in object_names["red-tulip"]:
        init_conds.append(f"( = ( item-hits {s} ) 0 )")
    init_conds.append("(= (agent-num-red-tulip steve) 0)")

    for s in object_names["orchid-flower"]:
        init_conds.append(f"( = ( item-hits {s} ) 0 )")
    init_conds.append("(= (agent-num-orchid-flower steve) 0)")

    for s in object_names["daisy-flower"]:
        init_conds.append(f"( = ( item-hits {s} ) 0 )")
    init_conds.append("(= (agent-num-daisy-flower steve) 0)")

    diamond_pick_name = object_names["diamond-pickaxe"][0]
    init_conds.extend(get_init_location_conds((0, 0, 0), diamond_pick_name))
    init_conds.append(f"( not ( present {diamond_pick_name} ) )")

    item_locations = OrderedDict()
    item_locations["stick"] = []
    for i, s in enumerate(object_names["stick"]):
        loc = (1, i, 0)
        item_locations["stick"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"( present {s} )")

    item_locations["flint"] = []
    for i, s in enumerate(object_names["flint"]):
        loc = (8, i, 0)
        item_locations["flint"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"( present {s} )")

    block_locations["iron-ore"] = []
    for i, s in enumerate(object_names["iron-ore"]):
        loc = (10, i, 0)
        block_locations["iron-ore"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"( present {s} )")

    item_locations["coal"] = []
    for i, s in enumerate(object_names["coal"]):
        loc = (9, i, 0)
        item_locations["coal"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"( present {s} )")

    item_locations["diamond"] = []
    for i, s in enumerate(object_names["diamond"]):
        loc = (2, i, 0)
        item_locations["diamond"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"(present {s})")

    item_locations["iron-ingot"] = []
    for i, s in enumerate(object_names["iron-ingot"]):
        loc = (0, i, 0)
        item_locations["iron-ingot"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"(not ( present {s} ))")

    item_locations["flint-and-steel"] = []
    for i, s in enumerate(object_names["flint-and-steel"]):
        loc = (0, i, 0)
        item_locations["flint-and-steel"].append(loc)
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"(not ( present {s} ))")

    item_locations["netherportal"] = []
    for i, s in enumerate(object_names["netherportal"]):
        item_locations["flint-and-steel"].append((0, i, 0))
        init_conds.extend(get_init_location_conds((0, i, 0), s))
        init_conds.append(f"(not ( present {s} ))")

    item_locations["red-tulip"] = []
    i = 0
    for x in range(6, 9):
        for y in range(3, 6):
            loc = (x, y, 0)
            item_locations["red-tulip"].append(loc)
            s = object_names["red-tulip"][i]
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")
            i += 1
    item_locations["daisy-flower"] = []
    i = 0
    for x in range(1, 4):
        for y in range(3, 6):
            loc = (x, y, 0)
            item_locations["daisy-flower"].append(loc)
            s = object_names["daisy-flower"][i]
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")
            i += 1
    item_locations["orchid-flower"] = []
    i = 0
    for x in range(4, 7):
        for y in range(6, 9):
            loc = (x, y, 0)
            item_locations["orchid-flower"].append(loc)
            s = object_names["orchid-flower"][i]
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")
            i += 1

    if add_irrel_items:
        item_locations["apple"] = []
        for i, s in enumerate(object_names["apple"]):
            loc = (3, i, 0)
            item_locations["apple"].append(loc)
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")

        item_locations["potato"] = []
        for i, s in enumerate(object_names["potato"]):
            loc = (4, i, 0)
            item_locations["potato"].append(loc)
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")

        item_locations["rabbit"] = []
        for i, s in enumerate(object_names["rabbit"]):
            loc = (7, i, 0)
            item_locations["rabbit"].append(loc)
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")

    for s in object_names["obsidian-block"]:
        init_conds.append(f"(block-present {s})")

    # End initial conditions

    if use_bedrock_boundaries:
        boundary_positions = get_boundary_positions(
            x_min, x_max, y_min, y_max, z_min, z_max
        )
        object_names["bedrock"] = [f"bed{i}" for i in range(len(boundary_positions))]
        # We don't add these to block_locations because we build the malmo boundaries using a different function
        for i in range(len(boundary_positions)):
            s = object_names["bedrock"][i]
            init_conds.extend(get_init_location_conds(boundary_positions[i], s))
            init_conds.append(f"(block-present {s})")

    init_conds = make_init_conds_str(init_conds)
    object_declaration = get_object_declarations(object_names)

    # bedrock_path_str = "-bedrock" if use_bedrock_boundaries else ""
    # tgt_path = f"domains/minecraft/prob-01{bedrock_path_str}.pddl"
    # print(prob_s)
    # with open(tgt_path, "w") as f:
    #     f.write(prob_s)
    # print(set_objects(item_counts))
    # print(set_initial_conditions(item_counts))
    item_counts = []
    for item_type in item_types:
        item_counts.append(len(object_names.get(item_type, [])))
    item_counts_dict = OrderedDict()
    for item_type in item_types:
        item_counts_dict[item_type] = len(object_names.get(item_type, []))
    state_space_underestimate = underestimate_state_space(
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        n_obsidian_total=len(object_names["obsidian-block"]),
        item_counts=item_counts,
    )
    state_space_conventional = conventional_state_space_count(
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        n_obsidian_total=len(object_names["obsidian-block"]),
        item_counts=item_counts_dict,
    )
    state_space_underestimate_s = f"; State space size if we allow any object to have any z > {state_space_underestimate}"
    state_space_conventional_s = (
        f"; Conventional state space size = {state_space_conventional}"
    )
    prob_parts = [
        header,
        object_declaration,
        init_conds,
        goal,
        state_space_underestimate_s,
        state_space_conventional_s,
        ")",
    ]
    prob_s = "\n\n\n".join(prob_parts)

    # Make malmo domain
    malmo_s = make_malmo_domain(
        block_locations,
        item_locations,
        agent_start_pos,
        inventory_count,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )

    return prob_s, malmo_s


def make_types_declaration(type_hierarchy):
    inverse_type_hierarchy = invert_dict(type_hierarchy)
    lines = []
    for parent, children in inverse_type_hierarchy.items():
        if parent is None:
            pass
            # from IPython import embed; embed()
            # lines.extend(children)
        else:
            l = " ".join(children) + " - " + parent
            lines.append(l)
    types_prefix = "(:types "
    types_suffix = ")"
    types_s = types_prefix + "\n\t" + "\n\t".join(lines) + "\n" + types_suffix
    return types_s


if __name__ == "__main__":
    # type_hierarchy = OrderedDict()
    # # locatables have position
    # # items have agent count and present, in addition to location
    # type_hierarchy["locatable"] = None
    # type_hierarchy["agent"] = "locatable"
    # type_hierarchy["item"] = "locatable"
    # item_types = ["apple", "potato", "rabbit", "diamond-axe", "orchid-flower", "daisy-flower"]
    # for i in item_types:
    #     type_hierarchy[i] = "item"
    # print(make_types_declaration(type_hierarchy))
    dom_s = make_domain()
    with open("domains/minecraft2/minecraft-contrived2.pddl", "w") as f:
        f.write(dom_s)
    prob_s, malmo_s = make_instance(
        start_with_pick=True, add_irrel_items=False, goal_var="make_netherportal"
    )
    with open("examples/minecraft2/prob_nether_with_pick.pddl", "w") as f:
        f.write(prob_s)
    with open("domains/malmo/problems/prob_nether_with_pick.xml", "w") as f:
        f.write(malmo_s)

    # prob_ir, malmo_ir = make_instance(start_with_pick=True, add_irrel_items=True, goal_var="make_netherportal")
    # with open("examples/minecraft2/prob_irrel_nether_with_pick.pddl","w") as f:
    #     f.write(prob_ir)
    # with open("domains/malmo/problems/prob_irrel_nether_with_pick.xml","w") as f:
    #     f.write(malmo_ir)

    # prob_s, malmo_s = make_instance(start_with_pick=True, add_irrel_items=False, goal_var="break_obsidian")
    # with open("examples/minecraft2/prob_obsidian_with_pick.pddl","w") as f:
    #     f.write(prob_s)
    # with open("domains/malmo/problems/prob_obsidian_with_pick.xml","w") as f:
    #     f.write(malmo_s)

    # prob_ir, malmo_ir = make_instance(start_with_pick=True, add_irrel_items=True, goal_var="break_obsidian")
    # with open("examples/minecraft2/prob_irrel_obsidian_with_pick.pddl","w") as f:
    #     f.write(prob_ir)
    # with open("domains/malmo/problems/prob_irrel_obsidian_with_pick.xml","w") as f:
    #     f.write(malmo_ir)

    # prob_s, malmo_s = make_instance(start_with_pick=True, add_irrel_items=False, goal_var="make_flint_and_steel")
    # with open("examples/minecraft2/prob_flint_with_pick.pddl","w") as f:
    #     f.write(prob_s)
    # with open("domains/malmo/problems/prob_flint_with_pick.xml","w") as f:
    #     f.write(malmo_s)

    # prob_ir, malmo_ir = make_instance(start_with_pick=True, add_irrel_items=True, goal_var="make_flint_and_steel")
    # with open("examples/minecraft2/prob_irrel_flint_with_pick.pddl","w") as f:
    #     f.write(prob_ir)
    # with open("domains/malmo/problems/prob_irrel_flint_with_pick.xml","w") as f:
    #     f.write(malmo_ir)
