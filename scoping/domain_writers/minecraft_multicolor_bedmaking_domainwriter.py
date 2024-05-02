from collections import OrderedDict
from itertools import product
import math
import operator as op
from functools import reduce
import copy
from oo_scoping.domain_writers.malmo_writer import make_malmo_domain
from oo_scoping.examples import domains_dir

item_types = ["diamond", "stick", "diamond-axe", "white-dye", "blue-dye", "red-dye"]
destructible_item_types = ["orchid-flower", "daisy-flower", "red-tulip"]


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
        if t != "netherportal" and t != "destructible-item":
            inventory_count_vars.append(f"(agent-num-{t} ?ag - agent)")
    return inventory_count_vars


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
    s = """(:action move-north\n :parameters (?ag - agent)
            :precondition (and (agent-alive ?ag)
                         (not (exists (?bl - block) (and (block-present ?bl) 
                                                         (= (x ?bl) (x ?ag))
                                                         (= (y ?bl) (+ (y ?ag) 1))
                                                         (= (z ?bl) (z ?ag)))))) 
            :effect (and (increase (y ?ag) 1))
)
                        
(:action move-south 
:parameters (?ag - agent) 
:precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (block-present ?bl)
                                                    (= (x ?bl) (x ?ag))
                                                    (= (y ?bl) (- (y ?ag) 1))
                                                    (= (z ?bl) (z ?ag)))))) 
:effect (and (decrease (y ?ag) 1))
)
            
(:action move-east
    :parameters (?ag - agent)
    :precondition (and (agent-alive ?ag)
                        (not (exists (?bl - block) (and (block-present ?bl)
                                                        (= (x ?bl) (+ (x ?ag) 1))
                                                        (= (y ?bl) (y ?ag))
                                                        (= (z ?bl) (z ?ag)))))) 
    :effect (and (increase (x ?ag) 1))
    )
             
(:action move-west
    :parameters (?ag - agent)
    :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (block-present ?bl)
                                                    (= (x ?bl) (- (x ?ag) 1))
                                                    (= (y ?bl) (y ?ag))
                                                    (= (z ?bl) (z ?ag))))))
    :effect (and (decrease (x ?ag) 1))
)"""
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
        if t != "netherportal" and t != "destructible-item":
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
        if t != "netherportal" and t != "diamond-axe" and t != "destructible-item":
            actions.append(action_template.format(t=t))
    return actions


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
                        (= (z ?b) (z ?ag))
                        (block-present ?b)
                        (< (block-hits ?b) 2){tool_precond})
    :effect (and (increase (block-hits ?b) 1))
    )"""
    destroy_s = f"""(:action destroy-{block_type}
    :parameters (?ag - agent ?b - {block_type})
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (z ?ag))
                        (block-present ?b)
                        (= (block-hits ?b) 2){tool_precond})
    :effect (and (not (block-present ?b))
                 (increase (agent-num-{block_type} ?ag) 1)
                 (assign (block-hits ?b) 0)
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
                        (= (z ?b) (z ?ag))
                        (present ?b)
                        (= (item-hits ?b) 0){tool_precond})
    :effect (and (not (present ?b))
                 (increase (agent-num-{item_type} ?ag) 1)
                 (assign (item-hits ?b) 0)
            )
    )"""
    return [destroy_s]


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


def get_wood_plank_crafting_action():
    wood_plank_action = f"""(:action craft-wooden-planks
 :parameters (?ag - agent ?wb - wooden-block)
 :precondition (and (not (block-present ?wb)) (>= (agent-num-wooden-block ?ag) 1) )
 :effect (and (decrease (agent-num-wooden-block ?ag) 1) (increase (agent-num-wooden-planks ?ag) 4)))"""
    return wood_plank_action


def get_wool_coloring_actions(coloring_dict):
    """Returns strings representing actions that enable wool to be dyed

    input:
        coloring_dict: a dict mapping strings of dye-names to ints representing an enum
                       of that color. e.g. {['white-dye' -> 0,'blue-dye' -> 1, 'red-dye' -> 2]}

    output:
        string representing the coloring actions
    """

    wool_coloring_actions = ""
    for dye_name in coloring_dict.keys():
        dye_color = coloring_dict[dye_name]
        wool_coloring_str = f"""(:action apply-{dye_name}
 :parameters (?ag - agent ?woolb - wool-block)
 :precondition (and (not (block-present ?woolb)) (>= (agent-num-wool-block ?ag) 1) (>= (agent-num-{dye_name} ?ag) 1))
 :effect (and (decrease (agent-num-{dye_name} ?ag) 1) (assign (wool-color ?woolb) {dye_color})))"""
        wool_coloring_actions += wool_coloring_str + "\n\n"

    return wool_coloring_actions


def get_bed_color_crafting_actions(coloring_dict):
    """Returns strings representing actions that enable wool to be dyed

    input:
        coloring_dict: a dict mapping strings of dye-names to ints representing an enum
                       of that color. e.g. {['white-dye' -> 0,'blue-dye' -> 1, 'red-dye' -> 2]}

    output:
        string representing the actions to craft different-colored beds
    """
    bed_crafting_actions = ""
    for dye_name in coloring_dict.keys():
        dye_color = coloring_dict[dye_name]
        bed_color_crafting_str = f"""(:action craft-bed-{dye_name}
 :parameters (?ag - agent ?woolb1 - wool-block ?woolb2 - wool-block ?woolb3 - wool-block ?bd - bed)
 :precondition (and (not (block-present ?woolb1)) (not (block-present ?woolb2)) (not (block-present ?woolb3)) 
                (= (wool-color ?woolb1) {dye_color}) (= (wool-color ?woolb2) {dye_color}) (= (wool-color ?woolb3) {dye_color}) 
                (not (= ?woolb1 ?woolb2)) (not (= ?woolb1 ?woolb3)) (not (= ?woolb2 ?woolb3))
                (not (block-present ?bd)) (>= (agent-num-wool-block ?ag) 3) (>= (agent-num-wooden-planks ?ag) 3))
 :effect (and (decrease (agent-num-wooden-planks ?ag) 3) (decrease (agent-num-wool-block ?ag) 3) (increase (agent-num-bed ?ag) 1) (assign (bed-color ?bd) {dye_color})))"""
        bed_crafting_actions += bed_color_crafting_str + "\n\n"
    return bed_crafting_actions


def make_domain():
    sections = []
    header = "(define (domain minecraft-bedmaking)\n(:requirements :equality :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)"
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
    type_hierarchy["wooden-block"] = "destructible-block"
    type_hierarchy["wooden-planks"] = "destructible-block"
    type_hierarchy["wool-block"] = "destructible-block"
    type_hierarchy["bed"] = "destructible-block"
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
    functions.extend(get_inventory_funcs(inverse_type_hierarchy["item"]))
    functions.extend(get_inventory_funcs(inverse_type_hierarchy["destructible-item"]))
    functions.append("(item-hits ?it - destructible-item)")
    functions.extend(get_inventory_funcs(inverse_type_hierarchy["destructible-block"]))
    functions.append("(block-hits ?b - destructible-block)")

    # Add a color enum for wool blocks and beds 0 = white, 1 = blue, 2 = red
    functions.append("(wool-color ?woolb - wool-block)")
    functions.append("(bed-color ?bd - bed)")

    for d in ["x", "y", "z"]:
        functions.append(f"({d} ?l - locatable)")

    predicates_s = get_predicates_str(predicates)
    sections.append(predicates_s)
    functions_s = get_functions_str(functions)
    sections.append(functions_s)

    # Create actions
    actions = []
    actions.append(get_move_actions())
    actions.extend(make_pickup_actions(inverse_type_hierarchy["item"]))
    actions.extend(make_drop_actions(inverse_type_hierarchy["item"], True))
    actions.extend(
        make_drop_actions(inverse_type_hierarchy["destructible-block"], False)
    )

    # Create wool coloring actions
    coloring_dict = {}
    color_enum = 0
    for item in item_types:
        if "dye" in item:
            coloring_dict[item] = color_enum
            color_enum += 1
    actions.append(get_wool_coloring_actions(coloring_dict))

    # Create bed crafting actions
    actions.append(get_bed_color_crafting_actions(coloring_dict))

    # Create crafting actions
    diamond_pick_inputs = OrderedDict([("stick", 2), ("diamond", 3)])
    diamond_pick_outputs = OrderedDict([("diamond-axe", 1)])
    craft_diamond_pickaxe = get_crafting_action(
        "craft-diamond-axe", diamond_pick_inputs, diamond_pick_outputs
    )
    actions.append(craft_diamond_pickaxe)

    actions.append(get_wood_plank_crafting_action())

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
            get_destructible_block_action(block_type, needed_tool="diamond-axe")
        )

    for item_type in inverse_type_hierarchy["destructible-item"]:
        actions.extend(get_destructible_item_action(item_type))

    sections.extend(actions)
    sections.append(footer)
    domain_s = "\n\n".join(sections)
    print(domain_s)
    return domain_s


def make_instance(
    start_with_pick=True,
    use_bedrock_boundaries=False,
    add_irrel_items=True,
    goal_var="",
):
    agent_name = "steve"

    object_names = OrderedDict()
    object_names["agent"] = ["steve"]
    object_names["diamond-axe"] = ["old-pointy"]
    object_names["orchid-flower"] = ["of0", "of1", "of2"]
    object_names["wooden-block"] = [
        "wb0",
        "wb1",
        "wb2",
        "wb3",
        "wb4",
        "wb5",
        "wb6",
        "wb7",
        "wb8",
        "wb9",
        "wb10",
        "wb11",
        "wb12",
        "wb13",
        "wb14",
        "wb15",
        "wb16",
        "wb17",
        "wb18",
        "wb19",
        "wb20",
        "wb21",
        "wb22",
        "wb23",
        "wb24",
        "wb25",
        "wb26",
        "wb27",
        "wb28",
        "wb29",
        "wb30",
    ]
    object_names["wool-block"] = ["woolb1", "woolb2", "woolb3"]
    object_names["bed"] = ["bed1"]
    if add_irrel_items:
        object_names["diamond"] = [
            "dmd0",
            "dmd1",
            "dmd2",
            "dmd3",
            "dmd4",
        ]
        object_names["stick"] = ["stick0", "stick1", "stick2", "stick3", "stick4"]
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
            "rt9",
            "rt10",
            "rt11",
            "rt12",
            "rt13",
            "rt14",
            "rt15",
            "rt16",
            "rt17",
            "rt18",
            "rt19",
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
            "df9",
            "df10",
            "df11",
        ]

    # Create a list of locations for all the blocks that are used to build the house
    # The first block is the only non-house block that will be needed to craft
    # wooden planks
    block_locations = OrderedDict()
    block_locations["wooden-block"] = [
        (7, 7, 0),  # (2,7,0),
        (6, 7, 0),
        (8, 7, 0),
        (5, 8, 0),
        (9, 8, 0),
        (5, 9, 0),
        (9, 9, 0),
        (6, 10, 0),
        (7, 11, 0),
        (8, 10, 0),
        (6, 7, 1),
        (8, 7, 1),
        (5, 8, 1),
        (9, 8, 1),
        (5, 9, 1),
        (9, 9, 1),
        (6, 10, 1),
        (7, 11, 1),
        (8, 10, 1),
        (6, 7, 2),
        (7, 7, 2),
        (8, 7, 2),
        (6, 8, 2),
        (8, 8, 2),
        (6, 9, 2),
        (8, 9, 2),
        (6, 10, 2),
        (8, 10, 2),
        (6, 10, 2),
        (7, 10, 2),
        (8, 10, 2),
    ]
    # Start of setting up of initial conditions
    init_conds = [f"(agent-alive {agent_name})"]

    goal_wood_block_conds = []
    # Create the inital conditions for all the wooden blocks and
    # leverage these to create the goal conditions for these blocks
    for i, loc in enumerate(block_locations["wooden-block"]):
        s = object_names["wooden-block"][i]
        block_init_loc = get_init_location_conds(loc, s)
        block_present_cond = f"(block-present {s})"
        init_conds.extend(block_init_loc)
        init_conds.append(block_present_cond)
        if i == 0:
            goal_wood_block_conds.append("(not " + block_present_cond + ")")
        else:
            goal_wood_block_conds.append(block_present_cond)

    # from IPython import embed; embed()

    header = """(define (problem MINECRAFTCONTRIVED-3)
    (:domain minecraft-bedmaking)"""

    if goal_var == "dye_wool":
        goal = f"""(:goal (and
                (= (wool-color woolb1) 1)
                (= (wool-color woolb2) 1)
                (= (wool-color woolb3) 1)
                )
            )
        """
    else:
        if goal_var == "get_planks":
            goal_header = (
                f"""(:goal (and (>= (agent-num-wooden-planks {agent_name}) 3)"""
            )

        elif goal_var == "make_bed":
            goal_header = f"""(:goal (and 
                    (= (x bed1) 7)
                    (= (y bed1) 9)
                    (= (z bed1) 0)
                    (= (bed-color bed1) 1)
            """
        else:
            print("Not a valid goal specification!")
            exit(1)

        goal_wood_conds_str = "\n                ".join(goal_wood_block_conds)

        goal_parts = [goal_header, goal_wood_conds_str, ")\n)"]
        goal = "".join(goal_parts)

    # These constants are for state-space size counting purposes
    x_min, x_max = 0, 12
    y_min, y_max = 0, 12
    z_min, z_max = 0, 1

    # Position in which everything worked (7,3,0), (7,2,0) with opt-blind for first 2 tasks and opt-hrmax for bedmaking
    agent_start_pos = (7, 1, 0)
    init_conds.extend(get_init_location_conds(agent_start_pos, agent_name))
    inventory_count = OrderedDict()
    for item_type in item_types:
        inventory_count[item_type] = 0
    inventory_count["wool-block"] = 3
    if start_with_pick:
        inventory_count["diamond-axe"] = 1
    for item_type, item_count in inventory_count.items():
        if item_type != "netherportal":
            init_conds.append(
                f"( = ( agent-num-{item_type} {agent_name} ) {item_count} )"
            )

    for s in object_names["wooden-block"]:
        init_conds.append(f"( = ( block-hits {s} ) 0 )")
    init_conds.append("(= (agent-num-wooden-block steve) 0)")

    init_conds.append("(= (agent-num-wooden-planks steve) 0)")

    i = 0
    for s in object_names["wool-block"]:
        init_conds.append(f"( = ( block-hits {s} ) 0 )")
        init_conds.append(f"(not (block-present {s}))")
        if i == 0:
            init_conds.append(f"( = ( wool-color {s} ) 0 )")
        else:
            init_conds.append(f"( = ( wool-color {s} ) 1 )")
        i += 1

    init_conds.append("(= (agent-num-wool-block steve) 3)")

    for s in object_names["bed"]:
        init_conds.extend(get_init_location_conds((0, 0, 0), s))
        init_conds.append(f"( = ( block-hits {s} ) 0 )")
        init_conds.append(f"( = ( bed-color {s} ) 0 )")
        init_conds.append(f"(not (block-present {s}))")
    init_conds.append("(= (agent-num-bed steve) 0)")

    for s in object_names["orchid-flower"]:
        init_conds.append(f"( = ( item-hits {s} ) 0 )")
    init_conds.append("(= (agent-num-orchid-flower steve) 0)")

    if add_irrel_items:
        for s in object_names["red-tulip"]:
            init_conds.append(f"( = ( item-hits {s} ) 0 )")
        init_conds.append("(= (agent-num-red-tulip steve) 0)")

        for s in object_names["daisy-flower"]:
            init_conds.append(f"( = ( item-hits {s} ) 0 )")
        init_conds.append("(= (agent-num-daisy-flower steve) 0)")

    diamond_pick_name = object_names["diamond-axe"][0]
    init_conds.extend(get_init_location_conds((0, 0, 0), diamond_pick_name))
    init_conds.append(f"( not ( present {diamond_pick_name} ) )")

    item_locations = OrderedDict()

    item_locations["orchid-flower"] = []
    i = 0
    for x in range(4, 7):
        loc = (x, 4, 0)
        item_locations["orchid-flower"].append(loc)
        s = object_names["orchid-flower"][i]
        init_conds.extend(get_init_location_conds(loc, s))
        init_conds.append(f"( present {s} )")
        i += 1
    # item_locations["orchid-flower"] = []
    # i = 0
    # for x in range(4,7):
    #     loc = (x,4,0)
    #     item_locations["orchid-flower"].append(loc)
    #     s = object_names["orchid-flower"][i]
    #     init_conds.extend(get_init_location_conds(loc,s))
    #     if i == 0 or i == 2:
    #         init_conds.append(f"(not( present {s} ))")
    #     else:
    #         init_conds.append(f"( present {s} )")
    #     i += 1

    if add_irrel_items:
        item_locations["stick"] = []
        for i, s in enumerate(object_names["stick"]):
            loc = (0, 2 + i, 0)
            item_locations["stick"].append(loc)
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")

        item_locations["diamond"] = []
        for i, s in enumerate(object_names["diamond"]):
            loc = (1, i + 2, 0)
            item_locations["diamond"].append(loc)
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"(present {s})")

        item_locations["red-tulip"] = []
        i = 0
        for loc in [
            (2, 6, 0),
            (3, 6, 0),
            (4, 6, 0),
            (5, 6, 0),
            (6, 6, 0),
            (7, 6, 0),
            (8, 6, 0),
            (8, 5, 0),
            (8, 4, 0),
            (8, 3, 0),
            (8, 2, 0),
            (7, 2, 0),
            (6, 2, 0),
            (5, 2, 0),
            (4, 2, 0),
            (3, 2, 0),
            (2, 2, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
        ]:
            item_locations["red-tulip"].append(loc)
            s = object_names["red-tulip"][i]
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"(present {s})")
            i += 1
        item_locations["daisy-flower"] = []
        i = 0
        for loc in [
            (3, 5, 0),
            (4, 5, 0),
            (5, 5, 0),
            (6, 5, 0),
            (7, 5, 0),
            (7, 4, 0),
            (7, 3, 0),
            (6, 3, 0),
            (5, 3, 0),
            (4, 3, 0),
            (3, 3, 0),
            (3, 4, 0),
        ]:
            item_locations["daisy-flower"].append(loc)
            s = object_names["daisy-flower"][i]
            init_conds.extend(get_init_location_conds(loc, s))
            init_conds.append(f"( present {s} )")
            i += 1

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

    prob_parts = [header, object_declaration, init_conds, goal, ")"]
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
    dom_s = make_domain()
    # Create domain file
    with open(f"{domains_dir}/minecraft3/minecraft-contrived3.pddl", "w") as f:
        f.write(dom_s)
    # Create problem file and MALMO file for domain including all irrelevant
    # items. This is for dye_wool task
    prob_s, malmo_s = make_instance(start_with_pick=True, goal_var="dye_wool")
    with open(f"{domains_dir}/minecraft3/prob_get_dyed_wool_irrel.pddl", "w") as f:
        f.write(prob_s)
    with open(f"{domains_dir}/malmo/problems/prob_dyed_wool.xml", "w") as f:
        f.write(malmo_s)
    # Create a problem file for dye_wool that excludes all the irrelevant stuff
    prob_s, malmo_s = make_instance(
        start_with_pick=True, goal_var="dye_wool", add_irrel_items=False
    )
    with open(f"{domains_dir}/minecraft3/prob_get_dyed_wool_rel.pddl", "w") as f:
        f.write(prob_s)

    # Create problem file for get_planks task
    prob_s, malmo_s = make_instance(start_with_pick=True, goal_var="get_planks")
    with open(f"{domains_dir}/minecraft3/prob_make_wooden_planks_irrel.pddl", "w") as f:
        f.write(prob_s)
    # Create a problem file for get_planks that excludes irrelevant stuff
    prob_s, malmo_s = make_instance(
        start_with_pick=True, goal_var="get_planks", add_irrel_items=False
    )
    with open(f"{domains_dir}/minecraft3/prob_make_wooden_planks_rel.pddl", "w") as f:
        f.write(prob_s)

    # Create problem file for make_bed task
    prob_s, malmo_s = make_instance(start_with_pick=True, goal_var="make_bed")
    with open(f"{domains_dir}/minecraft3/prob_make_bed_irrel.pddl", "w") as f:
        f.write(prob_s)

    # Create a problem file for the make_bed task excluding all the irrelevant items
    prob_s, malmo_s = make_instance(
        start_with_pick=True, goal_var="make_bed", add_irrel_items=False
    )
    with open(f"{domains_dir}/minecraft3/prob_make_bed_rel.pddl", "w") as f:
        f.write(prob_s)

# TODO:
# Start debugging the PDDL domain
# Sub-tasks
# 1. Dye 1 wool
# 2. Mine the wood block and craft 3 wood planks
# 3. Craft a bed and place it in the house!


# Things to do now:
# IMPORTANT notes:
# PROBLEM! All flowers are relevant because wool-color is an int, and if it's relevant
# then all the things touching it are relevant in turn...

# 1. Try to make it such that blocks cannot be dropped atop other blocks?
# 2. Right now, there are no wooden plank blocks or dye items instantiated in the problem.
# As a result, even though the agent can increase its count of these items, it can't actually
# drop them. Consider remedying this..
# 3. Right now, beds only take up one block of space in the PDDL version
