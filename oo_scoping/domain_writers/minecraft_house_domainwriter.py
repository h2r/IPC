from collections import OrderedDict
from itertools import product
import math
import operator as op
from functools import reduce
import copy
from oo_scoping.domains.malmo_writer import make_malmo_domain

item_types = ["diamond-pickaxe"]

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



def underestimate_state_space(x_min, x_max, y_min, y_max, z_min, z_max, n_obsidian_total, item_counts):
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
            obsidian_placements *= (n_locations - i)
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
        lines.append( " ".join(object_names) + " - " + type_name)
    return prefix + "\n\t".join(lines) + suffix


def get_init_location_conds(pos, object_name):
    x,y,z = pos
    init_conds = []
    init_conds.append(f"(= (x {object_name}) {x})")
    init_conds.append(f"(= (y {object_name}) {y})")
    init_conds.append(f"(= (z {object_name}) {z})")
    return init_conds


def get_boundary_positions(x_min, x_max, y_min, y_max, z_min, z_max):
    positions = []
    for x,y in product(range(x_min - 1, x_max + 2), range(y_min - 1, y_max + 2)):
        # Ceiling
        positions.append((x,y,z_max + 1))
        # Floor
        positions.append((x,y,z_min - 1))

    for x, z in product(range(x_min - 1, x_max + 2), range(z_min - 1, z_max + 2)):
        # Front wall
        positions.append((x,y_min - 1,z))
        # Back wall
        positions.append((x, y_max + 1, z))
    for z,y in product(range(z_min - 1, z_max + 2), range(y_min - 1, y_max + 2)):
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
    return prefix + '\n' + body + "\n" + suffix

def get_predicates_str(predicates):
    prefix = "(:predicates"
    suffix = ")"
    lines = ["\t " + f for f in predicates]
    body = "\n".join(lines)
    return prefix + '\n' + body + "\n" + suffix

def get_move_actions():
    # TODO block can't be at same z or higher z
    s = "(:action move-north\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (x ?ag))\n                                                    (= (y ?bl) (+ (y ?ag) 1))\n                                                    (= (z ?bl) (z ?ag))))))\n :effect (and (increase (y ?ag) 1))\n)\n\n(:action move-south\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (x ?ag))\n                                                    (= (y ?bl) (- (y ?ag) 1))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (decrease (y ?ag) 1))\n)\n\n(:action move-east\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (+ (x ?ag) 1))\n                                                    (= (y ?bl) (y ?ag))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (increase (x ?ag) 1))\n)\n\n(:action move-west\n :parameters (?ag - agent)\n :precondition (and (agent-alive ?ag)\n                    (not (exists (?bl - block) (and (= (x ?bl) (- (x ?ag) 1))\n                                                    (= (y ?bl) (y ?ag))\n                                                    (= (z ?bl) (+ (z ?ag) 1))))))\n :effect (and (decrease (x ?ag) 1))\n)"
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
        if(t != "netherportal"):
            actions.append(action_template.format(t=t))
    return actions

def make_drop_actions(item_types, item_or_block=True):
    if(item_or_block):
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
        action_template = """(:action drop-ahead-{t}
 :parameters (?ag - agent ?b - {t})
 :precondition (and (>= (agent-num-{t} ?ag) 1)
                    (not (block-present ?b))
                    (not (exists (?bl - block)
                        (and
                        (= (x ?ag) (x ?bl)) 
                        (= (+ (y ?ag) 1) (y ?bl))
                        (= (z ?ag) (z ?bl))))))
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
        if(t != "netherportal" and t != "diamond-pickaxe"):
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

def get_destructible_block_action(block_type, needed_tool = None):
    # TODO either set x,y,z to far away, or check for block existence in movement actions
    # TODO both hit and destroy are possible when block-hits = 3. Bug?
    if needed_tool is None:
        tool_precond = ""
    else:
        tool_precond = f"\n                        ( >= ( agent-num-{needed_tool} ?ag ) 1 )"
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

def make_domain():
    sections = []
    header = "(define (domain minecraft-house)\n(:requirements :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)"
    footer = ")"
    sections.append(header)
    type_hierarchy = OrderedDict()
    # locatables have position
    # items have agent count and present, in addition to location
    type_hierarchy["object"] = None
    type_hierarchy["locatable"] = "object"
    # type_hierarchy["item"] = "object"
    type_hierarchy["agent"] = "locatable"
    type_hierarchy["item"] = "locatable"
    type_hierarchy["block"] = "locatable"
    type_hierarchy["bedrock"] = "block"
    type_hierarchy["destructible-block"] = "block"
    type_hierarchy["obsidian-block"] = "destructible-block"
    type_hierarchy["oak_wood-block"] = "destructible-block"
    type_hierarchy["oak_wood_stairs-block"] = "destructible-block"
    for i in item_types:
        type_hierarchy[i] = "item"
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
    functions.extend(get_inventory_funcs(inverse_type_hierarchy["destructible-block"]))
    for d in ["x","y","z"]:
        functions.append(f"({d} ?l - locatable)")

    predicates_s = get_predicates_str(predicates)
    sections.append(predicates_s)
    functions_s = get_functions_str(functions)
    sections.append(functions_s)

    actions = []
    actions.append(get_move_actions())
    # actions.extend(make_pickup_actions(inverse_type_hierarchy["item"]))
    # actions.extend(make_drop_actions(inverse_type_hierarchy["item"]))
    actions.extend(make_drop_actions(inverse_type_hierarchy["destructible-block"], False))

    for block_type in inverse_type_hierarchy["destructible-block"]:
        if block_type == "obsidian-block":
            actions.extend(get_destructible_block_action(block_type, needed_tool = "diamond-pickaxe"))
        else:
            actions.extend(get_destructible_block_action(block_type))

    sections.extend(actions)
    sections.append(footer)
    domain_s = "\n\n".join(sections)
    print(domain_s)
    return domain_s

def get_crafting_action(name, inputs, outputs, extra_preconditions = tuple()):
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

def make_instance_1(start_with_pick = True, use_bedrock_boundaries = False, add_irrel_items = False, goal_var = ""):
    object_names = OrderedDict()
    # object_names["obsidian-block"] = ["obsidian0", "obsidian1"]
    object_names["obsidian-block"] = ["obsidian0"]
    object_names["agent"] = ["steve"]
    object_names["diamond-pickaxe"] = ["old-pointy"]
    object_names["oak_wood-block"] = ["oak1", "oak2", "oak3", "oak4", "oak5", "oak6", "oak7",
                                      "oak8", "oak9", "oak10", "oak11", "oak12", "oak13", "oak14",
                                      "oak15", "oak16", "oak17", "oak18", "oak19", "oak20", "oak21",
                                      "oak22", "oak23", "oak24", "oak25", "oak26", "oak27", "oak28",
                                      "oak29", "oak30", "oak31", "oak32", "oak33", "oak34", "oak35",
                                      "oak36", "oak37", "oak38", "oak39", "oak40", "oak41", "oak42",
                                      "oak43", "oak44", "oak45", "oak46", "oak47", "oak48"]
    object_names["oak_wood_stairs-block"] = ["oak_stairs1"]
    
    tgt_obsidian = object_names["obsidian-block"][0]
    agent_name = "steve"
    house_origin_x = 1
    house_origin_y = 1

    header = """(define (problem MINECRAFTHUT)
    (:domain minecraft-house)"""

    # TODO: Figure out what the placement of the blocks should be for the goal (see if we can use an 
    # existnetial!)
    # Make the block on predicates and block placing actions correctly for this domain
    # Also write out the initial state!
    # finish up things and see if we can't build this house!
    if (goal_var == "build_oak_stupa"):
        goal = f"""(:goal (and
                (block-present oak_stairs1 )
                (= (x oak_stairs1) {house_origin_x})
                (= (y oak_stairs1) {house_origin_y})
                (= (z oak_stairs1) 0)

                (= (x oak1) {house_origin_x})
                (= (y oak1) {house_origin_y + 1})
                (= (z oak1) 0)
                (= (x oak2) {house_origin_x - 1})
                (= (y oak2) {house_origin_y + 1})
                (= (z oak2) 0)
                (= (x oak3) {house_origin_x + 1})
                (= (y oak3) {house_origin_y + 1})
                (= (z oak3) 0)
                (= (x oak4) {house_origin_x})
                (= (y oak4) {house_origin_y + 2})
                (= (z oak4) 0)
                (= (x oak5) {house_origin_x - 1})
                (= (y oak5) {house_origin_y + 2})
                (= (z oak5) 0)
                (= (x oak6) {house_origin_x + 1})
                (= (y oak6) {house_origin_y + 2})
                (= (z oak6) 0)
                (= (x oak7) {house_origin_x + 2})
                (= (y oak7) {house_origin_y + 2})
                (= (z oak7) 0)
                (= (x oak8) {house_origin_x - 2})
                (= (y oak8) {house_origin_y + 2})
                (= (z oak8) 0)
                (= (x oak9) {house_origin_x})
                (= (y oak9) {house_origin_y + 3})
                (= (z oak9) 0)
                (= (x oak10) {house_origin_x - 1})
                (= (y oak10) {house_origin_y + 3})
                (= (z oak10) 0)
                (= (x oak11) {house_origin_x + 1})
                (= (y oak11) {house_origin_y + 3})
                (= (z oak11) 0)
                (= (x oak12) {house_origin_x + 2})
                (= (y oak12) {house_origin_y + 3})
                (= (z oak12) 0)
                (= (x oak13) {house_origin_x - 2})
                (= (y oak13) {house_origin_y + 3})
                (= (z oak13) 0)
                (= (x oak14) {house_origin_x})
                (= (y oak14) {house_origin_y + 4})
                (= (z oak14) 0)
                (= (x oak15) {house_origin_x - 1})
                (= (y oak15) {house_origin_y + 4})
                (= (z oak15) 0)
                (= (x oak16) {house_origin_x + 1})
                (= (y oak16) {house_origin_y + 4})
                (= (z oak16) 0)
                (= (x oak17) {house_origin_x + 2})
                (= (y oak17) {house_origin_y + 4})
                (= (z oak17) 0)
                (= (x oak18) {house_origin_x - 2})
                (= (y oak18) {house_origin_y + 4})
                (= (z oak18) 0)
                (= (x oak19) {house_origin_x})
                (= (y oak19) {house_origin_y + 5})
                (= (z oak19) 0)
                (= (x oak20) {house_origin_x - 1})
                (= (y oak20) {house_origin_y + 5})
                (= (z oak20) 0)
                (= (x oak21) {house_origin_x + 1})
                (= (y oak21) {house_origin_y + 5})
                (= (z oak21) 0)
                
            )
        )
        """
    elif goal_var == "build_oak_stupa_simple":
        goal = f"""(:goal (and                
                (= (x oak1) {house_origin_x})
                (= (y oak1) {house_origin_y + 1})
                (= (z oak1) 0)
                (= (x oak2) {house_origin_x - 1})
                (= (y oak2) {house_origin_y + 1})
                (= (z oak2) 0)
                (= (x oak3) {house_origin_x + 1})
                (= (y oak3) {house_origin_y + 1})
                (= (z oak3) 0)
                
                ))"""

    x_min, x_max = 0, 11
    y_min, y_max = 0, 8
    z_min, z_max = 0, 4
    
    # Writing initial conditions!
    init_conds = [f"(agent-alive {agent_name})"]
    agent_start_pos = (0,0,0)
    init_conds.extend(get_init_location_conds(agent_start_pos,agent_name))
    inventory_count = OrderedDict()
    for item_type in item_types:
        inventory_count[item_type] = 0
    if start_with_pick:
        inventory_count["diamond-pickaxe"] = 1
    for item_type, item_count in inventory_count.items():
        if item_type != "netherportal":
            init_conds.append(f"( = ( agent-num-{item_type} {agent_name} ) {item_count} )")
    # init_conds.append(f"( = ( agent-num-diamond-pickaxe {agent_name} ) 1 )")
    block_locations = OrderedDict()
    block_locations["obsidian-block"] = [(11,8,1), (10,8,0)]
    block_locations["oak_wood-block"] = [(1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1),
                                         (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1),
                                         (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1),
                                         (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1),
                                         (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1),
                                         (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1),
                                         ]
    block_locations["oak_wood_stairs-block"] = [(1,1,1)]
    init_conds.extend(get_init_location_conds(block_locations["obsidian-block"][0],tgt_obsidian))
    # init_conds.extend(get_init_location_conds(block_locations["obsidian-block"][1],object_names["obsidian-block"][1]))
    for wood_block_loc in range(len(block_locations["oak_wood-block"])):
        block_name = object_names["oak_wood-block"][wood_block_loc]
        init_conds.extend(get_init_location_conds(block_locations["oak_wood-block"][wood_block_loc], block_name))
        init_conds.append(f"( not ( block-present {block_name} ) )")
    for wood_block_loc in range(len(block_locations["oak_wood_stairs-block"])):
        block_name = object_names["oak_wood_stairs-block"][wood_block_loc]
        init_conds.extend(get_init_location_conds(block_locations["oak_wood_stairs-block"][wood_block_loc], block_name))
        init_conds.append(f"( not ( block-present {block_name} ) )")
    for s in object_names["obsidian-block"]:
        init_conds.append(f"( = ( block-hits {s} ) 0 )")
    for s in object_names["oak_wood-block"]:
        init_conds.append((f"( = ( block-hits {s} ) 0 )"))
    for s in object_names["oak_wood_stairs-block"]:
        init_conds.append((f"( = ( block-hits {s} ) 0 )"))
    init_conds.append("(= (agent-num-obsidian-block steve) 0)")
    init_conds.append("(= (agent-num-oak_wood-block steve) 48)")
    init_conds.append("(= (agent-num-oak_wood_stairs-block steve) 1)")
    diamond_pick_name = object_names["diamond-pickaxe"][0]
    init_conds.extend(get_init_location_conds((0,0,0), diamond_pick_name))
    init_conds.append(f"( not ( present {diamond_pick_name} ) )")

    for s in object_names["obsidian-block"]:
        init_conds.append(f"(block-present {s})")


    if use_bedrock_boundaries:
        boundary_positions = get_boundary_positions(x_min, x_max, y_min, y_max, z_min, z_max)
        object_names["bedrock"] = [f"bed{i}" for i in range(len(boundary_positions))]
        # We don't add these to block_locations because we build the malmo boundaries using a different function
        for i in range(len(boundary_positions)):
            s = object_names["bedrock"][i]
            init_conds.extend(get_init_location_conds(boundary_positions[i], s))
            init_conds.append(f"(block-present {s})")
    init_conds = make_init_conds_str(init_conds)
    object_declaration = get_object_declarations(object_names)
    item_locations = OrderedDict()

    # bedrock_path_str = "-bedrock" if use_bedrock_boundaries else ""
    # tgt_path = f"domains/minecraft/prob-01{bedrock_path_str}.pddl"
    # print(prob_s)
    # with open(tgt_path, "w") as f:
    #     f.write(prob_s)
    # print(set_objects(item_counts))
    # print(set_initial_conditions(item_counts))
    item_counts = []
    for item_type in item_types:
        item_counts.append(len(object_names.get(item_type,[])))
    item_counts_dict = OrderedDict()
    for item_type in item_types:
        item_counts_dict[item_type] = len(object_names.get(item_type,[]))
    state_space_underestimate = underestimate_state_space(x_min, x_max, y_min, y_max, z_min, z_max
        , n_obsidian_total=len(object_names["obsidian-block"]),item_counts=item_counts)
    state_space_conventional = conventional_state_space_count(x_min, x_max, y_min, y_max, z_min, z_max
        , n_obsidian_total=len(object_names["obsidian-block"]),item_counts=item_counts_dict)
    state_space_underestimate_s = f"; State space size if we allow any object to have any z > {state_space_underestimate}"
    state_space_conventional_s = f"; Conventional state space size = {state_space_conventional}"
    prob_parts = [header, object_declaration, init_conds, goal, state_space_underestimate_s,state_space_conventional_s, ")"]
    prob_s = "\n\n\n".join(prob_parts)

    # Make malmo domain

    # malmo_s = make_malmo_domain(block_locations, item_locations, agent_start_pos
    #     ,inventory_count,x_min,x_max,y_min, y_max, z_min, z_max)

    return prob_s, None #malmo_s

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
    with open("domains/minecraft_housebuilding/minecraft-wood_stupa.pddl","w") as f:
        f.write(dom_s)
    prob_s, malmo_s = make_instance_1(start_with_pick=False, add_irrel_items=False, goal_var="build_oak_stupa_simple")
    with open("domains/minecraft_housebuilding/prob_stupa_with_pick.pddl","w") as f:
        f.write(prob_s)
    # with open("domains/malmo/problems/prob_nether_with_pick.xml","w") as f:
    #     f.write(malmo_s)