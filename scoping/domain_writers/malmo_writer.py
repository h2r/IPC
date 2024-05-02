from collections import OrderedDict
from itertools import product
import math
import operator as op
from functools import reduce
import copy
from scoping.examples import domains_dir

type_replacements = {
    "obsidian-block": "obsidian",
    "wool-block": "wool",
    "wooden-block": "log",
    "netherportal": "portal",
    "diamond-pickaxe": "diamond_pickaxe",
    "diamond-axe": "diamond_axe",
    "flint-and-steel": "flint_and_steel",
    "iron-ingot": "iron_ingot",
    "iron-ore": "iron_ore",
    "daisy-flower": "oxeye_daisy",
    "orchid-flower": "blue_orchid",
    "red-tulip": "red_tulip",
    "oak-sapling": "oak",
    "birch-sapling": "birch",
}

with open(f"{domains_dir}/malmo/block_types.txt", "r") as f:
    valid_block_types = f.read().splitlines()
with open(f"{domains_dir}/malmo/item_types.txt", "r") as f:
    valid_item_types = f.read().splitlines()


def pddl2malmo_coords(x, y, z, z_offset=209):
    # TODO test
    # TODO: IMPORTANT: the ground is at MALMO y=209. this needs to be accounted for
    return x, z + z_offset, -y


def draw_item(object_type, x, y, z):
    if object_type not in valid_item_types:
        raise ValueError(f"{object_type} not a valid item type")
    return f'<DrawItem x="{x}" y="{y}" z="{z}" type="{object_type}"/>'
    # return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'


def draw_block(object_type, x, y, z):
    if object_type not in valid_block_types:
        raise ValueError(f"{object_type} not a valid block type")
    return f'<DrawCuboid x1="{x}" y1="{y}" z1="{z}" x2="{x}" y2="{y}" z2="{z}" type="{object_type}"/>'
    # return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'


def draw_flower(object_type, x, y, z):
    return f'<DrawCuboid x1="{x}" y1="{y}" z1="{z}" x2="{x}" y2="{y}" z2="{z}" type="red_flower" variant="{object_type}"/>'


def draw_sapling(object_type, x, y, z):
    return f'<DrawCuboid x1="{x}" y1="{y}" z1="{z}" x2="{x}" y2="{y}" z2="{z}" type="sapling" variant="{object_type}"/>'


def add_decimal(x: int):
    return str(x) + ".0"


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def make_agent_placement(x, y, z, pitch=10, yaw=180):
    # Add .5 to center agent on block, so that discrete movement works
    x, z = x + sign(x) * 0.5, z + sign(z) * 0.5
    return f'<Placement x="{x}" y="{y}" z="{z}" pitch="{pitch}" yaw="{yaw}"/>'


def make_inventory(inventory_counts):
    inventory_lines = []
    slot_id = 0
    for item_type, item_count in inventory_counts.items():
        if item_count > 0:
            inventory_lines.append(
                f'<InventoryItem type="{item_type}" slot="{slot_id}" quantity="{item_count}"/>'
            )
            slot_id += 1
    inventory = "\n\t".join(inventory_lines)
    return inventory


def make_arena(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Returns pair of lines that place bedrock around, and air inside
    """
    # bedrock = f'<DrawCuboid x1="{x_min - 1}" y1="{y_min - 1}" z1="{z_min - 1}" x2="{x_max + 1}" y2="{y_max + 1}" z2="{z_max + 1}" type="bedrock"/>'
    # glowstone = f'<DrawCuboid x1="{x_min - 1}" y1="{y_max + 1}" z1="{z_min - 1}" x2="{x_max + 1}" y2="{y_max + 1}" z2="{z_max + 1}" type="glowstone"/>'
    air = f'<DrawCuboid x1="{x_min}" y1="{y_min}" z1="{z_min}" x2="{x_max}" y2="{y_max}" z2="{z_max}" type="air"/>'

    return [air]
    # return [bedrock, glowstone, air]


def make_malmo_domain(
    blocks,
    items,
    start_pos,
    inventory_counts,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    summary="Domain for Scoping",
    convert_coords=True,
):
    """
    All coordinates are PDDL coordinates. We will convert them to malmo coords.
    blocks: dict[type]-> list of positions. Make sure to only include present blocks
    items: dict[type] -> list of positions. Make sure to only include present items
    inventory: dict[type] -> count in inventory
    start_pos: x,y,z position of agent
    convert_coords: If true, we convert coords to malmo. If false, we assume coords are already in malmo format
    """
    # Update block and item types to match malmo types
    blocks_new = OrderedDict()
    for t, positions in blocks.items():
        t = type_replacements.get(t, t)
        blocks_new[t] = positions
    blocks = blocks_new
    del blocks_new

    items_new = OrderedDict()
    for t, positions in items.items():
        t = type_replacements.get(t, t)
        items_new[t] = positions
    items = items_new
    del items_new

    inventory_counts_new = OrderedDict()
    for t, positions in inventory_counts.items():
        t = type_replacements.get(t, t)
        inventory_counts_new[t] = positions
    inventory_counts = inventory_counts_new
    del inventory_counts_new
    # Convert to malmo coordinates
    if convert_coords:
        start_pos = pddl2malmo_coords(*start_pos)

        blocks_new = OrderedDict()
        for t, positions in blocks.items():
            blocks_new[t] = [pddl2malmo_coords(*p) for p in positions]
        blocks = blocks_new
        del blocks_new

        items_new = OrderedDict()
        for t, positions in items.items():
            items_new[t] = [pddl2malmo_coords(*p) for p in positions]
        items = items_new
        del items_new

        # y is negatized, which swaps max and min
        # x_min, y_min, z_min = x_min, z_min + 200, -y_max
        # x_min, y_min, z_min = x_min, z_min + 200, -y_max
        min_new = pddl2malmo_coords(x_min, y_max, z_min)
        max_new = pddl2malmo_coords(x_max, y_min, z_max)
        x_min, y_min, z_min = min_new
        x_max, y_max, z_max = max_new
        # x_min, y_min, z_min = pddl2malmo_coords(x_min, y_max, z_min)
        # x_max, y_max, z_max = pddl2malmo_coords(x_max, y_min, z_max)
    with open(f"{domains_dir}/malmo/templates/main_template.xml", "r") as f:
        domain = f.read()

    # Used to fill in blanks in template domain
    format_dict = OrderedDict()
    format_dict["arena_width"] = x_max - x_min + 1
    format_dict["arena_breadth"] = z_max - z_min + 1
    format_dict["arena_height"] = y_max - y_min + 1
    # Set summary
    # format_dict["summary"] = summary
    format_dict["summary"] = "{summary}"
    format_dict["video_requirements"] = "{video_requirements}"

    # Set agent start position
    format_dict["placement"] = make_agent_placement(*start_pos)
    # Set agent start inventory
    format_dict["inventory"] = make_inventory(inventory_counts)

    drawing_lines = []
    # Add bedrock and air
    drawing_lines.extend(make_arena(x_min, x_max, y_min, y_max, z_min, z_max))
    # Place blocks in the world
    for t, positions in blocks.items():
        for p in positions:
            drawing_lines.append(draw_block(t, *p))
    # Place items in the world
    for t, positions in items.items():
        if t in [
            "daisy-flower",
            "orchid-flower",
            "oxeye_daisy",
            "blue_orchid",
            "red_tulip",
        ]:
            draw_func = draw_flower
        elif t in ["birch", "oak"]:
            draw_func = draw_sapling
        else:
            draw_func = draw_item
        for p in positions:
            drawing_lines.append(draw_func(t, *p))

    format_dict["drawing_objects"] = "\n\t\t\t\t\t".join(drawing_lines)

    return domain.format(**format_dict)
