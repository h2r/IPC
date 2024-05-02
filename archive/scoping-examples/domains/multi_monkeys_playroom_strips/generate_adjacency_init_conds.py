object_definition_str_x = ""
object_definition_str_y = ""
init_conds_str = ""
grid_limit = 100
for x1 in range(1, grid_limit):
    object_definition_str_x += f"x{x1} "
    init_conds_str += f"(adjacent-x x{x1} x{x1+1})\n(adjacent-x x{x1+1} x{x1})\n"

for y1 in range(1, grid_limit):
    object_definition_str_y += f"y{y1} "
    init_conds_str += f"(adjacent-y y{y1} y{y1+1})\n(adjacent-y y{y1+1} y{y1})\n"

object_definition_str_x += f"x{grid_limit}"
object_definition_str_y += f"y{grid_limit}"

print(object_definition_str_x)
print(object_definition_str_y)
print(init_conds_str)
