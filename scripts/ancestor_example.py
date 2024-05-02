import pathlib

p_orig = pathlib.Path(__file__)
p_old = None
p = p_orig

for i in range(10):
    if p_old == p: break
    print(i)
    print(p)
    p_old = p
    p = p.parent
print(f"Path '{p_orig}' is {i - 1} deep")