begin_version
3
end_version
begin_metric
0
end_metric
5
begin_variable
var0
-1
2
Atom has-food(steve)
NegatedAtom has-food(steve)
end_variable
begin_variable
var1
-1
2
Atom hungry(steve)
NegatedAtom hungry(steve)
end_variable
begin_variable
var2
-1
2
Atom has-sticks(steve)
NegatedAtom has-sticks(steve)
end_variable
begin_variable
var3
-1
2
Atom has-stone(steve)
NegatedAtom has-stone(steve)
end_variable
begin_variable
var4
-1
2
Atom has-axe(steve)
NegatedAtom has-axe(steve)
end_variable
0
begin_state
1
1
1
1
1
end_state
begin_goal
2
1 1
4 0
end_goal
7
begin_operator
eat steve
0
2
0 0 0 1
0 1 0 1
1
end_operator
begin_operator
gather steve
1
1 0
1
0 0 1 0
1
end_operator
begin_operator
get_stick steve
0
1
0 2 1 0
1
end_operator
begin_operator
get_stone steve
0
1
0 3 1 0
1
end_operator
begin_operator
hunt steve
1
1 1
1
0 0 1 0
1
end_operator
begin_operator
make_axe steve
0
3
0 4 1 0
0 2 0 1
0 3 0 1
1
end_operator
begin_operator
wait steve
0
1
0 1 1 0
1
end_operator
0
