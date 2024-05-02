# Profiling and optimization notes
These observations are primarily made from scoping a minecraft domain.

## Bottlenecks
z3.__str__(). Seemingly responsible for most of the time of scope(). Each call takes 0.000 seconds, but we call it many times. If there is no way to speed up each call, we need to call it less. The obvious, but hard, solution is to lift our algorithm. I do not know other methods off hand. Perhaps we can speed it up by using a more bare-metal method to get the strings.

SkillPDDL.__repr__
    Used for sorting. Takes up half of merge_skills_pddl(). Likely slowed down by z3.__str__()