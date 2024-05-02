import pdb

from scoping.utils import condition_str2objects, get_all_objects, get_diff_and_int, str_iter
from scoping.hardcoded_blinker import prepare_taxi_domain as prepare_blinker_domain
from scoping.scoping import scope
from scoping.domain_writers.hardcoded_domains import make_domain

def test_blinker():
	# Won't work until we fix EffectTypes in hardcoded_blinker.py
	goals, skills, start_condition, pvars, state_constraints = prepare_blinker_domain(n_passegners=2, blinker=False, goal=(3,7))

	print(f"~~~Start condition~~~~\n{start_condition}")
	# pdb.set_trace()
	relevant_pvars, used_skills = scope(goals, skills, start_condition, state_constraints = state_constraints)
	print("\n~~~All Skills~~~")
	for s in skills: print(str(s) + "\n")
	print("\n~~~Relevant skills~~~")
	print("\n\n".join(map(str,used_skills)))
	# for s in used_skills: print(s)
	print("\n~~~All Objects~~~")
	all_objects = get_all_objects(skills)
	for o in all_objects: print(o)
	relevant_objects = []
	for p in relevant_pvars:
		relevant_objects.extend(condition_str2objects(str(p)))
	relevant_objects = sorted(list(set(relevant_objects)))
	print("\n~~~Relevant Objects~~~")
	for o in relevant_objects: print(o)
	print("\n~~~Relevant Pvars~~~")
	for p in relevant_pvars: print(p)

def test_scoping2():
	G, skills_rel, skills_ir, initial_condition, sv_rel, sv_ir = make_domain(causal_link = True, broken_causal_link = False, trivially_relevant = False,
				trivially_irrelevant = False, need_on_and_off = False)
	print(f"Initial Condition: {initial_condition}")
	print(f"Goal: {G}")
	skills = skills_rel + skills_ir
	sv = sv_rel + sv_ir
	relevant_pvars, used_skills = scope(G, skills, initial_condition)
	relevant_objects = []
	for p in relevant_pvars:
		relevant_objects.extend(condition_str2objects(str(p)))
	relevant_objects = sorted(list(set(relevant_objects)))
	sv_false_ir,  sv_correct_rel, sv_false_rel = get_diff_and_int(str_iter(sv_rel), str_iter(relevant_pvars))
	print("~~~~~Pvars~~~~~")
	print(f"Correctly relevant: {sv_correct_rel}")
	print(f"Falsely irrelevant: {sv_false_ir}")
	print(f"Falsely relevant: {sv_false_rel}")

	print("\n~~~~Skills~~~~")
	skills_false_ir,  skills_correct_rel, skills_false_rel = get_diff_and_int(str_iter(skills_rel), str_iter(used_skills))
	print(f"\n~~Correctly Relevant~~:")
	for s in skills_correct_rel: print(s,"\n")
	print(f"\n~~Falsely irrelevant~~:")
	for s in skills_false_ir: print(s,"\n")
	print(f"\n~~Falsely relevant~~:")
	for s in skills_false_rel: print(s,"\n")

def test_scoping3():
	G, skills_rel, skills_ir, initial_condition, sv_rel, sv_ir = make_domain(causal_link = True, broken_causal_link = True, trivially_relevant = True,
				trivially_irrelevant = True, need_on_and_off = True)
	print(f"Initial Condition: {initial_condition}")
	print(f"Goal: {G}")
	skills = skills_rel + skills_ir
	sv = sv_rel + sv_ir
	relevant_pvars, used_skills = scope(G, skills, initial_condition)
	relevant_objects = []
	for p in relevant_pvars:
		relevant_objects.extend(condition_str2objects(str(p)))
	relevant_objects = sorted(list(set(relevant_objects)))
	sv_false_ir,  sv_correct_rel, sv_false_rel = get_diff_and_int(str_iter(sv_rel), str_iter(relevant_pvars))
	print("~~~~~Pvars~~~~~")
	# print(f"Correctly relevant: {sv_correct_rel}")
	print(f"Falsely irrelevant: {sv_false_ir}")
	print(f"Falsely relevant: {sv_false_rel}")

	print("\n~~~~Skills~~~~")
	skills_false_ir,  skills_correct_rel, skills_false_rel = get_diff_and_int(str_iter(skills_rel), str_iter(used_skills))
	# print(f"\n~~Correctly Relevant~~:")
	# for s in skills_correct_rel: print(s,"\n")
	print(f"\n~~Falsely irrelevant~~:")
	for s in skills_false_ir: print(s,"\n")
	print(f"\n~~Falsely relevant~~:")
	for s in skills_false_rel: print(s,"\n")



if __name__ == "__main__":
	# test_blinker()
	# test_scoping3()
	test_blinker()