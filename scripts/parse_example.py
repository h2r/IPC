from action import Action
from PDDL import PDDL_Parser
import sys, pprint



# parser = PDDL_Parser()
# parser.scan_tokens(domain_path)
#
# domain = sys.argv[1]
# problem = sys.argv[2]
def print_parse(domain, problem):
	parser = PDDL_Parser()
	print('----------------------------')
	# pprint.pprint(parser.scan_tokens(problem))
	print('----------------------------')
	parser.parse_domain(domain)
	parser.parse_problem(problem)
	print('Domain name: ' + parser.domain_name)
	print(f'~~~Types~~~')
	for t in parser.types: print(t)
	print(f'~~~Predicates~~~')
	for nm, args in parser.predicates.items():
		print(f"{nm}({args}")
	#     print("")
	print('----------------------------')
	print(f'~~~Functions~~~')
	for nm, args in parser.functions.items():
		print(f"{nm}({args}")
	#     print("")
	print('----------------------------')
	print('~~~Actions~~~')
	for act in parser.actions:
		print(act)
		# print("effect types:")
		# for ef in action2EffectTypes(act): print(f"\t{ef}")
		print("\n")
	print('----------------------------')
	print('Problem name: ' + parser.problem_name)
	print('Objects: ' + str(parser.objects))
	print('State: ' + str(parser.state))
	print('Positive goals: ' + str(parser.positive_goals))
	print('Negative goals: ' + str(parser.negative_goals))

dinner_dom = "./domains/dinner/dinner.pddl"
dinner_prob = "./domains/dinner/pb1.pddl"

zeno_dom = "domains/zeno/zeno.pddl"
zeno_prob = "domains/zeno/pb1.pddl"

# print_parse(dinner_dom, dinner_prob)
print_parse(zeno_dom, zeno_prob)