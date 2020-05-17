from typing import List, Iterable, Tuple, Hashable
from itertools import chain
from collections import OrderedDict
import z3
import copy
from utils import simplify_disjunction

class EffectType():  #EffectTypes are Immutable
	def __init__(self, pvar: z3.ExprRef, index: int):
		self.pvar, self.index = pvar, index
	def __eq__(self, other):
		return z3.eq(self.pvar, other.pvar) and self.index == other.index
	def __lt__(self, other):
		if self.pvar > other.pvar: return False
		if self.pvar == other.pvar and self.index >= other.index: return False
		return True
	def __repr__(self):
		return f"ET({self.pvar},{self.index})"
	def __hash__(self):
		return hash((hash(self.pvar), hash(self.index)))

class Skill(): #Skills are Immutable
	def __init__(self, precondition: z3.ExprRef, action: str, effects: Iterable[EffectType]
				 , side_effects: Iterable[EffectType] = None):
		if side_effects is None: side_effects = ()
		# z3 doesn't like vanilla python bools, so we convert those to the z3-equivalent. This makes it so you can
		# pass in True or False as a precondition without ex. Skill.__eq__ throwing an error
		if isinstance(precondition,bool): precondition = z3.BoolVal(precondition)
		self.precondition, self.action = precondition, action
		self.effects: Tuple[EffectType] = tuple(sorted(set(effects)))
		self.side_effects: Tuple[EffectType] = tuple(sorted(set(side_effects)))
	@property
	def all_effects(self) -> Tuple[EffectType]:
		return tuple(set(self.effects + self.side_effects))
	def __eq__(self, other):
		if not isinstance(other, Skill): return False
		same_prec = z3.eq(self.precondition,other.precondition)
		same_action = (self.action == other.action)
		same_effects = self.effects == other.effects
		same_side_effets = self.side_effects == other.side_effects
		return  same_prec and same_action and same_effects and same_side_effets
	def __repr__(self):
		s = f"Precondition: {self.precondition}\nAction: {self.action}\nEffects: {self.effects}" \
			f"\nSide Effects: {self.side_effects}"
		return s
	def move_irrelevant2side_effects(self, relevant_pvars):
		"""Returns a new skill with irrelevant pvars moved to side effects"""
		# Check that no relevant vars are in side effects
		for e in self.side_effects:
			if e.pvar in relevant_pvars:
				raise ValueError(f"Skill has relevant pvar in side effects:\n{self}")

		new_effects = []
		new_side_effects = copy.copy(self.side_effects)
		for e in self.effects:
			if e.pvar in relevant_pvars:
				new_effects.append(e)
			else:
				new_side_effects.append(e)
		return Skill(self.precondition, self.action, new_effects, new_side_effects)

def merge_skills(skills: Iterable[Skill], relevant_pvars: Iterable[z3.ExprRef]):
	new_skills = []
	hashed_skills = OrderedDict()
	# Move irrelevant pvars to side effects and group skills by actions and effect types
	for s in skills:
		s.move_irrelevant2side_effects(relevant_pvars)
		k = (s.action, s.effects)
		if k not in hashed_skills.keys(): hashed_skills[k] = []
		hashed_skills[k].append(s)
	# Merge skills that share a key
	for (action, effects), sks in hashed_skills.items():
		side_effects = chain(*[s.side_effects for s in sks])
		precondition = simplify_disjunction([s.precondition for s in sks])
		new_skills.append(Skill(precondition, action, effects, side_effects))
	return new_skills


def test_effect_type_eq():
	pvar0 = z3.Bool("p0")
	pvar1 = z3.Bool("p1")
	et0 = EffectType(pvar0, 0)
	et0_1 = EffectType(pvar0,0)
	print((et0 == et0_1) == True)
	et1 = EffectType(pvar1, 0)
	print((et0 == et1) == False)

def test_skill_eq():
	pvars = [z3.Bool(f"b{i}") for i in range(2)]
	ets = [EffectType(p,0) for p in pvars]
	s0 = Skill(True, "action", [ets[0]])
	s0_copy = Skill(True, "action", [ets[0]])
	print(s0==s0_copy)
	s1 = Skill(pvars[1], "action", [ets[0]])
	print(s0 != s1)

def test_merge_skills():
	pvars = [z3.Bool(f"b{i}") for i in range(2)]
	ets = [EffectType(p,0) for p in pvars]
	# sboth = Skill(pvars[0], "action_both", ets)
	s0 = Skill(pvars[0], "action", [ets[0]])
	s1 = Skill(z3.Not(pvars[0]), "action", [ets[0]])
	merged = merge_skills([s0,s1], [pvars[0]])
	for m in merged:
		print(m)
	merged_correct = [Skill(z3.BoolVal(True), "action", [ets[0]])]
	print(f"\nCorrect: {merged_correct == merged}")
if __name__ == "__main__":
	# test_merge_skills()
	# test_effect_type_eq()
	test_skill_eq()