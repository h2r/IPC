from __future__ import annotations
from typing import (
    List,
    Iterable,
    Tuple,
    Hashable,
    Union,
    Optional,
    Set,
    Dict,
    OrderedDict,
	NewType
)
from itertools import chain, product
from collections import OrderedDict
import z3
import copy
from oo_scoping.utils import simplify_disjunction, flatten, get_unique_z3_vars


class EffectTypePDDL:  # EffectTypes are Immutable
    def __init__(self, pvar: z3.ExprRef, index, params: Optional[Iterable] = None):
        self.pvar, self.index = pvar, index
        # These params are NOT the same as pddl params, and should probably be renamed
        # These params are vars on which the effect relies - ex. in p.y <- t.y, t.y is a param
        self.params = params if params is not None else tuple()
        # Also save str()-cast versions of pvar, index and params because we end up
        # needing these strs a lot for comparison
        self.pvar_str = str(self.pvar)
        self.index_str = str(self.index)
        self.params_str = str(self.params)

    def __eq__(self, other):
        if not isinstance(other, EffectTypePDDL):
            return False
        return (
            z3.eq(self.pvar, other.pvar)
            and self.index == other.index
            and self.params == other.params
        )

    def __lt__(self, other: EffectTypePDDL):
        # TODO incorporate pvar type into sort.
        if self.pvar_str > other.pvar_str:
            return False
        elif self.pvar_str == other.pvar_str:
            if self.index_str >= other.index_str:
                return False
            elif self.index_str == other.index_str:
                if self.params_str >= other.params_str:
                    return False
        return True

    def __repr__(self):
        # return f"ET({self.pvar},{self.index},{self.params})"
        # I'm tempted to use unicode arrow
        return f"{self.pvar} <- {self.index}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((hash(self.pvar), hash(self.index), hash(self.params)))


class SkillPDDL:  # Skills are Immutable
    def __init__(
        self,
        precondition: z3.ExprRef,
        action: Union[str, List[str], Tuple[str, ...]],
        effects: Union[Iterable[EffectTypePDDL], EffectTypePDDL],
        side_effects: Union[Iterable[EffectTypePDDL], EffectTypePDDL] = None,
    ):
        if side_effects is None:
            side_effects = ()
        elif isinstance(side_effects, EffectTypePDDL):
            side_effects = (side_effects,)
        if isinstance(effects, EffectTypePDDL):
            effects = (effects,)
        # z3 doesn't like vanilla python bools, so we convert those to the z3-equivalent. This makes it so you can
        # pass in True or False as a precondition without ex. Skill.__eq__ throwing an error
        if isinstance(precondition, bool):
            precondition = z3.BoolVal(precondition)
        self.precondition, self.action = precondition, copy.copy(
            action
        )  # Copy in case we are passed a list
        if isinstance(effects, EffectTypePDDL):
            effects = [effects]
        self.effects: Tuple[EffectTypePDDL, ...] = tuple(sorted(set(effects)))
        self.side_effects: Tuple[EffectTypePDDL, ...] = tuple(sorted(set(side_effects)))
        self.action_str = str(action)

    @property
    def all_effects(self) -> Tuple[EffectTypePDDL, ...]:
        return tuple(set(self.effects + self.side_effects))

    def __eq__(self, other):
        if not isinstance(other, SkillPDDL):
            return False
        same_prec = z3.eq(self.precondition, other.precondition)
        same_action = self.action == other.action
        same_effects = self.effects == other.effects
        same_side_effects = self.side_effects == other.side_effects
        return same_prec and same_action and same_effects and same_side_effects

    def __repr__(self):
        s = (
            f"{self.action}\nPrecondition: {self.precondition}\nEffects: {self.effects}"
            f"\nSide Effects: {self.side_effects}"
        )
        return s

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        part_hashes = []
        for x in [self.precondition, self.action, self.effects, self.side_effects]:
            part_hashes.append(hash(x))
        return hash(tuple(part_hashes))

    def __lt__(self, other: SkillPDDL):
        # NOTE: This sort is arbitrary. We are defining it just to get a canonical ordering.
        # return self.str_repr < other.str_repr
        return self.action_str < other.action_str
        # return str(self) < str(other)

    def move_irrelevant2side_effects(self, rel_pvars: Set[str]):
        """Returns a new skill with irrelevant pvars moved to side effects"""
        # Check that no relevant vars are in side effects
        # for e in self.side_effects:
        # 	if e.pvar in relevant_pvars:
        # 		raise ValueError(f"Skill has relevant pvar in side effects:\n{self}")

        new_effects = []
        new_side_effects = move_copy_se(self.side_effects)
        move_inner_loop(self.effects, rel_pvars, new_effects, new_side_effects)
        return SkillPDDL(self.precondition, self.action, new_effects, new_side_effects)

    @property
    def params(self):
        params = []
        for x in self.effects:
            if hasattr(x, "params"):
                params.extend(x.params)
        return get_unique_z3_vars(params)
        # return tuple(sorted(list(set(params))))
        # return tuple(chain(*[x.params for x in self.effects]))

HashedSkills = NewType("HashedSkills", Dict[Tuple[EffectTypePDDL,...], List[SkillPDDL]])


def move_copy_se(x):
    return list(copy.copy(x))


def move_inner_loop(effects, rel_pvars, new_effects, new_side_effects):
    for e in effects:
        # if e.pvar in relevant_pvars:
        if move_inner_if(rel_pvars, e):
            new_effects.append(e)
        else:
            new_side_effects.append(e)


def move_inner_if(rel_pvars, e):
    pvar_str = move_inner_get_pvar(e)
    return move_inner_dict_get(rel_pvars, pvar_str)


def move_inner_get_pvar(e):
    return e.pvar_str


def move_inner_dict_get(rel_pvars, pvar_str):
    return pvar_str in rel_pvars



def group_skills_by_effects(
    skills: Iterable[SkillPDDL], rel_pvars: Set[str]
) -> HashedSkills:
    hashed_skills: HashedSkills = OrderedDict()
    for s in skills:
        s = s.move_irrelevant2side_effects(rel_pvars)
        k = s.effects
        if k not in hashed_skills.keys():
            hashed_skills[k] = []
        hashed_skills[k].append(s)
    return hashed_skills


def merge_skills_inner(hashed_skills: HashedSkills, solver: z3.Solver) -> List[SkillPDDL]:
    new_skills: List[SkillPDDL] = []
    for (effects), sks in hashed_skills.items():
        # Skip empty effects
        if len(effects) == 0:
            continue
        actions = sorted(list(set(flatten([s.action for s in sks]))))
        side_effects = chain(*[s.side_effects for s in sks])
        conds = [s.precondition for s in sks]
        precondition = simplify_disjunction(conds, my_solver=solver)
        # Actions is the list of actions that appeared in any of the parent skills
        if len(actions) == 1:
            actions = actions[0]
        s_merged = SkillPDDL(precondition, actions, effects, side_effects)
        new_skills.append(s_merged)
    return new_skills


def merge_skills_pddl(
    skills: Iterable[SkillPDDL],
    relevant_pvars: Iterable[z3.ExprRef],
    solver: Optional[z3.Solver] = None,
) -> List[SkillPDDL]:
    """
    Merges skills that have the same effects on relevant_pvars
    :param skills: Iterable of Skills
    :param relevant_pvars: Iterable of pvars, where each pvar is a z3 expression
    :param solver: A z3 solver. Use this arg to assume state constraints when simplifying disjunctions
    """
    rel_pvar_set: Set[str] = set(map(str, relevant_pvars))

    # Move irrelevant pvars to side effects and group skills by actions and effect types
    hashed_skills = group_skills_by_effects(skills, rel_pvar_set)

    # Merge skills that share a key
    new_skills = merge_skills_inner(hashed_skills, solver)

    def sort_in_merge_skills(x: List) -> List:
        """
        Used for profiling.
                This is just a wrapper around sorted.
                We should probably get rid fo this.
        """
        return sorted(x)

    return sort_in_merge_skills(new_skills)


if __name__ == "__main__":
    pass
