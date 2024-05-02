from oo_scoping.PDDL import PDDL_Parser, Action
import z3, re
from collections import OrderedDict
from oo_scoping.skill_classes import EffectTypePDDL, SkillPDDL
from oo_scoping.utils import (
    product_dict,
    nested_list_replace,
    get_atoms,
    get_unique_z3_vars,
)
from typing import List, Tuple, Dict, Iterable, Optional, Type, Union


class PDDL_Parser_z3(PDDL_Parser):
    def make_z3_atoms(
        self, things_dict: Dict, z3_class: Type, str2var: Optional[Dict] = None
    ) -> Dict[str, Union[z3.Bool, z3.Int]]:
        """
        things_dict should be parser.predicates or parser.functions
        z3_class should be z3.Bool or z3.Int
        str2var is the dictionary you want to update. If none, a new OrderedDict is created
        """
        if str2var is None:
            str2var = OrderedDict()
        for p_name, p_args in things_dict.items():
            groundings = product_dict(
                **OrderedDict(
                    [
                        (varnm, self.get_objects_of_type(vartype))
                        for (varnm, vartype) in p_args.items()
                    ]
                )
            )
            for x in groundings:
                s = p_name + "(" + ", ".join(x.values()) + ")"
                grounded_predicate = z3_class(s)
                assert (
                    s not in str2var.keys()
                ), f"{s}: {str2var[s]}, {grounded_predicate}"
                str2var[s] = grounded_predicate
        return str2var

    # TODO make this a property?
    def make_str2var_dict(self) -> Dict:
        str2var_dict = OrderedDict()
        str2var_dict = self.make_z3_atoms(self.predicates, z3.Bool, str2var_dict)
        str2var_dict = self.make_z3_atoms(self.functions, z3.Int, str2var_dict)
        return str2var_dict

    def str_grounded_actions2skills(self, str_grounded_actions, str2var_dict):
        skill_list = []
        for action_class in str_grounded_actions:
            for grounded_action in action_class:
                precond = self.action2precondition(grounded_action, str2var_dict)
                effect_types = action2effect_types(grounded_action, str2var_dict, self)
                skill = SkillPDDL(precond, grounded_action.name, effect_types)
                skill_list.append(skill)
        return skill_list

    def action2precondition(self, a: Action, str_var_dict) -> z3.BoolRef:
        clauses = [
            compile_expression(p, str_var_dict, parser=self)
            for p in a.positive_preconditions
        ] + [
            z3.Not(compile_expression(p, str_var_dict, parser=self))
            for p in a.negative_preconditions
        ]
        return z3.And(*clauses)

    def get_skills(self):
        str2var_dict = self.make_str2var_dict()
        # print("Got str2var dict")
        str_grounded_actions = [self.get_action_groundings(a) for a in self.actions]
        # print("Got action groundings")
        skill_list = self.str_grounded_actions2skills(
            str_grounded_actions, str2var_dict
        )
        # print("Got skill list")
        return skill_list

    def get_goal_cond(self):
        str2var_dict = self.make_str2var_dict()
        goal_list = [
            compile_expression(pos_goal_expr, str2var_dict, self)
            for pos_goal_expr in self.positive_goals
        ]
        goal_list += [
            z3.Not(compile_expression(neg_goal_expr, str2var_dict, self))
            for neg_goal_expr in self.negative_goals
        ]
        goal_cond = z3.And(*goal_list)
        return goal_cond

    def get_init_cond_list(self, pred_default=False):
        str2var_dict = self.make_str2var_dict()
        init_cond_list = [
            compile_expression(init_cond, str2var_dict, self)
            for init_cond in self.state
        ]
        # Set the preds not mentioned in the intitial state to the pred_default.
        if pred_default is not None:
            pred_processor = {True: lambda p: p, False: z3.Not}[pred_default]
            # Set boolean pvars not mentioned in initial condition to False
            pvars_in_init = []
            for c in init_cond_list:
                pvars_in_init.extend(get_atoms(c))
            pvars_in_init = get_unique_z3_vars(pvars_in_init)
            all_grounded_preds_str = []
            for p in self.predicates.items():
                all_grounded_preds_str.extend(self.get_predicate_groundings(p))
            str2var = self.make_str2var_dict()
            preds_not_in_init = [
                str2var[p]
                for p in all_grounded_preds_str
                if p not in map(str, pvars_in_init)
            ]
            # Assert the preds not mentioned in the initial state are False
            for p in preds_not_in_init:
                init_cond_list.append(pred_processor(p))
        return init_cond_list


str2operator = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "=": lambda a, b: a == b,
    "*": lambda a, b: a * b,
    "+": lambda a, b: a + b,
    "/": lambda a, b: a / b,
    "-": lambda a, b: a - b,
    "and": z3.And,
    "or": z3.Or,
    "increase": lambda a, b: a + b,
    "decrease": lambda a, b: a - b,
    "assign": lambda a, b: b,
}


def list2var_str(x):
    return x[0] + "(" + ", ".join(x[1:]) + ")"


def list_is_flat(l):
    for x in l:
        if isinstance(x, list):
            return False
    return True


def z3_identical(a, b):
    return a.sort() == b.sort() and str(a) == str(b)


def parse_tokens2(my_str):
    stack = []
    l2 = []
    for t in re.findall(r"[()]|[^\s()]+", my_str):
        if t == "(":
            stack.append(l2)
            l2 = []
        elif t == ")":
            if stack:
                l = l2
                l2 = stack.pop()
                l2.append(l)
            else:
                raise Exception("Missing open parentheses")
        else:
            l2.append(t)
    if stack:
        raise Exception("Missing close parentheses")
    if len(l2) != 1:
        raise Exception("Malformed expression")
    return l2[0]


def split_predicates2(group):
    pos = []
    neg = []
    if not type(group) is list:
        raise Exception("Error with ")
    if group == []:
        # pass
        return
        # from IPython import embed; embed()
    if group[0] == "and":
        group.pop(0)
    else:
        group = [group]
    for predicate in group:
        if predicate[0] == "not":
            if len(predicate) != 2:
                raise Exception("Unexpected not in ")
            neg.append(predicate[-1])
        else:
            pos.append(predicate)
    return pos, neg


def str2expression(cond_s: str, parser: PDDL_Parser_z3):
    str_var_dict = parser.make_str2var_dict()
    tokens = parse_tokens2(cond_s)
    pos, neg = split_predicates2(tokens)
    pos = [compile_expression(p, str_var_dict, parser) for p in pos]
    neg = [z3.Not(compile_expression(n, str_var_dict, parser)) for n in neg]
    return z3.And(*pos, *neg)


def compile_expression(expr, str_var_dict, parser=None):

    """
    :param expr - a (potentially nested) list of strings. Should be grounded to objects, except for quantified predicates
    :param str_var_dict - a dictionary mapping strings representing pvars to their corresponding z3 objects
    :param parser - a PDDL_Parser. Only needed to ground quantified objects
    """
    if isinstance(expr, int):
        return expr
    elif isinstance(expr, str):
        # the only str we can interpret in a vacuum is an int. (maybe bools later?)
        try:
            expr = int(expr)
            return expr
        except ValueError as e:
            # from IPython import embed; embed() #What is embed?
            raise e
    elif isinstance(expr, list):
        # A flat list corresponds to a pvar
        if list_is_flat(expr):
            # check equality
            if expr[0] == "=":
                return expr[1] == expr[2]
            else:
                # try:
                return str_var_dict[list2var_str(expr)]
                # except KeyError:
                #     from IPython import embed; embed()
        elif len(expr) == 2:
            # The only length 2 expression we can compile is ['not', [subexpression]]
            if expr[0] == "not":
                return z3.Not(compile_expression(expr[1], str_var_dict, parser))
            elif expr[0] in ["and", "or"]:
                if list_is_flat(expr[1]):
                    return compile_expression(expr[1], str_var_dict, parser)
                else:
                    return compile_expression([expr[0]] + expr[1], str_var_dict, parser)
            else:
                raise ValueError(f"Don't understand how to compile: {expr}")
        else:
            # assert len(expr) == 3, f"Don't understand how to compile: {expr}"
            # Handle quantifiers. It is a bit odd that we ground quantifiers here, not during grounding
            # Example quantifier: ['forall', ['?pn', '-', 'passenger'], ['not', ['in-taxi', '?pn', '?t']]]]
            if expr[0] in ["forall", "exists"]:
                # TODO edit to work with multiple quantified vars, ex (forall ?be - bell ?mo - monkey)
                quantifier, quantified_var, subexpr = expr
                obj2type = extract_typed_objects(quantified_var)
                # varnm, vartype = quantified_var[0], quantified_var[2]
                # Get all groundings for the quantified object
                obj2groundings = OrderedDict()
                for o, t in obj2type.items():
                    obj2groundings[o] = parser.get_objects_of_type(t)
                groundings = list(
                    product_dict(**obj2groundings)
                )  # Probably don't need list here
                # var_groundings = parser.get_objects_of_type(vartype)
                # Get all grounded versions of the subexpression
                # subexpr_groundings = [nested_list_replace(subexpr, {varnm: x}) for x in var_groundings]
                subexpr_groundings = []
                for g in groundings:
                    subexpr_g = nested_list_replace(subexpr, g)
                    subexpr_groundings.append(subexpr_g)
                # subexpr_groundings = [nested_list_replace(subexpr, x) for x in groundings]

                # Compile the grounded subexpressions
                compiled_subexpressions = [
                    compile_expression(x, str_var_dict, parser)
                    for x in subexpr_groundings
                ]
                # Combine the compiled subexpressions
                combinator = {"forall": z3.And, "exists": z3.Or}[quantifier]
                return combinator(*compiled_subexpressions)
            # Handle operations
            else:
                operator = str2operator[expr[0]]
                operator_args = [
                    compile_expression(x, str_var_dict, parser) for x in expr[1:]
                ]
                return operator(*operator_args)
    else:
        raise NotImplementedError(
            f"Don't understand how to compile non lists: {expr}; {type(expr)}"
        )


def extract_typed_objects(l):
    obj2type = OrderedDict()
    objs = []
    next_is_type = False
    for x in l:
        if next_is_type:
            for o in objs:
                obj2type[o] = x
            next_is_type = False
            objs = []
        elif x == "-":
            next_is_type = True
        else:
            x = x.replace(",", "").replace(" ", "")
            objs.append(x)
    return obj2type


def action2effect_types(a: Action, str_var_dict, parser=None) -> List[EffectTypePDDL]:
    effect_types = []

    for eff in a.add_effects:
        # Check for complex effects like 'increase'
        if eff[0] in ["increase", "decrease", "assign"]:
            effected_var = str_var_dict[list2var_str(eff[1])]
            effect_details = compile_expression(eff, str_var_dict, parser)
            params = tuple(
                [
                    x
                    for x in get_atoms(effect_details)
                    if not z3_identical(x, effected_var)
                ]
            )
            effect_type = EffectTypePDDL(effected_var, effect_details, params=params)
            effect_types.append(effect_type)
        else:
            effected_var = compile_expression(eff, str_var_dict, parser)
            # This may accidentally cause different EfectTypes to be identified bc True == 1.
            effect_details = True
            effect_type = EffectTypePDDL(effected_var, effect_details)
            effect_types.append(effect_type)
    # Assume for now that del_effects only sets bools to false
    for eff in a.del_effects:
        effected_var = compile_expression(eff, str_var_dict, parser)
        # This may accidentally cause different EfectTypes to be identified bc True == 1.
        effect_details = False
        effect_type = EffectTypePDDL(effected_var, effect_details)
        effect_types.append(effect_type)
    return tuple(effect_types)


if __name__ == "__main__":
    domain, problem = (
        "./domains/existential-taxi/taxi-domain.pddl",
        "./domains/existential-taxi/prob02.pddl",
    )
    parser = PDDL_Parser_z3()
    parser.parse_domain(domain)
    parser.parse_problem(problem)
    skills = parser.get_skills()
    print("\n\n".join(map(str, skills)))
