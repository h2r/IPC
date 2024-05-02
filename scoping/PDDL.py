#!/usr/bin/env python
# Four spaces as indentation [no tabs]
# Original version: https://github.com/pucrs-automated-planning/pddl-parser/blob/master/PDDL.py
import re
from collections import OrderedDict
from itertools import chain, product
from oo_scoping.action import Action
import copy
from oo_scoping.utils import product_dict, nested_list_replace
from typing import Dict


class PDDL_Parser:
    # TODO convert type hierarchy to ordered dict (parent: [children])

    SUPPORTED_REQUIREMENTS = [
        ":strips",
        ":negative-preconditions",
        ":typing",
        ":fluents",
        ":universal-preconditions",
        ":existential-preconditions",
        ":equality",
    ]

    # ------------------------------------------
    # Tokens
    # ------------------------------------------
    def scan_tokens(self, filename):
        with open(filename, "r") as f:
            # Remove single line comments
            my_str = re.sub(r";.*$", "", f.read(), flags=re.MULTILINE).lower()
        # Tokenize
        stack = []
        list = []
        for t in re.findall(r"[()]|[^\s()]+", my_str):
            if t == "(":
                stack.append(list)
                list = []
            elif t == ")":
                if stack:
                    l = list
                    list = stack.pop()
                    list.append(l)
                else:
                    raise Exception("Missing open parentheses")
            else:
                list.append(t)
        if stack:
            raise Exception("Missing close parentheses")
        if len(list) != 1:
            raise Exception("Malformed expression")
        return list[0]

    # This version handles types correctly, except it currently doesn't
    def scan_tokens_w_types(self, filename):
        with open(filename, "r") as f:
            # Remove single line comments
            my_str = re.sub(r";.*$", "", f.read(), flags=re.MULTILINE).lower()
        # Tokenize
        stack = []
        list = []
        # The :types statement uses newlines as syntax. This regex ignores newlines.
        # Workaround: Handle the :type elsewhere, and insert into the list
        in_type = False
        type_start = None
        for t in re.findall(r"[()]|[^\s()]+", my_str):
            if t == ":types":
                in_type = True
                type_start = len(stack)
            if t == "(":
                stack.append(list)
                list = []
            elif t == ")":
                in_type = False
                if stack:
                    l = list
                    list = stack.pop()
                    list.append(l)
                else:
                    raise Exception("Missing open parentheses")
            else:
                if not in_type:
                    list.append(t)
        if stack:
            raise Exception("Missing close parentheses")
        if len(list) != 1:
            raise Exception("Malformed expression")
        # Handle :types
        types_str = re.findall(
            ":types ([^()]*)", str, flags=(re.DOTALL | re.MULTILINE)
        )[0]
        if type_start is not None:
            list[0].insert(type_start, types_str)
        return list[0]

    # -----------------------------------------------
    # Parse domain
    # -----------------------------------------------

    def parse_domain(self, domain_filename: str) -> None:
        tokens = self.scan_tokens(domain_filename)
        if type(tokens) is list and tokens.pop(0) == "define":
            self.domain_name = "unknown"
            self.requirements = []
            self.types = []
            self.actions = []
            self.predicates = OrderedDict()
            self.functions = OrderedDict()
            while tokens:
                group = tokens.pop(0)
                t = group.pop(0)
                if t == "domain":
                    self.domain_name = group[0]
                elif t == ":requirements":
                    for req in group:
                        if not req in self.SUPPORTED_REQUIREMENTS:
                            raise Exception("Requirement " + req + " not supported")
                    self.requirements = group
                elif t == ":predicates":
                    self.parse_predicates(group)
                elif t == ":functions":
                    self.parse_functions(group)
                elif t == ":types":
                    # self.types = group
                    # This parser does the right thing when passed the entires types string
                    # The token scanner with types doens't currently work, so for now we just set types to the
                    # group list
                    # self.parse_types(group)
                    self.domain2types(domain_filename)
                elif t == ":action":
                    self.parse_action(group)
                else:
                    print(str(t) + " is not recognized in domain")
        else:
            raise Exception(
                "File " + domain_filename + " does not match domain pattern"
            )

    # -----------------------------------------------
    # Parse predicates
    # -----------------------------------------------

    def parse_predicates(self, group):
        for pred in group:
            predicate_name = pred.pop(0)
            if predicate_name in self.predicates:
                raise Exception("Predicate " + predicate_name + " redefined")
            arguments = OrderedDict()
            untyped_variables = []  # Not always untyped
            while pred:
                t = pred.pop(0)
                if t == "-":
                    if not untyped_variables:
                        raise Exception("Unexpected hyphen in predicates")
                    type = pred.pop(0)
                    while untyped_variables:
                        arguments[
                            untyped_variables.pop(0)
                        ] = type  # These untyped vars are typed...
                else:
                    untyped_variables.append(t)
            while untyped_variables:
                arguments[untyped_variables.pop(0)] = "object"
            self.predicates[predicate_name] = arguments

    # -----------------------------------------------
    # Parse functions
    # -----------------------------------------------

    def parse_functions(self, group):
        for fun in group:
            function_name = fun.pop(0)
            if function_name in self.functions:
                raise Exception("Function " + function_name + " redefined")
            arguments = OrderedDict()
            untyped_variables = []
            while fun:
                t = fun.pop(0)
                if t == "-":
                    if not untyped_variables:
                        raise Exception("Unexpected hyphen in functions")
                    type = fun.pop(0)
                    while untyped_variables:
                        arguments[untyped_variables.pop(0)] = type
                else:
                    untyped_variables.append(t)
            while untyped_variables:
                arguments[untyped_variables.pop(0)] = "object"
            self.functions[function_name] = arguments

    # -----------------------------------------------
    # Parse types
    # -----------------------------------------------

    def parse_types(self, group):
        types_lines = group.replace("\t", "").split("\n")
        # [(subtype, parentype)]
        type_hierarchy = OrderedDict()
        # child_pa = []
        for l in types_lines:
            l_split = l.split(" - ")
            subtypes = l_split[0].split(" ")
            base_type = l_split[-1]
            for st in subtypes:
                if base_type not in type_hierarchy.keys():
                    type_hierarchy[base_type] = []
                type_hierarchy[base_type].append(st)
                # type_hierarchy.append((st, base_type))
        # for tp in type_hierarchy: print(tp)
        for k, v in type_hierarchy.items():
            type_hierarchy[k] = sorted(v)
        # Sort type hierarchy (should probably sort by depth instead of alphabet)
        type_hierarchy_sorted = OrderedDict()
        for k in sorted(type_hierarchy.keys()):
            type_hierarchy_sorted[k] = type_hierarchy[k]
        type_hierarchy = type_hierarchy_sorted
        self.type_hierarchy = type_hierarchy_sorted
        self.types = list(type_hierarchy.keys())

    def domain2types(self, domain_filename):
        with open(domain_filename, "r") as f:
            domain_str = f.read()
        types_str = re.findall(
            ":types ([^()]*)", domain_str, flags=(re.DOTALL | re.MULTILINE)
        )[0]
        types_lines = types_str.replace("\t", "").split("\n")
        # [(subtype, parentype)]
        type_hierarchy = OrderedDict()
        # child_pa = []
        for l in types_lines:
            l_split = l.split(" - ")
            subtypes = l_split[0].split(" ")
            base_type = l_split[-1]
            for st in subtypes:
                if base_type not in type_hierarchy.keys():
                    type_hierarchy[base_type] = []
                if st not in type_hierarchy.keys():
                    type_hierarchy[st] = []
                type_hierarchy[base_type].append(st)
                # type_hierarchy.append((st, base_type))
        # for tp in type_hierarchy: print(tp)
        for k, v in type_hierarchy.items():
            type_hierarchy[k] = sorted(v)
        # Sort type hierarchy (should probably sort by depth instead of alphabet)
        type_hierarchy_sorted = OrderedDict()
        for k in sorted(type_hierarchy.keys()):
            type_hierarchy_sorted[k] = type_hierarchy[k]
        type_hierarchy = type_hierarchy_sorted
        self.type_hierarchy = type_hierarchy_sorted
        self.types = list(type_hierarchy.keys())
        # from IPython import embed; embed()

    # -----------------------------------------------
    # Parse action
    # -----------------------------------------------

    def parse_action(self, group):
        name = group.pop(0)
        if not type(name) is str:
            raise Exception("Action without name definition")
        for act in self.actions:
            if act.name == name:
                raise Exception("Action " + name + " redefined")
        parameters = []
        positive_preconditions = []
        negative_preconditions = []
        add_effects = []
        del_effects = []
        while group:
            t = group.pop(0)
            if t == ":parameters":
                if not type(group) is list:
                    raise Exception("Error with " + name + " parameters")
                parameters = []
                untyped_parameters = []
                p = group.pop(0)
                while p:
                    t = p.pop(0)
                    if t == "-":
                        if not untyped_parameters:
                            raise Exception(
                                "Unexpected hyphen in " + name + " parameters"
                            )
                        ptype = p.pop(0)
                        while untyped_parameters:
                            parameters.append([untyped_parameters.pop(0), ptype])
                    else:
                        untyped_parameters.append(t)
                while untyped_parameters:
                    parameters.append([untyped_parameters.pop(0), "object"])
            elif t == ":precondition":
                self.split_predicates(
                    group.pop(0),
                    positive_preconditions,
                    negative_preconditions,
                    name,
                    " preconditions",
                )
            elif t == ":effect":
                self.split_predicates(
                    group.pop(0), add_effects, del_effects, name, " effects"
                )
            else:
                print(str(t) + " is not recognized in action")
        self.actions.append(
            Action(
                name,
                parameters,
                positive_preconditions,
                negative_preconditions,
                add_effects,
                del_effects,
            )
        )

    # -----------------------------------------------
    # Parse problem
    # -----------------------------------------------

    def parse_problem(self, problem_filename):
        tokens = self.scan_tokens(problem_filename)
        if type(tokens) is list and tokens.pop(0) == "define":
            self.problem_name = "unknown"
            self.objects = dict()
            self.state = []
            self.positive_goals = []
            self.negative_goals = []
            while tokens:
                group = tokens.pop(0)
                t = group[0]
                if t == "problem":
                    self.problem_name = group[-1]
                elif t == ":domain":
                    if self.domain_name != group[-1]:
                        raise Exception("Different domain specified in problem file")
                elif t == ":requirements":
                    pass  # Ignore requirements in problem, parse them in the domain
                elif t == ":objects":
                    group.pop(0)
                    object_list = []
                    while group:
                        if group[0] == "-":
                            group.pop(0)
                            curr_object_type = group.pop(0)
                            if self.objects.get(curr_object_type) is None:
                                self.objects[curr_object_type] = object_list
                            else:
                                self.objects[curr_object_type] += object_list
                            object_list = []
                        else:
                            object_list.append(group.pop(0))
                    if object_list:
                        if not "object" in self.objects:
                            self.objects["object"] = []
                        self.objects["object"] += object_list
                elif t == ":init":
                    group.pop(0)
                    self.state = group
                elif t == ":goal":
                    self.split_predicates(
                        group[1], self.positive_goals, self.negative_goals, "", "goals"
                    )
                else:
                    print(str(t) + " is not recognized in problem")
        else:
            raise Exception(
                "File " + problem_filename + " does not match problem pattern"
            )

    # -----------------------------------------------
    # Split predicates
    # -----------------------------------------------

    def split_predicates(self, group, pos, neg, name, part):
        if not type(group) is list:
            raise Exception("Error with " + name + part)
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
                    raise Exception("Unexpected not in " + name + part)
                neg.append(predicate[-1])
            else:
                pos.append(predicate)

    # -----------------------------------------------
    # Get subtypes
    # -----------------------------------------------
    def get_subtypes(self, ancestors):
        """Note: a type is its own subtype"""
        if isinstance(ancestors, str):
            ancestors = [ancestors]
        return get_descendants(self.type_hierarchy, ancestors)

    # -----------------------------------------------
    # Get objects belonging to type
    # -----------------------------------------------
    def get_objects_of_type(self, my_types, subtypes=True):
        """
        :param my_types: type of object to get, or an iterable of types
        :param subtypes: If true, also get objects that are subtypes of my_types.
        """
        if isinstance(my_types, str):
            my_types = [my_types]
        if subtypes:
            my_types = self.get_subtypes(my_types)
        valid_objects = []
        for t in my_types:
            if t in self.objects.keys():
                valid_objects.extend(self.objects[t])
        return valid_objects

    def get_predicate_groundings(self, p):
        """
        Get's all groundings for a pddl predicate - ie a boolean pvar
        :param p: (pred_name, Dict mapping param name to type)
        """
        pred_name, pred_params = p
        # If there are no params
        if len(pred_params.keys()) == 0:
            return [f"{pred_name}()"]
        # If there are params
        else:
            grounding_dicts = product_dict(
                **OrderedDict(
                    [
                        (varnm, self.get_objects_of_type(vartype))
                        for (varnm, vartype) in pred_params.items()
                    ]
                )
            )
            grounded_preds = []
            for d in grounding_dicts:
                pred_s = pred_name + "(" + ", ".join(d.values()) + ")"
                grounded_preds.append(pred_s)
            return grounded_preds

    def get_action_groundings(self, a):
        # We handle quantifier grounding when converting to z3 expressions
        # Ground non-quantiifed vars in all possible ways
        grounding_dicts = product_dict(
            **OrderedDict(
                [
                    (varnm, self.get_objects_of_type(vartype))
                    for (varnm, vartype) in a.parameters
                ]
            )
        )
        grounded_actions = []
        for x in grounding_dicts:
            att_names = [
                "name",
                "parameters",
                "positive_preconditions",
                "negative_preconditions",
                "add_effects",
                "del_effects",
            ]
            grounded_atts = OrderedDict(
                [(s, nested_list_replace(getattr(a, s), x)) for s in att_names]
            )
            grounded_action = Action(**grounded_atts)
            grounded_actions.append(grounded_action)
        return grounded_actions

    def get_state_variable_count(self) -> int:
        """Return the number of grounded state variables.
        This works by computing all groundings for lifted predicates
        and lifted functions.

        This assumes that the parser has already run 
        self.parse_domain and self.parse_problem
        """
        grounded_state_vars = []
        for pred_name, pred_vars in self.predicates.items():
            groundings = self.get_predicate_groundings((pred_name, pred_vars))
            grounded_state_vars.extend(groundings)
        for pred_name, pred_vars in self.functions.items():
            groundings = self.get_predicate_groundings((pred_name, pred_vars))
            grounded_state_vars.extend(groundings)
        return len(grounded_state_vars)

#     def get_deepest_subtypes(self, ancestors):
#         descendants = self.get_descendants(ancestors)
#         deepest_subtypes = [x for x in descendants if len(self.type_hierarchy[x]) == 0]
#         return deepest_subtypes


def get_children(hierarchy, parents):
    """
    :param hierarchy: dict mapping parent to children
    :parents: [parents]
    Note: a parent is one of it's own children
    """
    children = copy.copy(parents)
    for p in parents:
        children.extend(hierarchy[p])
    return sorted(list(set(children)))


def get_descendants(hierarchy, ancestors):
    """
    :param hierarchy: dict mapping parent to children
    :ancestors: [ancestors]
    Note: an ancestor is one of it's own descendants
    """
    descendants = sorted(ancestors)
    descendants_old = copy.copy(descendants)
    descendants = sorted(list(set(get_children(hierarchy, descendants))))
    while descendants != descendants_old:
        # print("descendants:")
        # print(descendants)
        # print("descendants_old:")
        # print(descendants_old)
        descendants_old = copy.copy(descendants)
        descendants = sorted(list(set(get_children(hierarchy, descendants))))
    return descendants


#
# def get_deepest_descendants(hierarchy, base):
#     descendants = get_descendants(hierarchy, base)
#     all_nodes = sorted(list(set(chain(*hierarchy))))
#     # TODO redo
#     deepest_descendants = [x for x in descendants]
# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    import sys, pprint

    domain = sys.argv[1]
    problem = sys.argv[2]
    parser = PDDL_Parser()
    print("----------------------------")
    pprint.pprint(parser.scan_tokens(domain))
    print("----------------------------")
    pprint.pprint(parser.scan_tokens(problem))
    print("----------------------------")
    parser.parse_domain(domain)
    parser.parse_problem(problem)
    print("Domain name: " + parser.domain_name)
    for act in parser.actions:
        print(act)
    print("----------------------------")
    print("Problem name: " + parser.problem_name)
    print("Objects: " + str(parser.objects))
    print("State: " + str(parser.state))
    print("Positive goals: " + str(parser.positive_goals))
    print("Negative goals: " + str(parser.negative_goals))
