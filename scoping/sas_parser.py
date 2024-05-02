from __future__ import annotations
import re, os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, NewType, Optional, Iterable, TypeVar, Set
import itertools

import z3
from oo_scoping.downward_translate import sas_tasks as fd

from oo_scoping.skill_classes import SkillPDDL, EffectTypePDDL

SasVarVal = NewType("SasVarVal", str)

@dataclass(frozen=True, order=True)
class SasVar:
    nm: str
    axiom_layer: int
    range: int
    vals: Tuple[SasVarVal,...]

    @staticmethod
    def from_regex_tuple(m: Tuple[str, str, str, str]) -> SasVar:
        return SasVar(m[0], int(m[1]), int(m[2]), SasVar.split_values(m[3]))

    @staticmethod
    def split_values(s: str) -> Tuple[SasVarVal, ...]:
        return tuple([SasVarVal(x) for x in s.split("\n")])

    def lookup(self, value: SasVarVal) -> int:
        if value is None:
            return -1
        return self.vals.index(value)

    def get_var_val_pair(self, i: int) -> SasVarValPair:
        return SasVarValPair(self, i)


@dataclass(frozen=True, order=True)
class SasVarValPair:
    var: SasVar
    # val: SasVarVal
    val: int

    @property
    def val_nm(self) -> SasVarVal:
        return self.var.vals[self.val]

@dataclass(frozen=True, order=True)
class SasEffect:
    """
    Sas files distinguish between the condition on non-affected vars,
    and the condition on the affected var.
    Note that the var in condition_affected_var
    and the var in result must be the same
    It is maybe wasteful to keep it separate
    """
    cond: Tuple[SasVarValPair,...]
    var: int
    affected_var: SasVar
    pre: Optional[int]
    post: int

    @property
    def affected_var_condition_pair(self) -> Optional[SasVarValPair]:
        if self.pre is None:
            return None
        else:
            return SasVarValPair(self.affected_var, self.pre)

    @property
    def result_var_val_pair(self) -> SasVarValPair:
        return SasVarValPair(self.affected_var, self.post)

    @property
    def full_condition(self) -> Tuple[SasVarValPair,...]:
        """
        Combination of non-affected var condition
        and affected var condition
        MF: Should we sort by var? Nah.
        """
        if self.pre is None:
            return self.cond
        else:
            return self.cond + [self.affected_var_condition_pair]


@dataclass(frozen=True, order=True)
class SasOperator:
    nm: str
    prevail: Tuple[Tuple,...]
    effects: Tuple[SasEffect,...]
    cost: int = 1 #Default to 1, in case we don't use action cost


"""An axiom is basically an effect that is applied every timestep, if applicable"""
class SasAxiom(SasEffect):
    pass

@dataclass(frozen=True, order=True)
class SasMutex:
    facts: Tuple[SasVarValPair,...]


@dataclass(frozen=True, order=True)
class SasPartialState:
    """It would be nice to enforce uniqueness of keys"""
    var_value_pairs: Tuple[SasVarValPair,...]

    def __getitem__(self, key: SasVar) -> SasVarVal:
        candidates = [x.val for x in self.var_value_pairs if x.var == key]
        return candidates[0]


class SasState(SasPartialState):
    """It would be nice to enforce full specification of vars"""
    pass


class SasParser:
    """
    Parse sas planning files into python datastructures.
    There are three sets of methods:

    1. Parsing functions. One for each section, and one to string them together.
    2. Helper functions
    3. File writing functions
    """
    # Regex patterns used in parsing
    pattern_var_val_pair = "(?P<var_num>\d+) (?P<val_num>\d+)"
    pattern_operator = "begin_operator\n(?P<nm>.+)\n(?P<n_prevail>.+)\n(?P<prevail>(\d+ \d+\n)*?)(?P<n_effects>\d+)\n(?P<effects>[\s\S]+?)\n(?P<cost>\d+)\nend_operator"

    # Type annotations for parsed values
    s_sas: str
    sas_vars: Tuple[SasVar, ...]
    sas_operators: Tuple[SasOperator,...]
    sas_mutexes: Tuple[SasMutex,...]
    sas_axioms: Tuple[SasAxiom,...]
    initial_state: SasState
    goal: SasPartialState
    _z3vars: Dict[SasVar, z3.Int]

    def __init__(self, s_sas: Optional[str] = None, pth: Optional[str] = None) -> None:
        """
        Specify either s_sas or pth.

        """
        if s_sas is not None:
            self.s_sas = s_sas
        elif pth is not None:
            with open(pth, "r") as f:
                self.s_sas = f.read()
        else:
            raise ValueError(
                "Please specify either s_sas or the pth of the sas file when creating a SasParser."
            )
        self._z3vars: Dict[SasVar, z3.Int] = dict()

    # Parsing functions
    def parse(self):
        """Do entire parsing"""
        self.parse_version()
        self.parse_metric()
        self.parse_vars()
        self.parse_mutex()
        self.parse_initial_state()
        self.parse_goal()
        self.parse_axioms()
        self.parse_operators()


    def parse_version(self) -> str:
        pattern_version = "begin_version\n(?P<version>\d+)\nend_version"
        versions = re.findall(pattern_version, self.s_sas)
        if len(versions) != 1:
            raise ValueError(f"File specifies {len(versions)} versions. It should specify 1.")
        self.version = versions[0][0]
        return self.version

    def parse_metric(self) -> int:
        """The metric should be 0 or 1"""
        pattern_metric = "begin_metric\n(?P<version>\d+)\nend_metric"
        metrics = re.findall(pattern_metric, self.s_sas)
        if len(metrics) != 1:
            raise ValueError(f"File specifies {len(metrics)} metrics. It should specify 1.")
        self.metric = int(metrics[0][0])
        return self.metric


    def parse_vars(self) -> Tuple[SasVar, ...]:
        # Regex notes: [\s\S] matches all characters, including newlines
        # Putting ? after a quantifier makes it non-greedy, so that it stops as soon as it can
        var_pattern = "begin_variable\n(?P<nm>.*)\n(?P<axiom_layer>.*)\n(?P<range>.*)\n(?P<vals>[\s\S]*?)\nend_variable"
        # TODO use finditer instead, so that we can use named capture groups.
        # Less error prone, more clear
        matches = re.findall(var_pattern, self.s_sas)
        sas_vars: List[SasVar] = []
        for m in matches:
            sas_vars.append(SasVar.from_regex_tuple(m))
        sas_vars = tuple(sas_vars)
        self.sas_vars = sas_vars
        return sas_vars

    def parse_mutex(self) -> Tuple[SasMutex,...]:
        """
        Must be run after parse_vars
        """
        mutex_pattern = (
            "begin_mutex_group\n(?P<n_facts>\d+)\n(?P<facts>[\s\S]*?)\nend_mutex_group"
        )
        mutexes_lst: List[SasMutex] = []

        for mutex_group_match in re.finditer(mutex_pattern, self.s_sas):
            facts_lst: List[SasVarValPair] = []
            facts_strs = mutex_group_match.group("facts").splitlines()
            for fs in facts_strs:
                fact = self.get_sas_var_val_pair_from_str(fs)
                facts_lst.append(fact)
            mutex = SasMutex(tuple(facts_lst))
            mutexes_lst.append(mutex)

        mutexes = tuple(mutexes_lst)
        self.mutexes = mutexes
        return mutexes

    def parse_initial_state(self) -> SasState:
        pattern_initial_state = "begin_state\n(?P<state>[\s\S]+?)\nend_state"
        s_state = re.search(pattern_initial_state, self.s_sas).group("state")
        vals = s_state.splitlines()
        assert len(vals) == len(self.sas_vars)
        var_val_pairs = tuple([self.sas_vars[i].get_var_val_pair(int(vals[i])) for i in range(len(vals))])
        self.initial_state = SasState(var_val_pairs)
        return self.initial_state

    def parse_goal(self) -> SasPartialState:
        pattern_goal = "begin_goal\n(?P<n>\d+)\n(?P<var_vals>[\s\S]+?)\nend_goal"
        s_goal = re.search(pattern_goal, self.s_sas).group("var_vals")
        var_val_strs = s_goal.splitlines()
        var_val_pairs = tuple([self.get_sas_var_val_pair_from_str(s) for s in var_val_strs])
        self.goal = SasPartialState(var_val_pairs)
        return self.goal

    def parse_axioms(self) -> Tuple[SasAxiom,...]:
        pattern_head = "(?P<var_num>\d+) (?P<val_num_old>\d+) (?P<val_num_new>\d+)"
        pattern_axiom = f"begin_rule\n(?P<n>\d+)\n(?P<conditions>[\s\S]+?)\n{pattern_head}\nend_rule"
        axioms_lst: List[SasAxiom] = []
        for m_axiom in re.finditer(pattern_axiom, self.s_sas):
            conds_strs = m_axiom.group("conditions").splitlines()
            conds = [self.get_sas_var_val_pair_from_str(c) for c in conds_strs]
            i_affected_var = int(m_axiom.group("var_num"))
            i_val_old = int(m_axiom.group("val_num_old"))
            i_val_new = int(m_axiom.group("val_num_new"))
            affected_var = self.sas_vars[i_affected_var]
            axioms_lst.append(
                SasAxiom(
                    condition=tuple(conds),
                    affected_var=affected_var,
                    affected_var_condition=affected_var.vals[i_val_old],
                    result_val=affected_var.vals[i_val_new],
                )
            )
        self.axioms = tuple(axioms_lst)
        return self.axioms


    def parse_operators(self) -> Tuple[SasOperator,...]:
        pattern_operator_count = "end_goal\n(?P<n>\d+)\nbegin_operator"
        n_operators = int(re.search(pattern_operator_count, self.s_sas).group("n"))

        operators: List[SasOperator] = []
        for m in re.finditer(SasParser.pattern_operator, self.s_sas):
            prevail_lines = m.group("prevail").splitlines()
            prevail = tuple([self.get_sas_var_val_pair_from_str(x) for x in prevail_lines])
            effect_lines = m.group("effects").splitlines()
            effects = tuple([self.parse_effect(x) for x in effect_lines])
            operators.append(SasOperator(
                nm = m.group("nm"),
                prevail = prevail,
                effects=effects,
                cost=m.group("cost")

            ))
        if len(operators) != n_operators:
            raise ValueError(f"The sas file claims to have {n_operators} operators, but we found {len(operators)}")
        self.operators = tuple(operators)
        return self.operators

    def parse_effect(self, s: str) -> SasEffect:
        # n_cond = int(re.match("(?P<n_cond>\d+).+", s).group("n_cond"))
        s_split = s.split(" ")
        n_cond = int(s_split[0])

        conds: List[SasVarValPair] = []
        i_conds_start = 1
        for i_pair in range(n_cond):
            num_var = int(s_split[i_conds_start + i_pair*2])
            num_val = int(s_split[i_conds_start + (i_pair*2) + 1])
            conds.append(self.get_sas_var_val_pair_from_ints(num_var, num_val))

        i_conds_end = i_conds_start + n_cond * 2
        num_var_affected = int(s_split[i_conds_end])
        var_affected = self.sas_vars[num_var_affected]
        num_val_cond = int(s_split[i_conds_end + 1])
        # -1 means that there is no condition on the affected var
        if num_val_cond == -1:
            val_cond = None
        else:
            val_cond = var_affected.vals[num_val_cond]

        num_val_result = int(s_split[i_conds_end + 2])
        if i_conds_end + 2 != len(s_split) - 1:
            raise ValueError("We miscounted")

        return SasEffect(
            cond=tuple(conds),
            var=num_var_affected,
            affected_var=var_affected,
            pre=val_cond,
            post=num_val_result
        )

    # Helper functions
    ## Getting SasVarValPairs
    def get_sas_var_val_pair_from_ints(
        self, var_num: int, val_num: int
    ) -> SasVarValPair:
        var0 = self.sas_vars[var_num]
        # val0 = var0.vals[val_num]
        return SasVarValPair(var0, val_num)

    @staticmethod
    def get_var_val_nums_from_str(s: str) -> Tuple[int, int]:
        m = re.match(SasParser.pattern_var_val_pair, s)
        if m is None:
            raise ValueError(f"The string is not a pair of ints:\n{s}")
        var_num = int(m.group("var_num"))
        val_num = int(m.group("val_num"))
        return var_num, val_num

    def get_sas_var_val_pair_from_str(self, s: str) -> SasVarValPair:
        var_num, val_num = SasParser.get_var_val_nums_from_str(s)
        # return self.get_sas_var_val_pair_from_ints(var_num, val_num)
        return (var_num, val_num)


    def var_val_pair2ints(self, p: SasVarValPair) -> Tuple[int, int]:
        """Returns the pair of ints a sas file would use to represent p"""
        i_var = self.sas_vars.index(p.var)
        # i_val = p.var.vals.index(p.val)
        return (i_var, p.val)

    def var_val_pair2sas_str(self, p: SasVarValPair) -> str:
        return " ".join(map(str, self.var_val_pair2ints(p)))

    # Writing back to SAS
    def generate_sas(self) -> str:
        pieces: List[str] = [
            self.generate_version_and_metric_sections(),
            self.generate_variables_section(),
            self.generate_mutexes_section(),
            self.generate_initial_state_section(),
            self.generate_goal_section(),
            self.generate_operators_section(),
            self.generate_axioms_section()
        ]
        return "\n".join(pieces) + "\n"


    def generate_version_and_metric_sections(self) -> str:
        return f"begin_version\n{self.version}\nend_version\nbegin_metric\n{self.metric}\nend_metric"

    ## Variables
    def generate_variables_section(self) -> str:
        pieces: List[str] = [str(len(self.sas_vars))]
        for v in self.sas_vars:
            pieces.append(self.generate_variable_str(v))
        return "\n".join(pieces)

    def generate_variable_str(self, v: SasVar) -> str:
        pieces: List[str] = [
            "begin_variable",
            v.nm,
            str(v.axiom_layer),
            str(v.range)
        ]
        for val in v.vals:
            pieces.append(val)
        pieces.append("end_variable")
        return "\n".join(pieces)

    ## Mutexes
    def generate_mutexes_section(self) -> str:
        pieces: List[str] = [str(len(self.mutexes))]
        pieces.extend([self.generate_mutex_str(m) for m in self.mutexes])
        return "\n".join(pieces)

    def generate_mutex_str(self, m: SasMutex) -> str:
        pieces: List[str] = ["begin_mutex_group"]
        pieces.append(str(len(m.facts)))
        # raise NotImplementedError
        for f in m.facts:
            i_var = self.sas_vars.index(f.var)
            # i_val = f.var.vals.index(f.val)
            pieces.append(f"{i_var} {f.val}")
        pieces.append("end_mutex_group")
        return "\n".join(pieces)

    ## Initial State
    def generate_initial_state_section(self) -> str:
        pieces: List[str] = ["begin_state"]
        # pieces.extend([self.var_val_pair2sas_str(p) for p in self.initial_state.var_value_pairs])
        pieces.extend([str(p.val) for p in self.initial_state.var_value_pairs])
        pieces.append("end_state")
        return "\n".join(pieces)

    ## Goals
    def generate_goal_section(self) -> str:
        pieces: List[str] = ["begin_goal", str(len(self.goal.var_value_pairs))]
        pieces.extend([self.var_val_pair2sas_str(p) for p in self.goal.var_value_pairs])
        pieces.append("end_goal")
        return "\n".join(pieces)

    def generate_operators_section(self) -> str:
        pieces: List[str] = [str(len(self.operators))]
        pieces.extend([self.generate_operator_str(o) for o in self.operators])
        return "\n".join(pieces)

    ## Operators
    def generate_operator_str(self, o: SasOperator) -> str:
        pieces: List[str] = [
            "begin_operator",
            o.nm,
            str(len(o.prevail))
        ]
        # Add prevail conditions
        pieces.extend([self.var_val_pair2sas_str(p) for p in o.prevail])

        # Add effects
        pieces.append(str(len(o.effects)))
        pieces.extend([self.effect2sas_str(e) for e in o.effects])

        # Add cost
        pieces.append(str(o.cost))
        pieces.append("end_operator")
        return "\n".join(pieces)


    def effect2sas_str(self, e: SasEffect) -> str:
        pieces: List[str] = [str(len(e.cond))]
        # Effect conditions
        pieces.extend([self.var_val_pair2sas_str(p) for p in e.cond])
        # Affected var
        pieces.append(str(self.sas_vars.index(e.affected_var)))
        # Affected var condition
        if e.pre is None:
            pieces.append("-1")
        else:
            pieces.append(str(e.affected_var.vals.index(e.pre)))
        # Result value
        pieces.append(str(e.post))
        return " ".join(pieces)

    ## Axioms
    def generate_axioms_section(self) -> str:
        pieces: List[str] = [str(len(self.axioms))]
        pieces.extend([self.axiom2sas_str(a) for a in self.axioms])
        return "\n".join(pieces)

    def axiom2sas_str(self, a: SasAxiom) -> str:
        pieces: List[str] = ["begin_rule", str(len(a.cond))]
        # Conditions
        pieces.extend([self.var_val_pair2sas_str(p) for p in a.cond])
        # Affected var
        i_var = self.sas_vars.index(a.affected_var)
        # i_val_old = a.affected_var.vals.index(a.affected_var_condition)
        # i_val_new = a.affected_var.vals.index(a.result_val)
        pieces.append(f"{i_var} {a.pre} {a.post}")
        pieces.append("end_rule")
        return "\n".join(pieces)

    # Check parse
    def check_parse(self) -> bool:
        return self.generate_sas() == self.s_sas

    # Converting to scopeable representation
    def to_fd(self) -> fd.SASTask:
        # Variables
        ranges = [v.range for v in self.sas_vars]
        axiom_layers = [v.axiom_layer for v in self.sas_vars]
        value_names = [v.nm for v in self.sas_vars]
        variables = fd.SASVariables(ranges, axiom_layers, value_names)

        # Mutexes
        mutexes = [fd.SASMutexGroup(m.facts) for m in self.mutexes]

        # Init
        init = fd.SASInit([p.val for p in self.initial_state.var_value_pairs])

        # Goal
        goal = fd.SASGoal(self.goal.var_value_pairs)

        # Operators
        operators = []
        for op in self.operators:
            pre_post = []
            for e in op.effects:
                pre_as_int = self.sas_vars[e.var].lookup(e.pre)
                pre_post.append( (e.var, pre_as_int, e.post, e.cond) )
            operators.append(fd.SASOperator(op.nm, op.prevail, pre_post, op.cost))

        # Axioms
        if self.axioms:
            raise NotImplementedError("Axioms not implemented")
        axioms = []

        # Metric
        metric = self.metric

        sas_task = fd.SASTask(variables, mutexes, init, goal, operators, axioms, metric)
        return sas_task


def test():
    repo_root = "../../.."
    # pth = "../../../gripper-painting.sas"
    pth_sas_dir = f"{repo_root}/generated_sas"
    os.makedirs(pth_sas_dir, exist_ok=True)
    pth_sas_in = f"{pth_sas_dir}/gripper-painting.sas"
    cmd_s = f"python {repo_root}/oo_scoping/downward_translate/translate_and_scope.py {repo_root}/examples/gripper-painting-domain/domain.pddl {repo_root}/examples/gripper-painting-domain/prob04.pddl --sas-file {pth_sas_in} --scope True"
    os.system(cmd_s)

    parser = SasParser(pth=pth_sas_in)
    parser.parse()

    def add_pre_extension(s: str, s_suffix: str) -> str:
        s_split = s.split(".")
        s_split = s_split[:-1] + [s_suffix] + s_split[-1:]
        return ".".join(s_split)

    s_sas_out = parser.generate_sas()
    with open(add_pre_extension(pth_sas_in, "regen"), "w") as f:
        f.write(s_sas_out)

    print(s_sas_out == parser.s_sas)
