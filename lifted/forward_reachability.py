from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, NewType, Optional, Iterable, TypeVar, Set

import numpy as np
import pandas as pd

@dataclass(frozen=True, order=True)
class Action:
    name: str
    variables: List[str]
    precondition: List[Tuple[str]]
    effect: List[Tuple[str]]

def build_fact_table(fact_list: List[Tuple[str]], predicate: Tuple[str]) -> pd.DataFrame:
    """Build a pandas DataFrame containing all facts in fact_list for a given predicate"""
    facts = [fact[1:] for fact in fact_list if fact[0] == predicate]
    assert np.allclose(list(map(len, facts)), len(facts[0]))
    columns = [f'v{i}' for i in range(len(facts[0]))]
    data = pd.DataFrame(facts, columns=columns)
    data.name = predicate
    return data

def initialize_fact_tables(fact_list: List[Tuple[str]]) -> Dict[str, pd.DataFrame]:
    """Build a dictionary from predicate names to fact table DataFrame objects"""
    predicate_names = sorted(list(set([fact[0] for fact in fact_list])))
    return {name:build_fact_table(fact_list, name) for name in predicate_names}

def join_tables_using_predicate_list(tables:Dict[str, pd.DataFrame], predicate_list:List[Tuple[str]]) -> pd.DataFrame:
    """Relabel table columns according to predicate list, then join on any matching variable names"""
    remapped_tables = []
    for predicate in predicate_list:
        table_name, *params = predicate
        table = tables[table_name]
        column_names = dict(zip(table.columns, params))
        table = table.rename(columns=column_names)
        remapped_tables.append(table)
    result, *rest = remapped_tables
    for table in rest:
        try:
            result = result.merge(table)
        except pd.errors.MergeError:
            result = result.join(table, how='cross')
    return result

def select_positive_groundings(groundings_table:pd.DataFrame, predicate_list:Tuple[str]) -> Dict[str, pd.DataFrame]:
    """For each positive predicate in predicate_list, extract relevant cols from groundings_table"""
    results = {}
    for predicate in predicate_list:
        if predicate[0] == 'not':
            continue
        table_name, *parameters = predicate
        # TODO: this overwrites if table_name appears twice in same predicate_list. it shouldn't!
        results[table_name] = groundings_table[parameters]
    return results

def extend_fact_tables(tables: Dict[str, pd.DataFrame], updates: Dict[str, pd.DataFrame]) -> bool:
    """Update tables with any new items in `updates`, and return whether new items were added."""
    did_extend = False
    for table_name, new_facts in updates.items():
        table = tables[table_name]
        column_mapping = {new: orig for new, orig in zip(new_facts.columns, table.columns)}
        new_facts = new_facts.rename(columns=column_mapping)
        new_table = pd.concat((table, new_facts)).drop_duplicates()
        if len(new_table) > len(table):
            did_extend = True
        tables[table_name] = new_table
        return did_extend

def generate_next_fact_layer(tables: Dict[str, pd.DataFrame], actions: List[Action]) -> Dict[str, pd.DataFrame]:
    """Apply each action to fact layer represented by `tables` and add any new (positive) facts"""
    did_extend = False
    for action in actions:
        result = join_tables_using_predicate_list(tables, action.precondition)
        updates = select_positive_groundings(result, action.effect)
        did_extend = did_extend or extend_fact_tables(tables, updates)
    return did_extend


def main():
    init_state = [
        ('at', 'obj1', 'l1'),
        ('at', 'obj2', 'l1'),
        ('at', 'obj3', 'l3'),
        ('at', 'obj4', 'l2'),
        ('path', 'l1', 'l2'),
        ('path', 'l1', 'l3'),
        ('path', 'l2', 'l3'),
        ('path', 'l3', 'l4'),
        ('alive', 'agent')
    ]

    a1 = Action(
        name='a1',
        variables = ['x', 'y', 'w', 'z', 'ag'],
        precondition = [
            ('at', 'x', 'y'),
            ('path', 'y', 'w'),
            ('path', 'w', 'z'),
            ('alive', 'ag')
        ],
        effect = [
            ('at', 'x', 'z'),
            ('not', ('at', 'x', 'y'))
        ],
    )
    a2 = Action(
        name='a2',
        variables = ['x', 'y', 'w', 'z', 'ag'],
        precondition = [
            ('at', 'x', 'z'),
            ('path', 'y', 'w'),
            ('path', 'w', 'z'),
            ('alive', 'ag')
        ],
        effect = [
            ('at', 'x', 'y'),
            ('not', ('at', 'x', 'z'))
        ],
    )
    actions = [a1]
    # actions = [a1, a2]
    tables = initialize_fact_tables(init_state)
    while generate_next_fact_layer(tables, actions):
        pass

    tables
