import z3
from oo_scoping.skill_classes import EffectTypePDDL, SkillPDDL
import pdb
from typing import List
from typing import NamedTuple

# TODO incorporate EffectTypePDDLs
# TODO add pickup/dropoff


class Passenger(NamedTuple):
    name: int
    x: z3.Int
    y: z3.Int
    intaxi: z3.Bool


class Taxi(NamedTuple):
    name: int
    x: z3.Int
    y: z3.Int
    blinker: z3.Bool


# NOTE: We assume we only have 1 taxi.


def make_passengers(n=2):
    passengers = []
    for i in range(n):
        name = f"p{i}"
        intaxi = z3.Bool(f"in-taxi({name},t0)")
        x = z3.Int(f"pass-x-curr({name})")
        y = z3.Int(f"pass-y-curr({name})")
        p = Passenger(name, x, y, intaxi)
        passengers.append(p)
    return passengers


def make_taxis(n=1):
    taxis = []
    for i in range(n):
        name = f"t{i}"
        x = z3.Int(f"taxi-x({name})")
        y = z3.Int(f"taxi-y({name})")
        blinker = z3.Bool(f"taxi-blinker({name})")
        taxis.append(Taxi(name, x, y, blinker))
    return taxis


def make_movement_skills(
    passengers: List[Passenger],
    taxi: Taxi,
    action_name,
    var_affected,
    blinker=False,
    blinker_affects=None,
    effect_index=0,
):
    skills = []
    # 	Empty Taxi
    cond = z3.And(*[z3.Not(p.intaxi) for p in passengers])
    effect = [EffectTypePDDL(getattr(taxi, var_affected), effect_index)]
    if blinker:
        cond_b_off = z3.And(cond, z3.Not(taxi.blinker))
        cond_b_on = z3.And(cond, taxi.blinker)
        effect_b_off = effect
        effect_b_on = effect + [
            EffectTypePDDL(getattr(taxi, blinker_affects), effect_index)
        ]
        skills.append(SkillPDDL(cond_b_off, action_name, effect_b_off))
        skills.append(SkillPDDL(cond_b_on, action_name, effect_b_on))
    else:
        skills.append(SkillPDDL(cond, action_name, effect))
    # 	Passenger in taxi
    for i, p in enumerate(passengers):
        conds = [p.intaxi]
        for i1, p1 in enumerate(passengers):
            if i1 != i:
                conds.append(z3.Not(p1.intaxi))
        cond = z3.And(*conds)
        effect = [
            EffectTypePDDL(getattr(taxi, var_affected), 0),
            EffectTypePDDL(getattr(p, var_affected), effect_index),
        ]
        if blinker:
            cond_b_off = z3.And(cond, z3.Not(taxi.blinker))
            cond_b_on = z3.And(cond, taxi.blinker)
            effect_b_off = effect
            effect_b_on = effect + [
                EffectTypePDDL(getattr(taxi, blinker_affects), effect_index),
                EffectTypePDDL(getattr(p, blinker_affects), effect_index),
            ]
            skills.append(SkillPDDL(cond_b_off, action_name, effect_b_off))
            skills.append(SkillPDDL(cond_b_on, action_name, effect_b_on))
        else:
            skills.append(SkillPDDL(cond, action_name, effect))
    return skills


def make_initial_condition(passengers: List[Passenger], taxi: Taxi):
    cond = z3.And(*([z3.Not(p.intaxi) for p in passengers] + [taxi.blinker]))
    return cond


def make_goal(passenger, x=None, y=None):
    conds = [z3.Not(passenger.intaxi)]
    if x is not None:
        conds.append(passenger.x == x)
    if y is not None:
        conds.append(passenger.y == y)
    return z3.And(conds)


def make_pickup_dropoff_skills(ps: List[Passenger], t: Taxi):
    skills = []
    for p_id, p in enumerate(ps):
        # Taxi and passenger colocated
        conds = [p.x == t.x, p.y == t.y]
        # Taxi is empty
        for p_id2, p2 in enumerate(ps):
            conds.append(z3.Not(p.intaxi))
        prec = z3.And(*conds)
        effect = EffectTypePDDL(p.intaxi, True)
        pickup_skill = SkillPDDL(prec, "pickup()", effect)
        skills.append(pickup_skill)
        dropoff_skill = SkillPDDL(
            p.intaxi, "dropoff()", EffectTypePDDL(p.intaxi, False)
        )
        skills.append(dropoff_skill)
    return skills


def make_state_constraints(passengers: List[Passenger]):
    conds = []
    for i in range(len(passengers)):
        for j in range(len(passengers)):
            if i != j:
                conds.append(z3.Not(z3.And(passengers[i].intaxi, passengers[j].intaxi)))
    return z3.And(*conds)


def prepare_taxi_domain(n_passegners=2, blinker=False, goal=(6, 8)):
    """Returns skills, init_cond, goal"""
    # if blinker:
    # 	blinker_affects = {"move_north()": "x", "move_south()": "x", "move_east()":"y", "move_west()":"y"}
    passengers = make_passengers(n_passegners)
    taxi = make_taxis(1)[0]
    move_north = make_movement_skills(
        passengers, taxi, "move_north()", "y", blinker, "x", effect_index=1
    )
    move_south = make_movement_skills(
        passengers, taxi, "move_south()", "y", False, "x", effect_index=-1
    )
    move_east = make_movement_skills(
        passengers, taxi, "move_east()", "x", False, "y", effect_index=1
    )
    move_west = make_movement_skills(
        passengers, taxi, "move_west()", "x", False, "y", effect_index=-1
    )
    pickup_dropoff = make_pickup_dropoff_skills(passengers, taxi)
    skills = move_north + move_south + move_east + move_west + pickup_dropoff
    start_condition = make_initial_condition(passengers, taxi)
    goal = make_goal(passengers[0], x=goal[0], y=goal[1])
    pvars = [taxi.y] + [p.y for p in passengers] + [p.intaxi for p in passengers]
    state_constraints = make_state_constraints(passengers)
    return goal, skills, start_condition, pvars, state_constraints


# return pas


if __name__ == "__main__":
    passengers = make_passengers(2)
    taxi = make_taxis(1)[0]
    move_north = make_movement_skills(passengers, taxi, "move_north()", "y")
    move_south = make_movement_skills(passengers, taxi, "move_south()", "y")

# for s in move_north: print(s)
