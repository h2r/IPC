; Note (<redacted>): Replaced 'at' with 'located-at' so ENHSP works

(define (domain Depot)
(:requirements :typing :fluents)
(:types place locatable - object
	depot distributor - place
        truck hoist surface - locatable
        pallet crate - surface)

(:predicates (located-at ?x - locatable ?y - place) 
             (on ?x - crate ?y - surface)
             (in ?x - crate ?y - truck)
             (lifting ?x - hoist ?y - crate)
             (available ?x - hoist)
             (clear ?x - surface)
)

(:functions 
	(load_limit ?t - truck) 
	(current_load ?t - truck) 
	(weight ?c - crate)
	(fuel-cost)
)
	
(:action drive
:parameters (?x - truck ?y - place ?z - place) 
:precondition (and (located-at ?x ?y))
:effect (and (not (located-at ?x ?y)) (located-at ?x ?z)
		(increase (fuel-cost) 10)))

(:action lift
:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
:precondition (and (located-at ?x ?p) (available ?x) (located-at ?y ?p) (on ?y ?z) (clear ?y))
:effect (and (not (located-at ?y ?p)) (lifting ?x ?y) (not (clear ?y)) (not (available ?x)) 
             (clear ?z) (not (on ?y ?z)) (increase (fuel-cost) 1)))

(:action drop 
:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
:precondition (and (located-at ?x ?p) (located-at ?z ?p) (clear ?z) (lifting ?x ?y))
:effect (and (available ?x) (not (lifting ?x ?y)) (located-at ?y ?p) (not (clear ?z)) (clear ?y)
		(on ?y ?z)))

(:action load
:parameters (?x - hoist ?y - crate ?z - truck ?p - place)
:precondition (and (located-at ?x ?p) (located-at ?z ?p) (lifting ?x ?y)
		(<= (+ (current_load ?z) (weight ?y)) (load_limit ?z)))
:effect (and (not (lifting ?x ?y)) (in ?y ?z) (available ?x)
		(increase (current_load ?z) (weight ?y))))

(:action unload 
:parameters (?x - hoist ?y - crate ?z - truck ?p - place)
:precondition (and (located-at ?x ?p) (located-at ?z ?p) (available ?x) (in ?y ?z))
:effect (and (not (in ?y ?z)) (not (available ?x)) (lifting ?x ?y)
		(decrease (current_load ?z) (weight ?y))))

)
