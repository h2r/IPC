(define (domain toy-example)
(:requirements :typing :equality :fluents)

(:types object)

(:predicates
    (hungry ?ag - object)
    (has-axe ?ag - object)
)

(:functions
    (n-food ?ag - object)
    (n-sticks ?ag - object)
    (n-stone ?ag - object)
)

(:action hunt
    :parameters (?ag - object)
    :precondition (and
        (< (n-food ?ag) 5)
        (not (hungry ?ag))
    )
    :effect (and
        (increase (n-food ?ag) 1)
        (hungry ?ag)
    )
)

(:action gather
    :parameters (?ag - object)
    :precondition (and
        (< (n-food ?ag) 5)
        (hungry ?ag)
    )
    :effect (and
        (increase (n-food ?ag) 1)
    )
)

(:action get_stick
    :parameters (?ag - object)
    :precondition (and (< (n-sticks ?ag) 5))
    :effect (and (increase (n-sticks ?ag) 1))
)

(:action get_stone
    :parameters (?ag - object)
    :precondition (and (< (n-stone ?ag) 5))
    :effect (and (increase (n-stone ?ag) 1))
)

(:action eat
    :parameters (?ag - object)
    :precondition (and (> (n-food ?ag) 0))
    :effect (and
        (decrease (n-food ?ag) 1)
        (not (hungry ?ag))
    )
)

(:action make_axe
    :parameters (?ag - object)
    :precondition (and
        (> (n-sticks ?ag) 0)
        (> (n-stone ?ag) 0)
        (not (has-axe ?ag))
    )
    :effect (and
        (decrease (n-sticks ?ag) 1)
        (decrease (n-stone ?ag) 1)
        (has-axe ?ag)
    )
)

)
