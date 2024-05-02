(define (domain toy-example)
(:requirements :strips)

(:predicates
    (has-food ?ag)
    (has-sticks ?ag)
    (has-stone ?ag)
    (hungry ?ag)
    (has-axe ?ag)
)

(:action hunt
    :parameters (?ag)
    :precondition (and
        (not (has-food ?ag))
        (not (hungry ?ag))
    )
    :effect (and (has-food ?ag))
)

(:action gather
    :parameters (?ag)
    :precondition (and
        (not (has-food ?ag))
        (hungry ?ag)
    )
    :effect (and (has-food ?ag))
)

(:action get_stick
    :parameters (?ag)
    :precondition (and (not (has-sticks ?ag)))
    :effect (and (has-sticks ?ag))
)

(:action get_stone
    :parameters (?ag)
    :precondition (and (not (has-stone ?ag)))
    :effect (and (has-stone ?ag))
)

(:action eat
    :parameters (?ag)
    :precondition (and (has-food ?ag) (hungry ?ag))
    :effect (and
        (not (has-food ?ag))
        (not (hungry ?ag))
    )
)

(:action make_axe
    :parameters (?ag)
    :precondition (and
        (has-sticks ?ag)
        (has-stone ?ag)
        (not (has-axe ?ag))
    )
    :effect (and
        (not (has-sticks ?ag))
        (not (has-stone ?ag))
        (has-axe ?ag)
    )
)

(:action wait
    :parameters (?ag)
    :precondition (and (not (hungry ?ag)))
    :effect (and (hungry ?ag))
)

)
