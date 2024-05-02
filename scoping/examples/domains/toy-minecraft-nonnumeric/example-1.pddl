(define (problem example-1)

(:domain toy-example)

(:objects steve)

(:init
	(not (has-food steve))
	(not (has-sticks steve))
	(not (has-stone steve))
	(not (hungry steve))
	(not (has-axe steve))
)

(:goal (and
		(not (hungry steve))
		(has-axe steve)
	)
)

)
