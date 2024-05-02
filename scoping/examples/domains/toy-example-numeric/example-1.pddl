(define (problem example-1)

(:domain toy-example)

(:objects steve - object)

(:init
	(= (n-food steve) 0)
	(= (n-sticks steve) 0)
	(= (n-stone steve) 0)
	(not (hungry steve))
	(not (has-axe steve))
)

(:goal (and
		(not (hungry steve))
		(has-axe steve)
	)
)

)
