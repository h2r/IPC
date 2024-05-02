(define (problem TAXINUMERIC-03)
(:domain universal_multipasstaxi)
(:objects
	curly smoov edison - passenger
    t0 - taxi
)
(:init
    (= (taxi-x t0) 0)
    (= (taxi-y t0) 0)

    (not (in-taxi curly t0))
    (not (in-taxi smoov t0))
    (not (in-taxi edison t0))

    (= (pass-x curly) 3329)
    (= (pass-y curly) 3615)

    (= (pass-x smoov) 3459)
    (= (pass-y smoov) 1262)

    (= (pass-x edison) 1291)
    (= (pass-y edison) 5037)

)
(:goal (and
    (= (pass-y curly) 10000)
	(= (pass-x curly) 9000)
    (not (in-taxi curly t0))
	
    )
)

)