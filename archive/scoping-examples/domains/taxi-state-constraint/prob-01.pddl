(define (problem UNITAXINUMERIC-1)
(:domain universal_multipasstaxi)
(:objects
	curly - passenger
    t0 - taxi
)
(:init
    (= (taxi-x t0) 0)
    (= (taxi-y t0) 0)
    (= (pass-x curly) 3)
    (= (pass-y curly) 3)
)
(:goal (and
    (= (pass-y curly) 10000)
	(= (pass-x curly) 9000)
    (not (in-taxi curly t0))	
    )
    )
)