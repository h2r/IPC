(define (problem TAXINUMERIC-2)
(:domain universal_multipasstaxi)
(:objects
	curly - passenger
    t0 - taxi
)
(:init
    (= (taxi-x t0) 0)
    (= (taxi-y t0) 0)
    (not (in-taxi curly t0))
;    (not (in-taxi smoov t0))
    (= (pass-x curly) 3329)
    (= (pass-y curly) 3614)
;    (= (pass-x smoov) 3459)
;    (= (pass-y smoov) 1462)
)
(:goal (and
    (= (pass-y curly) 10000)
	(= (pass-x curly) 9000)
    (not (in-taxi curly t0))
    
	
    )
    )

)