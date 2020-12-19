(define (problem TAXINUMERIC-2)
(:domain multipasstaxi)
(:objects
	curly smoov - passenger

	; curly smoov edison isbell cornelius nero kaelbling perez levine abbeel - passenger
	; curly - passenger
    t0 - taxi
)
(:init
    (= (taxi-x t0) 0)
    (= (taxi-y t0) 0)
    (not (in-taxi curly t0))
    (not (in-taxi smoov t0))
    ; (not (in-taxi edison t0))
    ; (not (in-taxi isbell t0))
    ; (not (in-taxi cornelius t0))
    ; (not (in-taxi nero t0))
    ; (not (in-taxi kaelbling t0))
    ; (not (in-taxi perez t0))
    ; (not (in-taxi levine t0))
    ; (not (in-taxi abbeel t0))
    (= (pass-x curly) 15)
    (= (pass-y curly) 23)
    (= (pass-x smoov) 2)
    (= (pass-y smoov) 1)
    ; (= (pass-x edison) 1)
    ; (= (pass-y edison) 2)
    ; (= (pass-x isbell) 1)
    ; (= (pass-y isbell) 1)
    ; (= (pass-x cornelius) 2)
    ; (= (pass-y cornelius) 2)
    ; (= (pass-x nero) 0)
    ; (= (pass-y nero) 1)
    ; (= (pass-x kaelbling) 1)
    ; (= (pass-y kaelbling) 0)
    ; (= (pass-x perez) 0)
    ; (= (pass-y perez) 2)
    ; (= (pass-x levine) 2)
    ; (= (pass-y levine) 0)
    ; (= (pass-x abbeel) 0)
    ; (= (pass-y abbeel) 0)
    (= (passenger-count t0) 0)
)
(:goal (and
    (= (pass-y curly) 1050)
	(= (pass-x curly) 830)
    (not (in-taxi curly t0))
	
    )
    )
;(:metric  minimize (total-fuel-used) )

)