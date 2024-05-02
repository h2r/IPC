(define (problem ZTRAVEL-1-2)
(:domain zenotravel)
(:objects
	plane1 - aircraft
	person1 - person
	city0 - city
	city2 - city
	)
(:init
	(located plane1 city0)
	(= (capacity plane1) 6000)
	(= (fuel plane1) 4000)
	(= (slow-burn plane1) 4)
	(= (fast-burn plane1) 15)
	(= (onboard plane1) 0)
	(= (zoom-limit plane1) 8)
	(located person1 city0)
	(= (distance city0 city0) 0)
	(= (distance city0 city2) 775)
	(= (distance city2 city0) 775)
	(= (distance city2 city2) 0)
	(= (total-fuel-used) 0)

)
(:goal (and	
	(located person1 city2)
	))
;(:metric  minimize (total-fuel-used) )

)