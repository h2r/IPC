(define (problem ZTRAVEL-1-2)
(:domain ipc_composite)
(:objects
	plane1 - aircraft
	person1 - person
	person2 - person
	person3 - person
	city0 - city
	city1 - city
	city2 - city
	)
(:init
	(located-zeno plane1 city0)
	(= (capacity plane1) 6000)
	(= (fuel-zeno plane1) 4000)
	(= (slow-burn plane1) 4)
	(= (fast-burn plane1) 15)
	(= (onboard plane1) 0)
	(= (zoom-limit plane1) 8)
	(located-zeno person1 city0)
	(located-zeno person2 city0)
	(located-zeno person3 city1)
	(= (distance city0 city0) 0)
	(= (distance city0 city1) 678)
	(= (distance city0 city2) 775)
	(= (distance city1 city0) 678)
	(= (distance city1 city1) 0)
	(= (distance city1 city2) 810)
	(= (distance city2 city0) 775)
	(= (distance city2 city1) 810)
	(= (distance city2 city2) 0)
	(= (total-fuel-used) 0)

)
(:goal (and	
	(located-zeno person1 city2)
	(located-zeno person2 city1)
	(located-zeno person3 city2)
	)
)
)
