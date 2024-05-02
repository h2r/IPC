(define (problem composite_prob_1)
(:domain ipc_composite)
(:objects
	; Depot
	depot0 - depot
	distributor0 distributor1 - distributor
	truck0 truck1 - truck-depot
	pallet0 pallet1 pallet2 - pallet
	crate0 crate1 crate2 crate3 crate4 crate5 - crate
	hoist0 hoist1 hoist2 - hoist

	; Satellite
;	satellite0 - satellite
;	instrument0 instrument1 - instrument
;	infrared0 infrared1 image2 - satellite-mode
;	groundstation1 star0 groundstation2 planet3 planet4 phenomenon5 phenomenon6 star7 - direction

	; Driverlog
;	driver1 - driver
;	driver2 - driver
;	truckd1 - truck-driverlog
;	truckd2 - truck-driverlog
;	package1 - obj
;	package2 - obj
;	package3 - obj
;	s0 - location
;	s1 - location
;	s2 - location
;	p0-1 - location
;	p0-2 - location
;	p1-0 - location
;	p2-1 - location

	; ZenoTravel
;	plane1 - aircraft
;	person1 person2 person3 - person
;	city0 city1 city2 - city
)

(:init
	; Depot
	(located-at-depot pallet0 depot0)
	(clear crate1)
	(located-at-depot pallet1 distributor0)
	(clear crate4)
	(located-at-depot pallet2 distributor1)
	(clear crate5)
	(located-at-depot truck0 depot0)
	(= (current_load truck0) 0)
	(= (load_limit truck0) 457)
	(located-at-depot truck1 distributor0)
	(= (current_load truck1) 0)
	(= (load_limit truck1) 331)
	(located-at-depot hoist0 depot0)
	(available-depot hoist0)
	(located-at-depot hoist1 distributor0)
	(available-depot hoist1)
	(located-at-depot hoist2 distributor1)
	(available-depot hoist2)
	(located-at-depot crate0 distributor0)
	(on-depot crate0 pallet1)
	(= (weight crate0) 99)
	(located-at-depot crate1 depot0)
	(on-depot crate1 pallet0)
	(= (weight crate1) 89)
	(located-at-depot crate2 distributor1)
	(on-depot crate2 pallet2)
	(= (weight crate2) 67)
	(located-at-depot crate3 distributor0)
	(on-depot crate3 crate0)
	(= (weight crate3) 81)
	(located-at-depot crate4 distributor0)
	(on-depot crate4 crate3)
	(= (weight crate4) 4)
	(located-at-depot crate5 distributor1)
	(on-depot crate5 crate2)
	(= (weight crate5) 50)
	(= (fuel-cost) 0)
; ----------------------------------------------------------------------------
	; Driverlog
;	(located-at-driverlog driver1 s0)
;	(located-at-driverlog driver2 s0)
;	(located-at-driverlog truckd1 s0)
;	(empty-driverlog truckd1)
;	(located-at-driverlog truckd2 s1)
;	(empty-driverlog truckd2)
;	(located-at-driverlog package1 s2)
;	(located-at-driverlog package2 s1)
;	(located-at-driverlog package3 s1)
;	(path s0 p0-1)
;	(path p0-1 s0)
;	(path s1 p0-1)
;	(path p0-1 s1)
;	(= (time-to-walk s0 p0-1) 37)
;	(= (time-to-walk p0-1 s0) 37)
;	(= (time-to-walk s1 p0-1) 39)
;	(= (time-to-walk p0-1 s1) 39)
;	(path s0 p0-2)
;	(path p0-2 s0)
;	(path s2 p0-2)
;	(path p0-2 s2)
;	(= (time-to-walk s0 p0-2) 68)
;	(= (time-to-walk p0-2 s0) 68)
;	(= (time-to-walk s2 p0-2) 7)
;	(= (time-to-walk p0-2 s2) 7)
;	(path s2 p2-1)
;	(path p2-1 s2)
;	(path s1 p2-1)
;	(path p2-1 s1)
;	(= (time-to-walk s2 p2-1) 30)
;	(= (time-to-walk p2-1 s2) 30)
;	(= (time-to-walk s1 p2-1) 19)
;	(= (time-to-walk p2-1 s1) 19)
;	(link s0 s2)
;	(link s2 s0)
;	(= (time-to-drive s0 s2) 52)
;	(= (time-to-drive s2 s0) 52)
;	(link s1 s0)
;	(link s0 s1)
;	(= (time-to-drive s1 s0) 63)
;	(= (time-to-drive s0 s1) 63)
;	(link s1 s2)
;	(link s2 s1)
;	(= (time-to-drive s1 s2) 86)
;	(= (time-to-drive s2 s1) 86)
	(= (driven) 0)
	(= (walked) 0)
; ----------------------------------------------------------------------------
	; Satellite
;	(supports instrument0 infrared1)
;	(supports instrument0 infrared0)
;	(calibration_target instrument0 star0)
;	(supports instrument1 image2)
;	(supports instrument1 infrared1)
;	(supports instrument1 infrared0)
;	(calibration_target instrument1 groundstation2)
;	(on_board instrument0 satellite0)
;	(on_board instrument1 satellite0)
;	(power_avail satellite0)
;	(pointing satellite0 planet4)
;	(= (data_capacity satellite0) 1000)
;	(= (fuel-satellite satellite0) 129)
;	(= (data planet3 infrared0) 183)
;	(= (data planet4 infrared0) 142)
;	(= (data phenomenon5 infrared0) 26)
;	(= (data phenomenon6 infrared0) 166)
;	(= (data star7 infrared0) 155)
;	(= (data planet3 infrared1) 146)
;	(= (data planet4 infrared1) 99)
;	(= (data phenomenon5 infrared1) 148)
;	(= (data phenomenon6 infrared1) 2)
;	(= (data star7 infrared1) 160)
;	(= (data planet3 image2) 137)
;	(= (data planet4 image2) 184)
;	(= (data phenomenon5 image2) 107)
;	(= (data phenomenon6 image2) 182)
;	(= (data star7 image2) 165)
;	(= (slew_time groundstation1 star0) 45)
;	(= (slew_time star0 groundstation1) 45)
;	(= (slew_time groundstation2 star0) 82)
;	(= (slew_time star0 groundstation2) 82)
;	(= (slew_time groundstation2 groundstation1) 42)
;	(= (slew_time groundstation1 groundstation2) 42)
;	(= (slew_time planet3 star0) 13)
;	(= (slew_time star0 planet3) 13)
;	(= (slew_time planet3 groundstation1) 17)
;	(= (slew_time groundstation1 planet3) 17)
;	(= (slew_time planet3 groundstation2) 17)
;	(= (slew_time groundstation2 planet3) 17)
;	(= (slew_time planet4 star0) 1)
;	(= (slew_time star0 planet4) 1)
;	(= (slew_time planet4 groundstation1) 14)
;	(= (slew_time groundstation1 planet4) 14)
;	(= (slew_time planet4 groundstation2) 58)
;	(= (slew_time groundstation2 planet4) 58)
;	(= (slew_time planet4 planet3) 30)
;	(= (slew_time planet3 planet4) 30)
;	(= (slew_time phenomenon5 star0) 78)
;	(= (slew_time star0 phenomenon5) 78)
;	(= (slew_time phenomenon5 groundstation1) 56)
;	(= (slew_time groundstation1 phenomenon5) 56)
;	(= (slew_time phenomenon5 groundstation2) 39)
;	(= (slew_time groundstation2 phenomenon5) 39)
;	(= (slew_time phenomenon5 planet3) 21)
;	(= (slew_time planet3 phenomenon5) 21)
;	(= (slew_time phenomenon5 planet4) 15)
;	(= (slew_time planet4 phenomenon5) 15)
;	(= (slew_time phenomenon6 star0) 87)
;	(= (slew_time star0 phenomenon6) 87)
;	(= (slew_time phenomenon6 groundstation1) 75)
;	(= (slew_time groundstation1 phenomenon6) 75)
;	(= (slew_time phenomenon6 groundstation2) 25)
;	(= (slew_time groundstation2 phenomenon6) 25)
;	(= (slew_time phenomenon6 planet3) 26)
;	(= (slew_time planet3 phenomenon6) 26)
;	(= (slew_time phenomenon6 planet4) 82)
;	(= (slew_time planet4 phenomenon6) 82)
;	(= (slew_time phenomenon6 phenomenon5) 36)
;	(= (slew_time phenomenon5 phenomenon6) 36)
;	(= (slew_time star7 star0) 30)
;	(= (slew_time star0 star7) 30)
;	(= (slew_time star7 groundstation1) 16)
;	(= (slew_time groundstation1 star7) 16)
;	(= (slew_time star7 groundstation2) 50)
;	(= (slew_time groundstation2 star7) 50)
;	(= (slew_time star7 planet3) 80)
;	(= (slew_time planet3 star7) 80)
;	(= (slew_time star7 planet4) 67)
;	(= (slew_time planet4 star7) 67)
;	(= (slew_time star7 phenomenon5) 20)
;	(= (slew_time phenomenon5 star7) 20)
;	(= (slew_time star7 phenomenon6) 29)
;	(= (slew_time phenomenon6 star7) 29)
	(= (data-stored) 0)
	(= (fuel-used) 0)
; ----------------------------------------------------------------------------
	; ZenoTravel
;	(located-zeno plane1 city0)
;	(= (capacity plane1) 6000)
;	(= (fuel-zeno plane1) 4000)
;	(= (slow-burn plane1) 4)
;	(= (fast-burn plane1) 15)
;	(= (onboard plane1) 0)
;	(= (zoom-limit plane1) 8)
;	(located-zeno person1 city0)
;	(located-zeno person2 city0)
;	(located-zeno person3 city1)
;	(= (distance city0 city0) 0)
;	(= (distance city0 city1) 678)
;	(= (distance city0 city2) 775)
;	(= (distance city1 city0) 678)
;	(= (distance city1 city1) 0)
;	(= (distance city1 city2) 810)
;	(= (distance city2 city0) 775)
;	(= (distance city2 city1) 810)
;	(= (distance city2 city2) 0)
	(= (total-fuel-used) 0)

)

(:goal (and
		; Depot Goal
		(on-depot crate0 crate1)
		(on-depot crate1 pallet2)
		(on-depot crate2 pallet0)

		; Satellite Goal
;		; (have_image planet4 infrared0)
;		; (have_image phenomenon5 image2)
;		; (have_image phenomenon6 infrared0)
;		; (have_image star7 infrared0)

		; DriverLog Goal
;		; (located-at-driverlog driver1 s1)
;		; (located-at-driverlog driver2 s1)
;		; (located-at-driverlog truckd1 s2)
;		; (located-at-driverlog truckd2 s0)
;		; (located-at-driverlog package1 s0)
;		; (located-at-driverlog package2 s2)
;		; (located-at-driverlog package3 s0)

		;ZenoTravel goal
;		; (located-zeno person1 city2)
;		; (located-zeno person2 city1)
;		; (located-zeno person3 city2)

	)
)
)