(define (problem strips-sat-x-0)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
;	image1 spectrograph2 thermograph0 - mode
	GroundStation0 GroundStation1 GroundStation2 - direction
)
(:init
;	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation0)
	(= (data_capacity satellite0) 1000)
	(= (fuel satellite0) 112)
	(= (slew_time GroundStation2 GroundStation1) 68)
	(= (slew_time GroundStation1 GroundStation2) 68)
	(= (slew_time GroundStation0 GroundStation1) 10)
	(= (slew_time GroundStation1 GroundStation0) 10)
	(= (data-stored) 0)
	(= (fuel-used) 0)
)
(:goal (calibrated instrument0)
)

)