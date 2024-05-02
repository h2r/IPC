(define (problem strips-sat-x-0)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 instrument1 instrument2 instrument3 instrument4 instrument5 instrument6 - instrument
	image1 spectrograph2 thermograph0 - mode
	GroundStation0 GroundStation1 GroundStation2 - direction
)
(:init
	(supports instrument0 thermograph0)
    (supports instrument1 thermograph0)
    (supports instrument2 thermograph0)
    (supports instrument3 thermograph0)
    (supports instrument4 thermograph0)
    (supports instrument5 thermograph0)
    (supports instrument6 thermograph0)
    (supports instrument6 spectrograph2)


	(calibration_target instrument0 GroundStation2)
    (calibration_target instrument1 GroundStation2)
    (calibration_target instrument2 GroundStation2)
    (calibration_target instrument3 GroundStation2)
    (calibration_target instrument4 GroundStation2)
    (calibration_target instrument5 GroundStation2)
    (calibration_target instrument6 GroundStation2)

	(on_board instrument0 satellite0)
    (on_board instrument1 satellite0)
    (on_board instrument2 satellite0)
    (on_board instrument3 satellite0)
    (on_board instrument4 satellite0)
    (on_board instrument5 satellite0)
    (on_board instrument6 satellite0)

	(power_avail satellite0)
	(pointing satellite0 GroundStation0)
	(= (data_capacity satellite0) 1000)
	(= (fuel satellite0) 1120)
	(= (slew_time GroundStation2 GroundStation1) 68)
	(= (slew_time GroundStation1 GroundStation2) 68)
	(= (slew_time GroundStation0 GroundStation1) 10)
	(= (slew_time GroundStation1 GroundStation0) 10)
	(= (data-stored) 0)
	(= (fuel-used) 0)
)
(:goal (and (calibrated instrument0)
            (calibrated instrument1)
            (calibrated instrument2)
            (calibrated instrument3)
            (calibrated instrument4)
            (calibrated instrument5)
            (calibrated instrument6)))

)
