(define (problem roverprob01) (:domain satrovers)
(:objects
	general - lander
	colour high_res low_res - mode
	rover0 - rover
	rover0store - store
	waypoint0 waypoint1 - waypoint
	camera0 - camera
	objective0 objective1 - objective

	satellite0 - satellite
	instrument0 - instrument
	image1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	Star0 - direction
	GroundStation1 - direction
	GroundStation2 - direction
	Phenomenon3 - direction
	Phenomenon4 - direction
	Star5 - direction
	Phenomenon6 - direction

	)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation2)
	(= (data_capacity satellite0) 1000)
	(= (fuel satellite0) 112)
	(= (data Phenomenon3 image1) 22)
	(= (data Phenomenon4 image1) 120)
	(= (data Star5 image1) 203)
	(= (data Phenomenon6 image1) 144)
	(= (data Phenomenon3 spectrograph2) 125)
	(= (data Phenomenon4 spectrograph2) 196)
	(= (data Star5 spectrograph2) 68)
	(= (data Phenomenon6 spectrograph2) 174)
	(= (data Phenomenon3 thermograph0) 136)
	(= (data Phenomenon4 thermograph0) 134)
	(= (data Star5 thermograph0) 273)
	(= (data Phenomenon6 thermograph0) 219)
	(= (slew_time GroundStation1 Star0) 18)
	(= (slew_time Star0 GroundStation1) 18)
	(= (slew_time GroundStation2 Star0) 38)
	(= (slew_time Star0 GroundStation2) 38)
	(= (slew_time GroundStation2 GroundStation1) 68)
	(= (slew_time GroundStation1 GroundStation2) 68)
	(= (slew_time Phenomenon3 Star0) 14)
	(= (slew_time Star0 Phenomenon3) 14)
	(= (slew_time Phenomenon3 GroundStation1) 89)
	(= (slew_time GroundStation1 Phenomenon3) 89)
	(= (slew_time Phenomenon3 GroundStation2) 33)
	(= (slew_time GroundStation2 Phenomenon3) 33)
	(= (slew_time Phenomenon4 Star0) 35)
	(= (slew_time Star0 Phenomenon4) 35)
	(= (slew_time Phenomenon4 GroundStation1) 31)
	(= (slew_time GroundStation1 Phenomenon4) 31)
	(= (slew_time Phenomenon4 GroundStation2) 39)
	(= (slew_time GroundStation2 Phenomenon4) 39)
	(= (slew_time Phenomenon4 Phenomenon3) 25)
	(= (slew_time Phenomenon3 Phenomenon4) 25)
	(= (slew_time Star5 Star0) 36)
	(= (slew_time Star0 Star5) 36)
	(= (slew_time Star5 GroundStation1) 8)
	(= (slew_time GroundStation1 Star5) 8)
	(= (slew_time Star5 GroundStation2) 62)
	(= (slew_time GroundStation2 Star5) 62)
	(= (slew_time Star5 Phenomenon3) 10)
	(= (slew_time Phenomenon3 Star5) 10)
	(= (slew_time Star5 Phenomenon4) 64)
	(= (slew_time Phenomenon4 Star5) 64)
	(= (slew_time Phenomenon6 Star0) 77)
	(= (slew_time Star0 Phenomenon6) 77)
	(= (slew_time Phenomenon6 GroundStation1) 17)
	(= (slew_time GroundStation1 Phenomenon6) 17)
	(= (slew_time Phenomenon6 GroundStation2) 50)
	(= (slew_time GroundStation2 Phenomenon6) 50)
	(= (slew_time Phenomenon6 Phenomenon3) 14)
	(= (slew_time Phenomenon3 Phenomenon6) 14)
	(= (slew_time Phenomenon6 Phenomenon4) 28)
	(= (slew_time Phenomenon4 Phenomenon6) 28)
	(= (slew_time Phenomenon6 Star5) 29)
	(= (slew_time Star5 Phenomenon6) 29)
	(= (data-stored) 0)
	(= (fuel-used) 0)

	(visible waypoint1 waypoint0)
	(visible waypoint0 waypoint1)
	(= (recharges) 0)
	(at_soil_sample waypoint0)
	(in_sun waypoint0)
	(at_rock_sample waypoint1)
	(at_lander general waypoint0)
	(channel_free general)
	(= (energy rover0) 5)
	(located-at rover0 waypoint0)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_soil_analysis rover0)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(on_board-rover camera0 rover0)
    (can_traverse rover0 waypoint0 waypoint1)
    (can_traverse rover0 waypoint1 waypoint0)
	(calibration_target-rover camera0 objective1)
	(supports-camera-rover camera0 colour)
	(supports-camera-rover camera0 high_res)
	(visible_from objective0 waypoint0)
	(visible_from objective0 waypoint1)
	(visible_from objective1 waypoint0)
	(visible_from objective1 waypoint1)
)

(:goal (and
; (communicated_soil_data waypoint1)
(= (energy rover0) 1500)
; (have_image Phenomenon4 thermograph0)
; (have_image Star5 thermograph0)
; (have_image Phenomenon6 thermograph0)
; (communicated_rock_data waypoint1)
; (communicated_image_data objective1 high_res)
	)
)

)
