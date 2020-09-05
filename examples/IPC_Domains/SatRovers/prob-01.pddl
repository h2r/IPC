(define (problem strips-sat-x-0)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	image1 spectrograph2 thermograph0 - mode
	GroundStation0 GroundStation1 GroundStation2 - direction
	general - lander
	colour high_res low_res - mode
	rover0 - rover
	rover0store - store
	waypoint0 waypoint1 waypoint2 waypoint3 - waypoint
	camera0 - camera
	objective0 objective1 - objective
)
(:init
	(supports instrument0 thermograph0)
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
	(calibrated-satellite instrument0)

	(visible waypoint1 waypoint0)
	(visible waypoint0 waypoint1)
	(visible waypoint2 waypoint0)
	(visible waypoint0 waypoint2)
	(visible waypoint2 waypoint1)
	(visible waypoint1 waypoint2)
	(visible waypoint3 waypoint0)
	(visible waypoint0 waypoint3)
	(visible waypoint3 waypoint1)
	(visible waypoint1 waypoint3)
	(visible waypoint3 waypoint2)
	(visible waypoint2 waypoint3)
	(= (recharges) 0)
	(at_soil_sample waypoint0)
	(in_sun waypoint0)
	(at_rock_sample waypoint1)
	(at_soil_sample waypoint2)
	(at_rock_sample waypoint2)
	(at_soil_sample waypoint3)
	(at_rock_sample waypoint3)
	(at_lander general waypoint0)
	(channel_free general)
	(= (energy rover0) 50)
	(located-at rover0 waypoint3)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_soil_analysis rover0)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(can_traverse rover0 waypoint3 waypoint0)
	(can_traverse rover0 waypoint0 waypoint3)
	(can_traverse rover0 waypoint3 waypoint1)
	(can_traverse rover0 waypoint1 waypoint3)
	(can_traverse rover0 waypoint1 waypoint2)
	(can_traverse rover0 waypoint2 waypoint1)
	(on_board-rover camera0 rover0)
	(calibration_target-rover camera0 objective1)
	
	(supports-camera-rover camera0 high_res)
	(supports-camera-rover camera0 colour)
	(supports-camera-rover camera0 low_res)

	(visible_from objective0 waypoint0)
	(visible_from objective0 waypoint1)
	(visible_from objective0 waypoint2)
	(visible_from objective0 waypoint3)
	(visible_from objective1 waypoint0)
	(visible_from objective1 waypoint1)
	(visible_from objective1 waypoint2)
	(visible_from objective1 waypoint3)
)
(:goal (and
		(communicated_soil_data waypoint2)
		(communicated_rock_data waypoint3)
		(communicated_image_data objective1 high_res)
		(communicated_image_data objective1 colour)
		(communicated_image_data objective1 low_res)
	    (calibrated-satellite instrument0)
	   )
)

)
