(define (problem roverprob01) (:domain Rover)
(:objects
	general - lander
	colour high_res low_res - mode
	rover0 - rover
	rover0store - store
	waypoint0 waypoint1 - waypoint
	camera0 - camera
	objective0 objective1 - objective
	)
(:init
	(visible waypoint1 waypoint0)
	(visible waypoint0 waypoint1)
	(= (recharges) 0)
	(at_soil_sample waypoint1)
	(in_sun waypoint0)
    (in_sun waypoint1)
	(at_rock_sample waypoint1)
	(at_lander general waypoint0)
	(channel_free general)
	(= (energy rover0) 0)
	(located-at rover0 waypoint0)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_soil_analysis rover0)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(on_board camera0 rover0)
    (can_traverse rover0 waypoint0 waypoint1)
    (can_traverse rover0 waypoint0 waypoint0)
    (can_traverse rover0 waypoint1 waypoint0)
    (can_traverse rover0 waypoint1 waypoint1)
	(calibration_target camera0 objective1)
	(supports camera0 colour)
	(supports camera0 high_res)
    (supports camera0 low_res)
	(visible_from objective0 waypoint0)
	(visible_from objective0 waypoint1)
	(visible_from objective1 waypoint0)
	(visible_from objective1 waypoint1)
)

(:goal (and
(communicated_soil_data waypoint1)
(= (energy rover0) 750)
(communicated_rock_data waypoint1)
(communicated_image_data objective1 colour)
	)
)

)
