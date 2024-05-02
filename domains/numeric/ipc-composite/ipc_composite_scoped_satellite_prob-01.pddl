
(define (domain ipc_composite)
  (:requirements :typing :fluents :equality)
 (:types satellite direction instrument satellite-mode rover-mode rover waypoint store camera mode lander objective locatable_zeno city place locatable_depot location locatable-driverlog - object
         aircraft person - locatable_zeno
         depot distributor - place
         truck-depot hoist surface - locatable_depot
         pallet crate - surface
         driver truck-driverlog obj - locatable-driverlog
)

 (:predicates 
          ; Satellite
          (on_board ?i - instrument ?s - satellite)
	     (supports ?i - instrument ?m - satellite-mode)
	     (pointing ?s - satellite ?d - direction)
	     (power_avail ?s - satellite)
	     (power_on ?i - instrument)
	     (calibrated-satellite ?i - instrument)
	     (have_image ?d - direction ?m - satellite-mode)
	     (calibration_target ?i - instrument ?d - direction)
          
          ;Rover
          (located-at-rover ?x - rover ?y - waypoint) 
          (at_lander ?x - lander ?y - waypoint)
          (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
	     (equipped_for_soil_analysis ?r - rover)
          (equipped_for_rock_analysis ?r - rover)
          (equipped_for_imaging ?r - rover)
          (empty-rover ?s - store)
          (have_rock_analysis ?r - rover ?w - waypoint)
          (have_soil_analysis ?r - rover ?w - waypoint)
          (full ?s - store)
	     (calibrated ?c - camera ?r - rover)
	     (supports-camera-rover ?c - camera ?m - rover-mode)
          (available-rover ?r - rover)
          (visible ?w - waypoint ?p - waypoint)
          (have_image-rover ?r - rover ?o - objective ?m - rover-mode)
          (communicated_soil_data ?w - waypoint)
          (communicated_rock_data ?w - waypoint)
          (communicated_image_data ?o - objective ?m - rover-mode)
	     (at_soil_sample ?w - waypoint)
	     (at_rock_sample ?w - waypoint)
          (visible_from ?o - objective ?w - waypoint)
	     (store_of ?s - store ?r - rover)
	     (calibration_target-rover ?i - camera ?o - objective)
	     (on_board-rover ?i - camera ?r - rover)
	     (channel_free ?l - lander)
	     (in_sun ?w - waypoint)

          ;Zeno
          (located-zeno ?x - locatable_zeno  ?c - city)
          (in-zeno ?p - person ?a - aircraft)

          ; Depot
          (located-at-depot ?x - locatable_depot ?y - place) 
          (on-depot ?x - crate ?y - surface)
          (in-depot ?x - crate ?y - truck-depot)
          (lifting ?x - hoist ?y - crate)
          (available-depot ?x - hoist)
          (clear ?x - surface)

          ;driverlog
          (located-at-driverlog ?obj - locatable-driverlog ?loc - location)
		      (in ?obj1 - obj ?obj - truck-driverlog)
		      (driving ?d - driver ?v - truck-driverlog)
		      (link ?x ?y - location) (path ?x ?y - location)
          (empty-driverlog  ?v - truck-driverlog)
)

  (:functions (data_capacity ?s - satellite)
	         (data ?d - direction ?m - satellite-mode)
		    (slew_time ?a ?b - direction)
		    (data-stored)
		    (fuel-satellite ?s - satellite)
		    (fuel-used)
          
               ; Rover
               (energy ?r - rover) 
               (recharges)

               ; Zeno
               (fuel-zeno ?a - aircraft)
               (distance ?c1 - city ?c2 - city)
               (slow-burn ?a - aircraft)
               (fast-burn ?a - aircraft)
               (capacity ?a - aircraft)
               (total-fuel-used)
               (onboard ?a - aircraft)
               (zoom-limit ?a - aircraft)

               ; Depot
               (load_limit ?t - truck-depot) 
               (current_load ?t - truck-depot) 
               (weight ?c - crate)
               (fuel-cost)

               ; Driverlog
               (time-to-walk ?l1 ?l2 - location)
               (time-to-drive ?l1 ?l2 - location)
               (driven)
               (walked)
 
  )

;--------------------------------------------------------------------------------
; Satellite
  (:action turn_to
   :parameters (?s - satellite ?d_new - direction ?d_prev - direction)
   :precondition (and (pointing ?s ?d_prev)
                   (not (= ?d_new ?d_prev))
		(>= (fuel-satellite ?s) (slew_time ?d_new ?d_prev))
              )
   :effect (and  (pointing ?s ?d_new)
                 (not (pointing ?s ?d_prev))
		(decrease (fuel-satellite ?s) (slew_time ?d_new ?d_prev))
		(increase (fuel-used) (slew_time ?d_new ?d_prev))
           )
  )

 
  (:action switch_on
   :parameters (?i - instrument ?s - satellite)
 
   :precondition (and (on_board ?i ?s) 
                      (power_avail ?s)
                 )
   :effect (and (power_on ?i)
                (not (calibrated-satellite ?i))
                (not (power_avail ?s))
           )
          
  )
 
  (:action switch_off
   :parameters (?i - instrument ?s - satellite)
 
   :precondition (and (on_board ?i ?s)
                      (power_on ?i) 
                  )
   :effect (and (not (power_on ?i))
                (power_avail ?s)
           )
  )

  (:action calibrate-satellite
   :parameters (?s - satellite ?i - instrument ?d - direction)
   :precondition (and (on_board ?i ?s)
		      (calibration_target ?i ?d)
                      (pointing ?s ?d)
                      (power_on ?i)
                  )
   :effect (calibrated-satellite ?i)
  )
 

  (:action take_image-satellite
   :parameters (?s - satellite ?d - direction ?i - instrument ?m - satellite-mode)
   :precondition (and (calibrated-satellite ?i)
                      (on_board ?i ?s)
                      (supports ?i ?m)
                      (power_on ?i)
                      (pointing ?s ?d)
                      (power_on ?i)
			  (>= (data_capacity ?s) (data ?d ?m))
               )
   :effect (and (decrease (data_capacity ?s) (data ?d ?m)) (have_image ?d ?m)
		(increase (data-stored) (data ?d ?m)))
  )
;--------------------------------------------------------------------------------
; Rover
  ;(:action navigate
;:parameters (?x - rover ?y - waypoint ?z - waypoint) 
;:precondition (and (can_traverse ?x ?y ?z) (available-rover ?x) (located-at-rover ?x ?y) 
;                (visible ?y ?z) (>= (energy ?x) 8)
;	    )
;:effect (and (decrease (energy ?x) 8) (not (located-at-rover ?x ?y)) (located-at-rover ?x ?z)
;		)
;)

;(:action recharge
;:parameters (?x - rover ?w - waypoint)
;:precondition (and (located-at-rover ?x ?w) (in_sun ?w) (<= (energy ?x) 1500))
;:effect (and (increase (energy ?x) 20) (increase (recharges) 1)) 
;)

;(:action sample_soil
;:parameters (?x - rover ?s - store ?p - waypoint)
;:precondition (and (located-at-rover ?x ?p)(>= (energy ?x) 3) (at_soil_sample ?p) (equipped_for_soil_analysis ?x) (store_of ?s ?x) (empty-rover ?s)
;		)
;:effect (and (not (empty-rover ?s)) (full ?s) (decrease (energy ?x) 3) (have_soil_analysis ?x ?p) (not (at_soil_sample ?p))
;		)
;)

;(:action sample_rock
;:parameters (?x - rover ?s - store ?p - waypoint)
;:precondition (and  (located-at-rover ?x ?p) (>= (energy ?x) 5)(at_rock_sample ?p) (equipped_for_rock_analysis ?x) (store_of ?s ?x)(empty-rover ?s)
;		)
;:effect (and (not (empty-rover ?s)) (full ?s) (decrease (energy ?x) 5) (have_rock_analysis ?x ?p) (not (at_rock_sample ?p))
;		)
;)

;(:action drop-rover
;:parameters (?x - rover ?y - store)
;:precondition (and (store_of ?y ?x) (full ?y)
;		)
;:effect (and (not (full ?y)) (empty-rover ?y)
;	)
;)

;(:action calibrate-rover
; :parameters (?r - rover ?i - camera ?t - objective ?w - waypoint)
; :precondition (and (equipped_for_imaging ?r) (>= (energy ?r) 2)(calibration_target-rover ?i ?t) (located-at-rover ?r ?w) (visible_from ?t ?w)(on_board-rover ?i ?r)
;		)
; :effect (and (decrease (energy ?r) 2)(calibrated ?i ?r) )
;)

;(:action take_image-rover
; :parameters (?r - rover ?p - waypoint ?o - objective ?i - camera ?m - rover-mode)
; :precondition (and (calibrated ?i ?r)
;			 (on_board-rover ?i ?r)
;                      (equipped_for_imaging ?r)
;                      (supports-camera-rover ?i ?m)
;			  (visible_from ?o ?p)
;                     (located-at-rover ?r ?p)
;			(>= (energy ?r) 1)
;               )
; :effect (and (have_image-rover ?r ?o ?m)(not (calibrated ?i ?r))(decrease (energy ?r) 1)
;		)
;)

;(:action communicate_soil_data
; :parameters (?r - rover ?l - lander ?p - waypoint ?x - waypoint ?y - waypoint)
; :precondition (and (located-at-rover ?r ?x)(at_lander ?l ?y)(have_soil_analysis ?r ?p) 
;                   (visible ?x ?y)(available-rover ?r)(channel_free ?l)(>= (energy ?r) 4)
;            )
; :effect (and (not (available-rover ?r))(not (channel_free ?l))
;(channel_free ?l) (communicated_soil_data ?p)(available-rover ?r)(decrease (energy ?r) 4)
;	)
;)

;(:action communicate_rock_data
; :parameters (?r - rover ?l - lander ?p - waypoint ?x - waypoint ?y - waypoint)
; :precondition (and (located-at-rover ?r ?x)(at_lander ?l ?y)(have_rock_analysis ?r ?p)(>= (energy ?r) 4)
;                   (visible ?x ?y)(available-rover ?r)(channel_free ?l)
;            )
; :effect (and   (not (available-rover ?r))(not (channel_free ?l))
;(channel_free ?l)
;(communicated_rock_data ?p)(available-rover ?r)(decrease (energy ?r) 4)
;          )
;)


;(:action communicate_image_data
; :parameters (?r - rover ?l - lander ?o - objective ?m - rover-mode ?x - waypoint ?y - waypoint)
; :precondition (and (located-at-rover ?r ?x)(at_lander ?l ?y)(have_image-rover ?r ?o ?m)(visible ?x ?y)(available-rover ?r)(channel_free ?l)(>= (energy ?r) 6)
;            )
; :effect (and (not (available-rover ?r))(not (channel_free ?l))
;(channel_free ?l)
;(communicated_image_data ?o ?m)(available-rover ?r)(decrease (energy ?r) 6)
;          )
;)
;--------------------------------------------------------------------------------
; Zeno
;(:action board
; :parameters (?p - person ?a - aircraft ?c - city)
; :precondition (and (located-zeno ?p ?c)
;                 (located-zeno ?a ?c))
; :effect (and (not (located-zeno ?p ?c))
;              (in-zeno ?p ?a)
;		(increase (onboard ?a) 1)))


;(:action debark
; :parameters (?p - person ?a - aircraft ?c - city)
; :precondition (and (in-zeno ?p ?a)
;                 (located-zeno ?a ?c))
; :effect (and (not (in-zeno ?p ?a))
;              (located-zeno ?p ?c)
;		(decrease (onboard ?a) 1)))

;(:action fly-slow
; :parameters (?a - aircraft ?c1 ?c2 - city)
; :precondition (and (located-zeno ?a ?c1)
;                 (>= (fuel-zeno ?a) 
;                         (* (distance ?c1 ?c2) (slow-burn ?a))))
; :effect (and (not (located-zeno ?a ?c1))
;              (located-zeno ?a ?c2)
;              (increase (total-fuel-used)
;                         (* (distance ?c1 ?c2) (slow-burn ?a)))
;              (decrease (fuel-zeno ?a) 
;                         (* (distance ?c1 ?c2) (slow-burn ?a)))))
                                  
;(:action fly-fast
; :parameters (?a - aircraft ?c1 ?c2 - city)
; :precondition (and (located-zeno ?a ?c1)
;                 (>= (fuel-zeno ?a) 
;                         (* (distance ?c1 ?c2) (fast-burn ?a)))
;                 (<= (onboard ?a) (zoom-limit ?a)))
; :effect (and (not (located-zeno ?a ?c1))
;              (located-zeno ?a ?c2)
;              (increase (total-fuel-used)
;                         (* (distance ?c1 ?c2) (fast-burn ?a)))
;              (decrease (fuel-zeno ?a) 
;                         (* (distance ?c1 ?c2) (fast-burn ?a)))
;	)
;) 

;(:action refuel	
; :parameters (?a - aircraft)
; :precondition (and (> (capacity ?a) (fuel-zeno ?a)))
; :effect (and (assign (fuel-zeno ?a) (capacity ?a)))
;)

;--------------------------------------------------------------------------------
; Depot

;(:action drive
;:parameters (?x - truck-depot ?y - place ?z - place) 
;:precondition (and (located-at-depot ?x ?y))
;:effect (and (not (located-at-depot ?x ?y)) (located-at-depot ?x ?z)
;		(increase (fuel-cost) 10)))

;(:action lift
;:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
;:precondition (and (located-at-depot ?x ?p) (available-depot ?x) (located-at-depot ?y ?p) (on-depot ?y ?z) (clear ?y))
;:effect (and (not (located-at-depot ?y ?p)) (lifting ?x ?y) (not (clear ?y)) (not (available-depot ?x)) 
;             (clear ?z) (not (on-depot ?y ?z)) (increase (fuel-cost) 1)))

;(:action drop-depot
;:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
;:precondition (and (located-at-depot ?x ?p) (located-at-depot ?z ?p) (clear ?z) (lifting ?x ?y))
;:effect (and (available-depot ?x) (not (lifting ?x ?y)) (located-at-depot ?y ?p) (not (clear ?z)) (clear ?y)
;		(on-depot ?y ?z)))

;(:action load
;:parameters (?x - hoist ?y - crate ?z - truck-depot ?p - place)
;:precondition (and (located-at-depot ?x ?p) (located-at-depot ?z ?p) (lifting ?x ?y)
;		(<= (+ (current_load ?z) (weight ?y)) (load_limit ?z)))
;:effect (and (not (lifting ?x ?y)) (in-depot ?y ?z) (available-depot ?x)
;		(increase (current_load ?z) (weight ?y))))

;(:action unload 
;:parameters (?x - hoist ?y - crate ?z - truck-depot ?p - place)
;:precondition (and (located-at-depot ?x ?p) (located-at-depot ?z ?p) (available-depot ?x) (in-depot ?y ?z))
;:effect (and (not (in-depot ?y ?z)) (not (available-depot ?x)) (lifting ?x ?y)
;		(decrease (current_load ?z) (weight ?y))))

;--------------------------------------------------------------------------------
; Driverlog

;(:action load-truck-driverlog
;  :parameters
;   (?obj - obj
;    ?truck - truck-driverlog
;    ?loc - location)
;  :precondition
;   (and (located-at-driverlog ?truck ?loc) (located-at-driverlog ?obj ?loc))
;  :effect
;   (and (not (located-at-driverlog ?obj ?loc)) (in ?obj ?truck)))

;(:action unload-truck
;  :parameters
;   (?obj - obj
;    ?truck - truck-driverlog
;    ?loc - location)
;  :precondition
;   (and (located-at-driverlog ?truck ?loc) (in ?obj ?truck))
;  :effect
;   (and (not (in ?obj ?truck)) (located-at-driverlog ?obj ?loc)))

;(:action board-truck
;  :parameters
;   (?driver - driver
;    ?truck - truck-driverlog
;    ?loc - location)
;  :precondition
;   (and (located-at-driverlog ?truck ?loc) (located-at-driverlog ?driver ?loc) (empty-driverlog  ?truck))
;  :effect
;   (and (not (located-at-driverlog ?driver ?loc)) (driving ?driver ?truck) (not (empty-driverlog  ?truck))))

;(:action disembark-truck
;  :parameters
;   (?driver - driver
;    ?truck - truck-driverlog
;    ?loc - location)
;  :precondition
;   (and (located-at-driverlog ?truck ?loc) (driving ?driver ?truck))
;  :effect
;   (and (not (driving ?driver ?truck)) (located-at-driverlog ?driver ?loc) (empty-driverlog  ?truck)))

;(:action drive-truck
;  :parameters
;   (?truck - truck-driverlog
;    ?loc-from - location
;    ?loc-to - location
;    ?driver - driver)
;  :precondition
;   (and (located-at-driverlog ?truck ?loc-from)
;   (driving ?driver ?truck) (link ?loc-from ?loc-to))
;  :effect
;   (and (not (located-at-driverlog ?truck ?loc-from)) (located-at-driverlog ?truck ?loc-to)
;		(increase (driven) (time-to-drive ?loc-from ?loc-to))))

;(:action walk
;  :parameters
;   (?driver - driver
;    ?loc-from - location
;    ?loc-to - location)
;  :precondition
;   (and (located-at-driverlog ?driver ?loc-from) (path ?loc-from ?loc-to))
;  :effect
;   (and (not (located-at-driverlog ?driver ?loc-from)) (located-at-driverlog ?driver ?loc-to)
;	(increase (walked) (time-to-walk ?loc-from ?loc-to))))

)

