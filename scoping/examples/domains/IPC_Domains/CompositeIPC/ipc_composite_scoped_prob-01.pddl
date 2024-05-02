
(define (domain ipc_composite)
  (:requirements :typing :fluents :equality)
 (:types satellite direction instrument satellite-mode locatable_zeno city place locatable_depot location locatable-driverlog - object
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
  ;(:action turn_to
;   :parameters (?s - satellite ?d_new - direction ?d_prev - direction)
;   :precondition (and (pointing ?s ?d_prev)
;                   (not (= ?d_new ?d_prev))
;		(>= (fuel-satellite ?s) (slew_time ?d_new ?d_prev))
;              )
;   :effect (and  (pointing ?s ?d_new)
;                 (not (pointing ?s ?d_prev))
;		(decrease (fuel-satellite ?s) (slew_time ?d_new ?d_prev))
;		(increase (fuel-used) (slew_time ?d_new ?d_prev))
;           )
;  )


  ;(:action switch_on
;   :parameters (?i - instrument ?s - satellite)
;
;   :precondition (and (on_board ?i ?s)
;                      (power_avail ?s)
;                 )
;   :effect (and (power_on ?i)
;                (not (calibrated-satellite ?i))
;                (not (power_avail ?s))
;           )
;
;  )

  ;(:action switch_off
;   :parameters (?i - instrument ?s - satellite)
;
;   :precondition (and (on_board ?i ?s)
;                      (power_on ?i)
;                  )
;   :effect (and (not (power_on ?i))
;                (power_avail ?s)
;           )
;  )

  ;(:action calibrate-satellite
;   :parameters (?s - satellite ?i - instrument ?d - direction)
;   :precondition (and (on_board ?i ?s)
;		      (calibration_target ?i ?d)
;                      (pointing ?s ?d)
;                      (power_on ?i)
;                  )
;   :effect (calibrated-satellite ?i)
;  )


  ;(:action take_image-satellite
;   :parameters (?s - satellite ?d - direction ?i - instrument ?m - satellite-mode)
;   :precondition (and (calibrated-satellite ?i)
;                      (on_board ?i ?s)
;                      (supports ?i ?m)
;                      (power_on ?i)
;                      (pointing ?s ?d)
;                      (power_on ?i)
;			  (>= (data_capacity ?s) (data ?d ?m))
;               )
;   :effect (and (decrease (data_capacity ?s) (data ?d ?m)) (have_image ?d ?m)
;		(increase (data-stored) (data ?d ?m)))
;  )

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

(:action drive
:parameters (?x - truck-depot ?y - place ?z - place)
:precondition (and (located-at-depot ?x ?y))
:effect (and (not (located-at-depot ?x ?y)) (located-at-depot ?x ?z)
		(increase (fuel-cost) 10)))

(:action lift
:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
:precondition (and (located-at-depot ?x ?p) (available-depot ?x) (located-at-depot ?y ?p) (on-depot ?y ?z) (clear ?y))
:effect (and (not (located-at-depot ?y ?p)) (lifting ?x ?y) (not (clear ?y)) (not (available-depot ?x))
             (clear ?z) (not (on-depot ?y ?z)) (increase (fuel-cost) 1)))

(:action drop-depot
:parameters (?x - hoist ?y - crate ?z - surface ?p - place)
:precondition (and (located-at-depot ?x ?p) (located-at-depot ?z ?p) (clear ?z) (lifting ?x ?y))
:effect (and (available-depot ?x) (not (lifting ?x ?y)) (located-at-depot ?y ?p) (not (clear ?z)) (clear ?y)
		(on-depot ?y ?z)))

(:action load
:parameters (?x - hoist ?y - crate ?z - truck-depot ?p - place)
:precondition (and (located-at-depot ?x ?p) (located-at-depot ?z ?p) (lifting ?x ?y)
		(<= (+ (current_load ?z) (weight ?y)) (load_limit ?z)))
:effect (and (not (lifting ?x ?y)) (in-depot ?y ?z) (available-depot ?x)
		(increase (current_load ?z) (weight ?y))))

(:action unload
:parameters (?x - hoist ?y - crate ?z - truck-depot ?p - place)
:precondition (and (located-at-depot ?x ?p) (located-at-depot ?z ?p) (available-depot ?x) (in-depot ?y ?z))
:effect (and (not (in-depot ?y ?z)) (not (available-depot ?x)) (lifting ?x ?y)
		(decrease (current_load ?z) (weight ?y))))

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
