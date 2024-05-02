(define (domain driverlog)
  (:requirements :typing :fluents) 
  (:types           
    location locatable-driverlog - object
		driver truck-driverlog obj - locatable-driverlog)

  (:predicates 
		(located-at-driverlog ?obj - locatable-driverlog ?loc - location)
		(in ?obj1 - obj ?obj - truck-driverlog)
		(driving ?d - driver ?v - truck-driverlog)
		(link ?x ?y - location) (path ?x ?y - location)
		(empty-driverlog  ?v - truck-driverlog)
)
  (:functions (time-to-walk ?l1 ?l2 - location)
		(time-to-drive ?l1 ?l2 - location)
		(driven)
		(walked)
    )


(:action load-truck-driverlog
  :parameters
   (?obj - obj
    ?truck - truck
    ?loc - location)
  :precondition
   (and (located-at-driverlog ?truck ?loc) (located-at-driverlog ?obj ?loc))
  :effect
   (and (not (located-at-driverlog ?obj ?loc)) (in ?obj ?truck)))

(:action unload-truck
  :parameters
   (?obj - obj
    ?truck - truck
    ?loc - location)
  :precondition
   (and (located-at-driverlog ?truck ?loc) (in ?obj ?truck))
  :effect
   (and (not (in ?obj ?truck)) (located-at-driverlog ?obj ?loc)))

(:action board-truck
  :parameters
   (?driver - driver
    ?truck - truck
    ?loc - location)
  :precondition
   (and (located-at-driverlog ?truck ?loc) (located-at-driverlog ?driver ?loc) (empty-driverlog  ?truck))
  :effect
   (and (not (located-at-driverlog ?driver ?loc)) (driving ?driver ?truck) (not (empty-driverlog  ?truck))))

(:action disembark-truck
  :parameters
   (?driver - driver
    ?truck - truck
    ?loc - location)
  :precondition
   (and (located-at-driverlog ?truck ?loc) (driving ?driver ?truck))
  :effect
   (and (not (driving ?driver ?truck)) (located-at-driverlog ?driver ?loc) (empty-driverlog  ?truck)))

(:action drive-truck
  :parameters
   (?truck - truck
    ?loc-from - location
    ?loc-to - location
    ?driver - driver)
  :precondition
   (and (located-at-driverlog ?truck ?loc-from)
   (driving ?driver ?truck) (link ?loc-from ?loc-to))
  :effect
   (and (not (located-at-driverlog ?truck ?loc-from)) (located-at-driverlog ?truck ?loc-to)
		(increase (driven) (time-to-drive ?loc-from ?loc-to))))

(:action walk
  :parameters
   (?driver - driver
    ?loc-from - location
    ?loc-to - location)
  :precondition
   (and (located-at-driverlog ?driver ?loc-from) (path ?loc-from ?loc-to))
  :effect
   (and (not (located-at-driverlog ?driver ?loc-from)) (located-at-driverlog ?driver ?loc-to)
	(increase (walked) (time-to-walk ?loc-from ?loc-to))))

 
)
