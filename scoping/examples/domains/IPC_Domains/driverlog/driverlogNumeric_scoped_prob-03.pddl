(define (domain driverlog)
  (:requirements :typing :fluents) 
  (:types           
    location locatable - object
		driver truck obj - locatable)

  (:predicates 
		(located-at ?obj - locatable ?loc - location)
		(in ?obj1 - obj ?obj - truck)
		(driving ?d - driver ?v - truck)
		(link ?x ?y - location) (path ?x ?y - location)
		(empty ?v - truck)
)
  (:functions (time-to-walk ?l1 ?l2 - location)
		(time-to-drive ?l1 ?l2 - location)
		(driven)
		(walked))


;(:action load-truck
;  :parameters
;   (?obj - obj
;    ?truck - truck
;    ?loc - location)
;  :precondition
;   (and (located-at ?truck ?loc) (located-at ?obj ?loc))
;  :effect
;   (and (not (located-at ?obj ?loc)) (in ?obj ?truck)))

;(:action unload-truck
;  :parameters
;   (?obj - obj
;    ?truck - truck
;    ?loc - location)
;  :precondition
;   (and (located-at ?truck ?loc) (in ?obj ?truck))
;  :effect
;   (and (not (in ?obj ?truck)) (located-at ?obj ?loc)))

(:action board-truck
  :parameters
   (?driver - driver
    ?truck - truck
    ?loc - location)
  :precondition
   (and (located-at ?truck ?loc) (located-at ?driver ?loc) (empty ?truck))
  :effect
   (and (not (located-at ?driver ?loc)) (driving ?driver ?truck) (not (empty ?truck))))

(:action disembark-truck
  :parameters
   (?driver - driver
    ?truck - truck
    ?loc - location)
  :precondition
   (and (located-at ?truck ?loc) (driving ?driver ?truck))
  :effect
   (and (not (driving ?driver ?truck)) (located-at ?driver ?loc) (empty ?truck)))

(:action drive-truck
  :parameters
   (?truck - truck
    ?loc-from - location
    ?loc-to - location
    ?driver - driver)
  :precondition
   (and (located-at ?truck ?loc-from)
   (driving ?driver ?truck) (link ?loc-from ?loc-to))
  :effect
   (and (not (located-at ?truck ?loc-from)) (located-at ?truck ?loc-to)
		(increase (driven) (time-to-drive ?loc-from ?loc-to))))

(:action walk
  :parameters
   (?driver - driver
    ?loc-from - location
    ?loc-to - location)
  :precondition
   (and (located-at ?driver ?loc-from) (path ?loc-from ?loc-to))
  :effect
   (and (not (located-at ?driver ?loc-from)) (located-at ?driver ?loc-to)
	(increase (walked) (time-to-walk ?loc-from ?loc-to))))

 
)
