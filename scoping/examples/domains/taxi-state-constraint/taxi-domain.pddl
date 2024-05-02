;; Domain constructed by: <redacted for review>

(define (domain universal_multipasstaxi)
(:requirements :equality :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)

(:types passenger taxi - object)
(:predicates (in-taxi ?p - passenger ?t - taxi))
(:functions (pass-y ?p - passenger)
            (pass-x ?p - passenger)
            (taxi-x ?t - taxi)
            (taxi-y ?t - taxi))


(:action pickup
 :parameters (?p - passenger ?t - taxi)
 :precondition (and (= (pass-x ?p) (taxi-x ?t)) 
                    (= (pass-y ?p) (taxi-y ?t))
                    (forall (?pn - passenger)
                       (not (in-taxi ?pn ?t))
                    )
                )
 :effect (in-taxi ?p ?t))

(:action dropoff
 :parameters (?p - passenger ?t - taxi)
 :precondition (in-taxi ?p ?t)
 :effect (not (in-taxi ?p ?t)))

(:action move-north
 :parameters (?t - taxi)
 :precondition (forall (?pn - passenger)
                        (not (in-taxi ?pn ?t))
                    )
 :effect (and (increase (taxi-y ?t) 1)))

(:action move-south
 :parameters (?t - taxi)
 :precondition (forall (?pn - passenger)
                        (not (in-taxi ?pn ?t))
                    )
 :effect (and (decrease (taxi-y ?t) 1)))

(:action move-east
 :parameters (?t - taxi)
 :precondition (forall (?pn - passenger)
                        (not (in-taxi ?pn ?t))
                    )
 :effect (and (increase (taxi-x ?t) 1)))

(:action move-west
 :parameters (?t - taxi)
 :precondition (forall (?pn - passenger)
                        (not (in-taxi ?pn ?t))
                    )
 :effect (and (decrease (taxi-x ?t) 1)))

(:action move-north-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (and (in-taxi ?p ?t)
                    (forall (?pb - passenger) (or (= ?pb ?p) (not (in-taxi ?pb ?t)))))
 :effect (and (increase (taxi-y ?t) 1)
              (increase (pass-y ?p) 1)))

(:action move-south-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (and (in-taxi ?p ?t)
                    (forall (?pb - passenger) (or (= ?pb ?p) (not (in-taxi ?pb ?t)))))
 :effect (and (decrease (taxi-y ?t) 1)
              (decrease (pass-y ?p) 1)))

(:action move-east-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (and (in-taxi ?p ?t)
                    (forall (?pb - passenger) (or (= ?pb ?p) (not (in-taxi ?pb ?t)))))
 :effect (and (increase (taxi-x ?t) 1)
              (increase (pass-x ?p) 1)))

(:action move-west-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (and (in-taxi ?p ?t)
                    (forall (?pb - passenger) (or (= ?pb ?p) (not (in-taxi ?pb ?t)))))
 :effect (and (decrease (taxi-x ?t) 1)
              (decrease (pass-x ?p) 1)))

)

;STATE_CONSTRAINT_START
; (
; 	forall (?t - taxi ?p0 ?p1 - passenger) 
; 	(not
; 		(and
; 			(
;     			(in-taxi ?p0 ?t) (in-taxi ?p1 ?t) (not (= ?p0 ?p1))
;     		)
;     	)
;     )
; )
;STATE_CONSTRAINT_END