;; Domain constructed by: <redacted for review>

(define (domain infinite_multipasstaxi)
(:requirements :typing :fluents :negative-preconditions)
; (:requirements :typing)

(:types passenger taxi - object)
(:predicates (in-taxi ?p - passenger ?t - taxi) )
(:functions (pass-y ?p - passenger)
            (pass-x ?p - passenger)
            (taxi-x ?t - taxi)
            (taxi-y ?t - taxi))


(:action pickup
 :parameters (?p - passenger ?t - taxi)
 :precondition (and (= (pass-x ?p) (taxi-x ?t)) 
                    (= (pass-y ?p) (taxi-y ?t))
                    (not (in-taxi ?p ?t)))
 :effect (and (in-taxi ?p ?t)))

(:action dropoff
 :parameters (?p - passenger ?t - taxi)
 :precondition (in-taxi ?p ?t)
 :effect (not (in-taxi ?p ?t)))

(:action move-north
 :parameters (?t - taxi ?p - passenger)
 :precondition (not (in-taxi ?p ?t))
 :effect (and (increase (taxi-y ?t) 1)))

(:action move-south
 :parameters (?t - taxi ?p - passenger)
 :precondition (not (in-taxi ?p ?t))
 :effect (and (decrease (taxi-x ?t) 1)))

(:action move-east
 :parameters (?t - taxi ?p - passenger)
 :precondition (not (in-taxi ?p ?t))
 :effect (and (increase (taxi-x ?t) 1)))

(:action move-west
 :parameters (?t - taxi ?p - passenger)
 :precondition (not (in-taxi ?p ?t))
 :effect (and (decrease (taxi-y ?t) 1)))

(:action move-north-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (in-taxi ?p ?t)
 :effect (and (increase (taxi-y ?t) 1)
              (increase (pass-y ?p) 1)))

(:action move-south-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (in-taxi ?p ?t)
 :effect (and (decrease (taxi-y ?t) 1)
              (decrease (pass-y ?p) 1)))

(:action move-east-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (in-taxi ?p ?t)
 :effect (and (increase (taxi-x ?t) 1)
              (increase (pass-x ?p) 1)))

(:action move-west-pass
 :parameters (?t - taxi ?p - passenger)
 :precondition (in-taxi ?p ?t)
 :effect (and (decrease (taxi-x ?t) 1)
              (decrease (pass-x ?p) 1)))

)