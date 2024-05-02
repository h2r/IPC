;; Domain constructed by: <redacted for review>

; How does this domain work?
; There are some number of lightswitches, red buttons and green buttons. 
; To interact with any of them, the eye and hand must be over them, and then 
; the relevant interact action must be called.
; All light switches must be turned on in order to interact with either of the 
; buttons
; Red and green buttons are 'connected'. You can only press a red button if the 
; green button is on and vice-versa. When a green button is on, music is played, and 
; turning a red button on turns the music off and also toggles the green button
; Once all green buttons are on, the agent's eye and hand can pick up balls and throw them
; Throwing a ball causes it to go to the x-y position of the agent's marker
; If there is a bell at that x-y position, the bell will be rung
; Bells are being watched by monkeys. When a specific bell is rung, its connected monkey 
; will scream.
; The goal of this domain is to get a specific monkey to scream.

(define (domain multi_monkeys_playroom)
(:requirements :typing :fluents :negative-preconditions :universal-preconditions)

(:types eye hand marker ball bell redbutton greenbutton lightswitch monkey - object)
(:predicates (rbutton-on ?rb - redbutton)
             (gbutton-on ?gb - greenbutton)
             (light-on ?lt - lightswitch)
             (monkey-screaming ?mo - monkey)
             (monkey-watching-bell ?mo - monkey ?be - bell)
             (connected-buttons ?rb - redbutton ?gb - greenbutton))

(:functions (rbutton-x ?rb - redbutton)
            (rbutton-y ?rb - redbutton)

            (gbutton-x ?gb - greenbutton)
            (gbutton-y ?gb - greenbutton)

            (lightswitch-x ?ls - lightswitch)
            (lightswitch-y ?ls - lightswitch)

            (eye-x ?e - eye)
            (eye-y ?e - eye)

            (hand-x ?ha - hand)
            (hand-y ?ha - hand)

            (marker-x ?ma - marker)
            (marker-y ?ma - marker)

            (ball-x ?ba - ball)
            (ball-y ?ba - ball)

            (bell-x ?be - bell)
            (bell-y ?be - bell)

)

; Movement actions for the eye
(:action move-north-eye
 :parameters (?e - eye)
 :precondition ()
 :effect (and (increase (eye-y ?e) 1))
)
(:action move-south-eye
 :parameters (?e - eye)
 :precondition ()
 :effect (and (decrease (eye-y ?e) 1))
)
(:action move-east-eye
 :parameters (?e - eye)
 :precondition ()
 :effect (and (increase (eye-x ?e) 1))
)
(:action move-west-eye
 :parameters (?e - eye)
 :precondition ()
 :effect (and (decrease (eye-x ?e) 1))
)

; Movement actions for the hand
(:action move-north-hand
 :parameters (?ha - hand)
 :precondition ()
 :effect (and (increase (hand-y ?ha) 1))
)
(:action move-south-hand
 :parameters (?ha - hand)
 :precondition ()
 :effect (and (decrease (hand-y ?ha) 1))
)
(:action move-east-hand
 :parameters (?ha - hand)
 :precondition ()
 :effect (and (increase (hand-x ?ha) 1))
)
(:action move-west-hand
 :parameters (?ha - hand)
 :precondition ()
 :effect (and (decrease (hand-x ?ha) 1))
)

; Movement actions for the marker
(:action move-north-marker
 :parameters (?ma - marker)
 :precondition ()
 :effect (and (increase (marker-y ?ma) 1))
)
(:action move-south-marker
 :parameters (?ma - marker)
 :precondition ()
 :effect (and (decrease (marker-y ?ma) 1))
)
(:action move-east-marker
 :parameters (?ma - marker)
 :precondition ()
 :effect (and (increase (marker-x ?ma) 1))
)
(:action move-west-marker
 :parameters (?ma - marker)
 :precondition ()
 :effect (and (decrease (marker-x ?ma) 1))
)

; Turn on and off lights
(:action turn_on_light
 :parameters (?ls - lightswitch ?ha - hand ?e - eye)
 :precondition (and (= (lightswitch-x ?ls) (hand-x ?ha))
                    (= (lightswitch-y ?ls) (hand-y ?ha))
                    (= (lightswitch-x ?ls) (eye-x ?e))
                    (= (lightswitch-y ?ls) (eye-y ?e))
                    (not (light-on ?ls)))
 :effect (light-on ?ls)
)
(:action turn_off_light
 :parameters (?ls - lightswitch ?ha - hand ?e - eye)
 :precondition (and (= (lightswitch-x ?ls) (hand-x ?ha))
                    (= (lightswitch-y ?ls) (hand-y ?ha))
                    (= (lightswitch-x ?ls) (eye-x ?e))
                    (= (lightswitch-y ?ls) (eye-y ?e))
                    (light-on ?ls))
 :effect (not (light-on ?ls))
)

; Toggle red and green buttons
(:action turn_on_redbutton
 :parameters (?rb - redbutton ?gb - greenbutton ?ha - hand ?e - eye)
 :precondition (and (forall (?ls - lightswitch) (light-on ?ls) )
                    (= (rbutton-x ?rb) (hand-x ?ha))
                    (= (rbutton-y ?rb) (hand-y ?ha))
                    (= (rbutton-x ?rb) (eye-x ?e))
                    (= (rbutton-y ?rb) (eye-y ?e))
                    (connected-buttons ?rb ?gb)
                    (not (rbutton-on ?rb))
                    (gbutton-on ?gb))
 :effect (and (rbutton-on ?rb)
              (not (gbutton-on ?gb)))
)
(:action turn_on_greenbutton
 :parameters (?rb - redbutton ?gb - greenbutton ?ha - hand ?e - eye)
 :precondition (and (forall (?ls - lightswitch) (light-on ?ls) )
                    (= (gbutton-x ?gb) (hand-x ?ha))
                    (= (gbutton-y ?gb) (hand-y ?ha))
                    (= (gbutton-x ?gb) (eye-x ?e))
                    (= (gbutton-y ?gb) (eye-y ?e))
                    (connected-buttons ?rb ?gb)
                    (not (gbutton-on ?gb))
                    (rbutton-on ?rb))
 :effect (and (gbutton-on ?gb)
              (not (rbutton-on ?rb)))
)

; Throw balls
(:action throw_ball_miss_bell
    :parameters (?ba - ball ?e - eye ?ha - hand ?ma - marker ?be - bell)
    :precondition (and (forall (?gb - greenbutton) (gbutton-on ?gb))
                       (= (eye-x ?e) (ball-x ?ba))
                       (= (eye-y ?e) (ball-y ?ba))
                       (= (hand-x ?ha) (ball-x ?ba))
                       (= (hand-y ?ha) (ball-y ?ba))
                       (not (= (marker-x ?ma) (bell-x ?be)))
                       (not (= (marker-y ?ma) (bell-y ?be))))
    :effect (and (assign (ball-x ?ba) (marker-x ?ma))
                 (assign (ball-y ?ba) (marker-y ?ma)))
)
(:action throw_ball_hit_bell
:parameters (?ba - ball ?e - eye ?ha - hand ?ma - marker ?be - bell ?mo - monkey)
:precondition (and (forall (?gb - greenbutton) (gbutton-on ?gb))
                    (= (eye-x ?e) (ball-x ?ba))
                    (= (eye-y ?e) (ball-y ?ba))
                    (= (hand-x ?ha) (ball-x ?ba))
                    (= (hand-y ?ha) (ball-y ?ba))
                    (= (marker-x ?ma) (bell-x ?be))
                    (= (marker-y ?ma) (bell-y ?be))
                    (monkey-watching-bell ?mo ?be))
:effect (and (assign (ball-x ?ba) (marker-x ?ma))
             (assign (ball-y ?ba) (marker-y ?ma))
             (monkey-screaming ?mo))
)

)