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


(define (domain multi_monkeys_playroom_strips)
(:requirements :typing :negative-preconditions :universal-preconditions)

(:types lightswitch x-loc y-loc color - object)
(:predicates (light-on ?lt - lightswitch) 

             (lightswitch-x ?ls - lightswitch ?xl - x-loc)
             (lightswitch-y ?ls - lightswitch ?yl - y-loc)
             (lightswitch-color ?ls - lightswitch ?co - color)

             (eye-x ?xl - x-loc)
             (eye-y ?yl - y-loc)

             (hand-x ?xl - x-loc)
             (hand-y ?yl - y-loc)

             (adjacent-x ?x1 ?x2 - xloc)
             (adjacent-y ?y1 ?y2 - yloc)

)


; Movement actions for the eye
(:action move-y-eye
 :parameters (?ycurr - y-loc ?ynext - y-loc)
 :precondition (and (eye-y ?ycurr)
                    (adjacent-y ?ycurr ?ynext)
                )
 :effect (and (not (eye-y ?ycurr))
              (eye-y ?ynext)
         )
)
(:action move-x-eye
 :parameters (?xcurr - x-loc ?xnext - x-loc)
 :precondition (and (eye-x ?xcurr)
                    (adjacent-x ?xcurr ?xnext)
                )
 :effect (and (not (eye-x ?xcurr))
              (eye-x ?xnext)
         )
)

; Movement actions for the hand
(:action move-y-hand
 :parameters (?ycurr - y-loc ?ynext - y-loc)
 :precondition (and (hand-y ?ycurr)
                    (adjacent-y ?ycurr ?ynext)
                )
 :effect (and (not (hand-y ?ycurr))
              (hand-y ?ynext)
         )
)
(:action move-x-hand
 :parameters (?xcurr - x-loc ?xnext - x-loc)
 :precondition (and (hand-x ?xcurr)
                    (adjacent-x ?xcurr ?xnext)
                )
 :effect (and (not (hand-x ?xcurr))
              (hand-x ?xnext)
         )
)

; Turn on and off lights
(:action turn_on_light
 :parameters (?ls - lightswitch ?xl - x-loc ?yl - y-loc)
 :precondition (and (lightswitch-x ?ls ?xl)
                    (lightswitch-y ?ls ?yl)
                    (hand-x ?xl)
                    (hand-y ?yl)
                    (eye-x ?xl)
                    (eye-y ?yl)
                    (not (light-on ?ls)))
 :effect (light-on ?ls)
)
(:action turn_off_light
 :parameters (?ls - lightswitch ?xl - x-loc ?yl - y-loc)
 :precondition (and (lightswitch-x ?ls ?xl)
                    (lightswitch-y ?ls ?yl)
                    (hand-x ?xl)
                    (hand-y ?yl)
                    (eye-x ?xl)
                    (eye-y ?yl)
                    (light-on ?ls))
 :effect (not (light-on ?ls))
)


(:action change_light_color
 :parameters (?ls - lightswitch ?currco - color ?nextco - color)
 :precondition (lightswitch-color ?ls ?currco)
 :effect (and (lightswitch-color ?ls ?nextco)
              (not (lightswitch-color ?ls ?currco))
         )
)

)