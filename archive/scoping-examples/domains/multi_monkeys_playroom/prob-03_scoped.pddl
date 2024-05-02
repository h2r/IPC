(define (problem MULTIMONKEY-03)
(:domain multi_monkeys_playroom)
(:objects

	jack - monkey
    bell1 - bell
    ball1 - ball
    eye1 - eye
    hand1 - hand
    marker1 - marker
;	rb1 rb2 rb3 - redbutton
;	gb1 gb2 gb3 - greenbutton
;	ls1 ls2 ls3 - lightswitch
)
(:init
    (= (eye-x eye1) 0)
    (= (eye-y eye1) 0)

    (= (hand-x hand1) 0)
    (= (hand-y hand1) 0)

    (= (marker-x marker1) 0)
    (= (marker-y marker1) 0)

;    (= (lightswitch-x ls1) 2)
;    (= (lightswitch-y ls1) 2)
;    (light-on ls1)
;    (= (lightswitch-x ls2) 3)
;    (= (lightswitch-y ls2) 2)
;    (light-on ls2)
;    (= (lightswitch-x ls3) 4)
;    (= (lightswitch-y ls3) 2)
;    (light-on ls3)

;    (= (rbutton-x rb1) 3)
;    (= (rbutton-y rb1) 3)
;    (= (gbutton-x gb1) 3)
;    (= (gbutton-y gb1) 4)
;    (not (rbutton-on rb1))
;    (gbutton-on gb1)
;    (connected-buttons rb1 gb1)
;    (= (rbutton-x rb2) 3)
;    (= (rbutton-y rb2) 5)
;    (= (gbutton-x gb2) 3)
;    (= (gbutton-y gb2) 6)
;    (not (rbutton-on rb2))
;    (gbutton-on gb2)
;    (connected-buttons rb2 gb2)
;    (= (rbutton-x rb3) 3)
;    (= (rbutton-y rb3) 7)
;    (= (gbutton-x gb3) 3)
;    (= (gbutton-y gb3) 8)
;    (not (rbutton-on rb3))
;    (gbutton-on gb3)
;    (connected-buttons rb3 gb3)

    (= (ball-x ball1) 5)
    (= (ball-y ball1) 5)

    (= (bell-x bell1) 80)
    (= (bell-y bell1) 80)
 
    (not (monkey-screaming jack))
    (monkey-watching-bell jack bell1)

)

(:goal (monkey-screaming jack))

)