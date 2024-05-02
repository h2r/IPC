(define (problem MULTIMONKEY-01)
(:domain multi_monkeys_playroom_strips)
(:objects
	; jack - monkey
    ; bell1 - bell
    ; ball1 - ball
    ; rb1 - redbutton
    ; gb1 - greenbutton
    ls1 ls2 ls3 - lightswitch
    x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 - x-loc
    y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 - y-loc
    red blue green white black yellow - color
)
(:init
    ; establishing adjacency
    (adjacent-x x1 x2)
    (adjacent-x x2 x1)
    (adjacent-x x2 x3)
    (adjacent-x x3 x2)
    (adjacent-x x3 x4)
    (adjacent-x x4 x3)
    (adjacent-x x4 x5)
    (adjacent-x x5 x4)
    (adjacent-x x5 x6)
    (adjacent-x x6 x5)
    (adjacent-x x6 x7)
    (adjacent-x x7 x6)
    (adjacent-x x7 x8)
    (adjacent-x x8 x7)
    (adjacent-x x8 x9)
    (adjacent-x x9 x8)
    (adjacent-x x9 x10)
    (adjacent-x x10 x9)
    (adjacent-y y1 y2)
    (adjacent-y y2 y1)
    (adjacent-y y2 y3)
    (adjacent-y y3 y2)
    (adjacent-y y3 y4)
    (adjacent-y y4 y3)
    (adjacent-y y4 y5)
    (adjacent-y y5 y4)
    (adjacent-y y5 y6)
    (adjacent-y y6 y5)
    (adjacent-y y6 y7)
    (adjacent-y y7 y6)
    (adjacent-y y7 y8)
    (adjacent-y y8 y7)
    (adjacent-y y8 y9)
    (adjacent-y y9 y8)
    (adjacent-y y9 y10)
    (adjacent-y y10 y9)
    
    ; eye initial locations    
    (eye-x x1)
    (eye-y y1)

    ; hand initial locations
    (hand-x x1)
    (hand-y y1)

    ; lightswitch initial locations
    (lightswitch-x ls1 x8)
    (lightswitch-y ls1 y1)
    (light-on ls1)

    (lightswitch-x ls2 x9)
    (lightswitch-y ls2 y1)
    (not (light-on ls2))

    (lightswitch-x ls3 x9)
    (lightswitch-y ls3 y1)
    (not (light-on ls3))

    (lightswitch-color ls1 red)
    (lightswitch-color ls2 blue)
    (lightswitch-color ls3 green)

    ; red and green buttons initial conditions
    ; (rbutton-x rb1 x3)
    ; (rbutton-y rb1 y3)
    ; (gbutton-x gb1 x3)
    ; (gbutton-y gb1 y4)
    ; (not (gbutton-on gb1))
    ; (rbutton-on rb1)
    ; (connected-buttons rb1 gb1)

    ; ; monkey initial conditions
    ; (not (monkey-screaming jack))
    ; (monkey-watching-lights jack ls1)

)

(:goal (and (not (light-on ls1))
            (lightswitch-color ls1 red)
        )
    ; (and (eye-x x10)
    ;         (eye-y y10)
    ;         (hand-x x10)
    ;         (hand-y y10)
    ; )
)

)