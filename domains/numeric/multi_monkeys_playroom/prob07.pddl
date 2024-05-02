(define (problem MULTIMONKEY-07)
(:domain multi_monkeys_playroom)
(:objects

	jack - monkey
    bell1 - bell
    ball1 - ball
    eye1 - eye
    hand1 - hand
    marker1 - marker
    rb1 rb2 rb3 rb4 rb5 rb6 rb7 - redbutton
    gb1 gb2 gb3 gb4 gb5 gb6 gb7 - greenbutton
    ls1 ls2 ls3 ls4 ls5 ls6 ls7 - lightswitch
)
(:init
    (= (eye-x eye1) 0)
    (= (eye-y eye1) 0)

    (= (hand-x hand1) 0)
    (= (hand-y hand1) 0)

    (= (marker-x marker1) 0)
    (= (marker-y marker1) 0)

    (= (lightswitch-x ls1) 2)
    (= (lightswitch-y ls1) 2)
    (light-on ls1)
    (= (lightswitch-x ls2) 3)
    (= (lightswitch-y ls2) 2)
    (light-on ls2)
    (= (lightswitch-x ls3) 4)
    (= (lightswitch-y ls3) 2)
    (light-on ls3)
    (= (lightswitch-x ls4) 5)
    (= (lightswitch-y ls4) 2)
    (light-on ls4)
    (= (lightswitch-x ls5) 6)
    (= (lightswitch-y ls5) 2)
    (light-on ls5)
    (= (lightswitch-x ls6) 7)
    (= (lightswitch-y ls6) 2)
    (light-on ls6)
    (= (lightswitch-x ls7) 8)
    (= (lightswitch-y ls7) 2)
    (light-on ls7)

    (= (rbutton-x rb1) 3)
    (= (rbutton-y rb1) 3)
    (= (gbutton-x gb1) 3)
    (= (gbutton-y gb1) 4)
    (not (rbutton-on rb1))
    (gbutton-on gb1)
    (connected-buttons rb1 gb1)
    (= (rbutton-x rb2) 3)
    (= (rbutton-y rb2) 5)
    (= (gbutton-x gb2) 3)
    (= (gbutton-y gb2) 6)
    (not (rbutton-on rb2))
    (gbutton-on gb2)
    (connected-buttons rb2 gb2)
    (= (rbutton-x rb3) 3)
    (= (rbutton-y rb3) 7)
    (= (gbutton-x gb3) 3)
    (= (gbutton-y gb3) 8)
    (not (rbutton-on rb3))
    (gbutton-on gb3)
    (connected-buttons rb3 gb3)
    (= (rbutton-x rb4) 4)
    (= (rbutton-y rb4) 7)
    (= (gbutton-x gb4) 4)
    (= (gbutton-y gb4) 8)
    (not (rbutton-on rb4))
    (gbutton-on gb4)
    (connected-buttons rb4 gb4)
    (= (rbutton-x rb5) 5)
    (= (rbutton-y rb5) 7)
    (= (gbutton-x gb5) 5)
    (= (gbutton-y gb5) 8)
    (not (rbutton-on rb5))
    (gbutton-on gb5)
    (connected-buttons rb5 gb5)
    (= (rbutton-x rb6) 6)
    (= (rbutton-y rb6) 7)
    (= (gbutton-x gb6) 6)
    (= (gbutton-y gb6) 8)
    (not (rbutton-on rb6))
    (gbutton-on gb6)
    (connected-buttons rb6 gb6)
    (= (rbutton-x rb7) 7)
    (= (rbutton-y rb7) 7)
    (= (gbutton-x gb7) 7)
    (= (gbutton-y gb7) 8)
    (not (rbutton-on rb7))
    (gbutton-on gb7)
    (connected-buttons rb7 gb7)

    (= (ball-x ball1) 5)
    (= (ball-y ball1) 5)

    (= (bell-x bell1) 10)
    (= (bell-y bell1) 10)

    (not (monkey-screaming jack))
    (monkey-watching-bell jack bell1)

)

(:goal (monkey-screaming jack))

)
