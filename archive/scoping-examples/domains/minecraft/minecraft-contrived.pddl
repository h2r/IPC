;; Domain constructed by: <redacted for review>

; IMPORTANT: Haven't specified a way for the agent to die or pick stuff up
; yet. This can totally be hacked around using different actions with preconditions
; differing in whether there's lava/pick-uppable items ahead

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;Pickup
;       ALL items are picked up when walked over. This would be implemented 
;       via a foreach-expansion, but few planners support this, and ENHSP is no exception
;       Instead, we will assume that no location holds more than 1 item
;       Our scoper/parser probably would not handle the foreach version very well,
;       since we would have 2^(n items) move actions in each direction. There is probably
;       a good way to handle this in the future.
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (domain minecraft-contrived)
(:requirements :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)

(:types bedrock-block dirt-block redstone-block glass-block - block
        agent 
        apple potato rabbit diamond-axe orchid-flower daisy-flower - item
        
)

(:predicates (block-present ?b - block)
             (apple-present ?ap - apple)
             (orchid-present ?orc - orchid-flower)
             (daisy-present ?df - daisy-flower)
             (potato-present ?po - potato)
             (rabbit-present ?ra - rabbit)
             (rabbit-cooked ?ra - rabbit)
             (agent-alive ?ag - agent)
             (agent-has-pickaxe ?ag - agent)
)

(:functions (agent-x ?ag - agent)
            (agent-y ?ag - agent)
            (agent-z ?ag - agent)
            (agent-num-apples ?ag - agent)
            (agent-num-potatoes ?ag - agent)
            (agent-num-orchids ?ag - agent)
            (agent-num-daisies ?ag - agent)
            (agent-num-raw-rabbits ?ag - agent)
            (agent-num-cooked-rabbits ?ag - agent)

            (bl-x ?bl - block)
            (bl-y ?bl - block)
            (bl-z ?bl - block)

            (gb-hits ?gb - glass-block)

            (db-hits ?db - dirt-block)

            (apple-x ?ap - apple)
            (apple-y ?ap - apple)
            (apple-z ?ap - apple)

            (potato-x ?po - potato)
            (potato-y ?po - potato)
            (potato-z ?po - potato)

            (daisy-x ?df - daisy-flower)
            (daisy-y ?df - daisy-flower)
            (daisy-z ?df - daisy-flower)

            (rabbit-x ?ra - rabbit)
            (rabbit-y ?ra - rabbit)
            (rabbit-z ?ra - rabbit)
)

; Note: There actually cannot be a block at agent-z, agent-z + 1 or agent-z + 2 for any of the move actions
; Right now, I've only implemented agent-z (haven't done "or" condition yet)
; Also, blocks are only destroyed by hand (haven't allowed more easy destruction via pickaxe)
; Also, only blocks at the same z can be destroyed

(:action move-north
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (agent-x ?ag))
                                                    (= (bl-y ?bl) (+ (agent-y ?ag) 1))
                                                    (= (bl-z ?bl) (+ (agent-z ?ag) 1))))))
 :effect (and (increase (agent-y ?ag) 1))
)

(:action move-south
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (agent-x ?ag))
                                                    (= (bl-y ?bl) (- (agent-y ?ag) 1))
                                                    (= (bl-z ?bl) (+ (agent-z ?ag) 1))))))
 :effect (and (decrease (agent-y ?ag) 1))
)

(:action move-east
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (+ (agent-x ?ag) 1))
                                                    (= (bl-y ?bl) (agent-y ?ag))
                                                    (= (bl-z ?bl) (+ (agent-z ?ag) 1))))))
 :effect (and (increase (agent-x ?ag) 1))
)

(:action move-west
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (- (agent-x ?ag) 1))
                                                    (= (bl-y ?bl) (agent-y ?ag))
                                                    (= (bl-z ?bl) (+ (agent-z ?ag) 1))))))
 :effect (and (decrease (agent-x ?ag) 1))
)

(:action hit-glass-block
 :parameters (?ag - agent ?gb - glass-block)
 :precondition (and (= (bl-x ?gb) (agent-x ?ag))
                    (= (bl-y ?gb) (+ (agent-y ?ag) 1))
                    (= (bl-z ?gb) (agent-z ?ag))
                    (block-present ?gb)
                    (< (gb-hits ?gb) 7))
:effect (and (increase (gb-hits ?gb) 1))
)

(:action destroy-glass-block
 :parameters (?ag - agent ?gb - glass-block)
 :precondition (and (= (bl-x ?gb) (agent-x ?ag))
                    (= (bl-y ?gb) (+ (agent-y ?ag) 1))
                    (= (bl-z ?gb) (agent-z ?ag))
                    (block-present ?gb)
                    (= (gb-hits ?gb) 6))
:effect (not (block-present ?gb))
)

(:action hit-dirt-block
 :parameters (?ag - agent ?db - dirt-block)
 :precondition (and (= (bl-x ?db) (agent-x ?ag))
                    (= (bl-y ?db) (+ (agent-y ?ag) 1))
                    (= (bl-z ?db) (agent-z ?ag))
                    (block-present ?db)
                    (< (db-hits ?db) 4))
:effect (and (increase (db-hits ?db) 1))
)

(:action destroy-dirt-block
 :parameters (?ag - agent ?db - dirt-block)
 :precondition (and (= (bl-x ?db) (agent-x ?ag))
                    (= (bl-y ?db) (+ (agent-y ?ag) 1))
                    (= (bl-z ?db) (agent-z ?ag))
                    (block-present ?db)
                    (= (db-hits ?db) 3))
:effect (not (block-present ?db))
)

(:action pickup-apple
 :parameters (?ag - agent ?ap - apple)
 :precondition (and (apple-present ?ap)
                    (= (apple-x ?ap) (agent-x ?ag))
                    (= (apple-y ?ap) (agent-y ?ag))
                    (= (apple-z ?ap) (agent-z ?ag)))
 :effect (and (increase (agent-num-apples ?ag) 1)
              (not (apple-present ?ap)))
)

(:action dropoff-apple
 :parameters (?ag - agent ?ap - apple)
 :precondition (and (not (apple-present ?ap))
                    (> (agent-num-apples ?ag) 0))
 :effect (and (decrease (agent-num-apples ?ag) 1)
              (apple-present ?ap)
              (assign (apple-x ?ap) (agent-x ?ag))
              (assign (apple-y ?ap) (agent-y ?ag))
              (assign (apple-z ?ap) (agent-z ?ag)))
)

(:action pickup-potato
 :parameters (?ag - agent ?po - potato)
 :precondition (and (potato-present ?po)
                    (= (potato-x ?po) (agent-x ?ag))
                    (= (potato-y ?po) (agent-y ?ag))
                    (= (potato-z ?po) (agent-z ?ag)))
 :effect (and (increase (agent-num-potatoes ?ag) 1)
              (not (potato-present ?po)))
)

(:action dropoff-potato
 :parameters (?ag - agent ?po - potato)
 :precondition (and (not (potato-present ?po))
                    (> (agent-num-potatoes ?ag) 0))
 :effect (and (decrease (agent-num-potatoes ?ag) 1)
              (potato-present ?po)
              (assign (potato-x ?po) (agent-x ?ag))
              (assign (potato-y ?po) (agent-y ?ag))
              (assign (potato-z ?po) (agent-z ?ag)))
)

(:action pickup-raw-rabbit
 :parameters (?ag - agent ?rb - rabbit)
 :precondition (and (rabbit-present ?rb)
                    (not (rabbit-cooked ?rb))
                    (= (rabbit-x ?rb) (agent-x ?ag))
                    (= (rabbit-y ?rb) (agent-y ?ag))
                    (= (rabbit-z ?rb) (agent-z ?ag)))
 :effect (and (increase (agent-num-raw-rabbits ?ag) 1)
              (not (rabbit-present ?rb)))
)

(:action dropoff-raw-rabbit
 :parameters (?ag - agent ?rb - rabbit)
 :precondition (and (not (rabbit-present ?rb))
                    (> (agent-num-raw-rabbits ?ag) 0))
 :effect (and (decrease (agent-num-raw-rabbits ?ag) 1)
              (rabbit-present ?rb)
              (assign (rabbit-x ?rb) (agent-x ?ag))
              (assign (rabbit-y ?rb) (agent-y ?ag))
              (assign (rabbit-z ?rb) (agent-z ?ag)))
)

(:action pickup-cooked-rabbit
 :parameters (?ag - agent ?rb - rabbit)
 :precondition (and (rabbit-present ?rb)
                    (rabbit-cooked ?rb)
                    (= (rabbit-x ?rb) (agent-x ?ag))
                    (= (rabbit-y ?rb) (agent-y ?ag))
                    (= (rabbit-z ?rb) (agent-z ?ag)))
 :effect (and (increase (agent-num-cooked-rabbits ?ag) 1)
              (not (rabbit-present ?rb)))
)

(:action dropoff-cooked-rabbit
 :parameters (?ag - agent ?rb - rabbit)
 :precondition (and (not (rabbit-present ?rb))
                    (> (agent-num-cooked-rabbits ?ag) 0))
 :effect (and (decrease (agent-num-cooked-rabbits ?ag) 1)
              (rabbit-present ?rb)
              (assign (rabbit-x ?rb) (agent-x ?ag))
              (assign (rabbit-y ?rb) (agent-y ?ag))
              (assign (rabbit-z ?rb) (agent-z ?ag)))
)

(:action cook-rabbit
    :parameters (?ag - agent ?rb - rabbit)
    :precondition (> (agent-num-raw-rabbits ?ag) 0)
    :effect (and (decrease (agent-num-raw-rabbits ?ag) 1)
                 (increase (agent-num-cooked-rabbits ?ag) 1))
)

)