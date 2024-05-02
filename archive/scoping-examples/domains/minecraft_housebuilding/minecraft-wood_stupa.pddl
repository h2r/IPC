(define (domain minecraft-house)
(:requirements :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)

(:types 
	locatable - object
	agent item block - locatable
	bedrock destructible-block - block
	oak_wood-block oak_wood_stairs-block - destructible-block
	diamond-pickaxe - item
)

(:predicates
	 (present ?i - item)
	 (block-present ?b - block)
	 (agent-alive ?ag - agent)
)

(:functions
	(block-hits ?b - destructible-block)
	(agent-num-diamond-pickaxe ?ag - agent)
	(agent-num-oak_wood-block ?ag - agent)
	(agent-num-oak_wood_stairs-block ?ag - agent)
	(x ?l - locatable)
	(y ?l - locatable)
	(z ?l - locatable)
)

(:action move-north
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (x ?bl) (x ?ag))
                                                    (= (y ?bl) (+ (y ?ag) 1))
                                                    (= (z ?bl) (z ?ag))))))
 :effect (and (increase (y ?ag) 1))
)

(:action move-south
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (x ?bl) (x ?ag))
                                                    (= (y ?bl) (- (y ?ag) 1))
                                                    (= (z ?bl) (z ?ag))))))
 :effect (and (decrease (y ?ag) 1))
)

(:action move-east
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (x ?bl) (+ (x ?ag) 1))
                                                    (= (y ?bl) (y ?ag))
                                                    (= (z ?bl) (z ?ag) )))))
 :effect (and (increase (x ?ag) 1))
)

(:action move-west
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (x ?bl) (- (x ?ag) 1))
                                                    (= (y ?bl) (y ?ag))
                                                    (= (z ?bl) (z ?ag))))))
 :effect (and (decrease (x ?ag) 1))
)

(:action drop-ahead-oak_wood-block
 :parameters (?ag - agent ?b - oak_wood-block)
 :precondition (and (>= (agent-num-oak_wood-block ?ag) 1)
                    (not (block-present ?b))
                    (not (exists (?bl - block)
                        (and
                        (= (x ?ag) (x ?bl)) 
                        (= (+ (y ?ag) 1) (y ?bl))
                        (= (z ?ag) (z ?bl))))))
 :effect (and (block-present ?b)
              (assign (x ?b) (x ?ag))
              (assign (y ?b) (+ (y ?ag) 1))
              (assign (z ?b) (z ?ag))
              (decrease (agent-num-oak_wood-block ?ag) 1)
         )
)


(:action drop-ahead-oak_wood_stairs-block
 :parameters (?ag - agent ?b - oak_wood_stairs-block)
 :precondition (and (>= (agent-num-oak_wood_stairs-block ?ag) 1)
                    (not (block-present ?b))
                    (not (exists (?bl - block)
                        (and
                        (= (x ?ag) (x ?bl)) 
                        (= (+ (y ?ag) 1) (y ?bl))
                        (= (z ?ag) (z ?bl))))))
 :effect (and (block-present ?b)
              (assign (x ?b) (x ?ag))
              (assign (y ?b) (+ (y ?ag) 1))
              (assign (z ?b) (z ?ag))
              (decrease (agent-num-oak_wood_stairs-block ?ag) 1)
         )
)


(:action hit-oak_wood-block
    :parameters (?ag - agent ?b - oak_wood-block)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (block-present ?b)
                        (< (block-hits ?b) 4))
    :effect (and (increase (block-hits ?b) 1))
    )

(:action destroy-oak_wood-block
    :parameters (?ag - agent ?b - oak_wood-block)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (block-present ?b)
                        (= (block-hits ?b) 3))
    :effect (and (not (block-present ?b))
                 (increase (agent-num-oak_wood-block ?ag) 1)
            )
    )

(:action hit-oak_wood_stairs-block
    :parameters (?ag - agent ?b - oak_wood_stairs-block)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (block-present ?b)
                        (< (block-hits ?b) 4))
    :effect (and (increase (block-hits ?b) 1))
    )

(:action destroy-oak_wood_stairs-block
    :parameters (?ag - agent ?b - oak_wood_stairs-block)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (+ (z ?ag) 1))
                        (block-present ?b)
                        (= (block-hits ?b) 3))
    :effect (and (not (block-present ?b))
                 (increase (agent-num-oak_wood_stairs-block ?ag) 1)
            )
    )

)