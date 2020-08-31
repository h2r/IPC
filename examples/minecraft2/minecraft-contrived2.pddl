(define (domain minecraft-contrived)
(:requirements :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)

(:types 
	locatable
	agent item destructible-block - locatable
	obsidian-block - destructible-block
	diamond stick iron diamond-pickaxe shears - item
)

(:predicates
	 present ?i - item
)

(:functions
	( agent-num-diamond ?ag - agent )
	( agent-num-stick ?ag - agent )
	( agent-num-iron ?ag - agent )
	( agent-num-diamond-pickaxe ?ag - agent )
	( agent-num-shears ?ag - agent )
	( x ?l - locatable )
	( y ?l - locatable )
	( z ?l - locatable )
)

(:action move-north
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (x ?ag))
                                                    (= (bl-y ?bl) (+ (y ?ag) 1))
                                                    (= (bl-z ?bl) (+ (z ?ag) 1))))))
 :effect (and (increase (y ?ag) 1))
)

(:action move-south
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (x ?ag))
                                                    (= (bl-y ?bl) (- (y ?ag) 1))
                                                    (= (bl-z ?bl) (+ (z ?ag) 1))))))
 :effect (and (decrease (y ?ag) 1))
)

(:action move-east
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (+ (x ?ag) 1))
                                                    (= (bl-y ?bl) (y ?ag))
                                                    (= (bl-z ?bl) (+ (z ?ag) 1))))))
 :effect (and (increase (x ?ag) 1))
)

(:action move-west
 :parameters (?ag - agent)
 :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (= (bl-x ?bl) (- (x ?ag) 1))
                                                    (= (bl-y ?bl) (y ?ag))
                                                    (= (bl-z ?bl) (+ (z ?ag) 1))))))
 :effect (and (decrease (x ?ag) 1))
)

(:action pickup-diamond
 :parameters (?ag - agent ?i - diamond)
 :precondition (and (present ?i)
                    (= (x ?i) (x ?ag))
                    (= (y ?i) (y ?ag))
                    (= (z ?i) (z ?ag)))
 :effect (and (increase (agent-num-diamond ?ag) 1)
              (not (present ?i)))
)


(:action pickup-stick
 :parameters (?ag - agent ?i - stick)
 :precondition (and (present ?i)
                    (= (x ?i) (x ?ag))
                    (= (y ?i) (y ?ag))
                    (= (z ?i) (z ?ag)))
 :effect (and (increase (agent-num-stick ?ag) 1)
              (not (present ?i)))
)


(:action pickup-iron
 :parameters (?ag - agent ?i - iron)
 :precondition (and (present ?i)
                    (= (x ?i) (x ?ag))
                    (= (y ?i) (y ?ag))
                    (= (z ?i) (z ?ag)))
 :effect (and (increase (agent-num-iron ?ag) 1)
              (not (present ?i)))
)


(:action pickup-diamond-pickaxe
 :parameters (?ag - agent ?i - diamond-pickaxe)
 :precondition (and (present ?i)
                    (= (x ?i) (x ?ag))
                    (= (y ?i) (y ?ag))
                    (= (z ?i) (z ?ag)))
 :effect (and (increase (agent-num-diamond-pickaxe ?ag) 1)
              (not (present ?i)))
)


(:action pickup-shears
 :parameters (?ag - agent ?i - shears)
 :precondition (and (present ?i)
                    (= (x ?i) (x ?ag))
                    (= (y ?i) (y ?ag))
                    (= (z ?i) (z ?ag)))
 :effect (and (increase (agent-num-shears ?ag) 1)
              (not (present ?i)))
)


(:action craft-diamond-pickaxe
    :parameters ( ?ag - agent )
    :precondition ( and
                      ( > (agent-num-stick ?ag) 2 )
                      ( > (agent-num-diamond ?ag) 3 )
                  )
    :effect (and (increase (agent-num-diamond-pickaxe ?ag) 1))

)

(:action craft-shears
    :parameters ( ?ag - agent )
    :precondition ( and
                      ( > (agent-num-iron ?ag) 2 )
                  )
    :effect (and (increase (agent-num-shears ?ag) 1))

)

)