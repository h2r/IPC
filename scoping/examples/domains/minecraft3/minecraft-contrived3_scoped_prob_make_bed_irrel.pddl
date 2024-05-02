(define (domain minecraft-bedmaking)
(:requirements :equality :typing :fluents :negative-preconditions :universal-preconditions :existential-preconditions)

(:types 
	locatable - object
	agent item block - locatable
	bedrock destructible-block - block
	wooden-block wooden-planks wool-block bed - destructible-block
	destructible-item diamond stick diamond-axe blue-dye - item
	orchid-flower oak-sapling birch-sapling - destructible-item
)

(:predicates
	 (present ?i - item)
	 (block-present ?b - block)
	 (agent-alive ?ag - agent)
)

(:functions
	(agent-num-diamond ?ag - agent)
	(agent-num-stick ?ag - agent)
	(agent-num-diamond-axe ?ag - agent)
	(agent-num-blue-dye ?ag - agent)
	(agent-num-orchid-flower ?ag - agent)
	(agent-num-oak-sapling ?ag - agent)
	(agent-num-birch-sapling ?ag - agent)
	(item-hits ?it - destructible-item)
	(agent-num-wooden-block ?ag - agent)
	(agent-num-wooden-planks ?ag - agent)
	(agent-num-wool-block ?ag - agent)
	(agent-num-bed ?ag - agent)
	(block-hits ?b - destructible-block)
	(wool-color ?woolb - wool-block)
	(bed-color ?bd - bed)
	(x ?l - locatable)
	(y ?l - locatable)
	(z ?l - locatable)
)

(:action move-north
 :parameters (?ag - agent)
            :precondition (and (agent-alive ?ag)
                         (not (exists (?bl - block) (and (block-present ?bl) 
                                                         (= (x ?bl) (x ?ag))
                                                         (= (y ?bl) (+ (y ?ag) 1))
                                                         (= (z ?bl) (z ?ag)))))) 
            :effect (and (increase (y ?ag) 1))
)
                        
(:action move-south 
:parameters (?ag - agent) 
:precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (block-present ?bl)
                                                    (= (x ?bl) (x ?ag))
                                                    (= (y ?bl) (- (y ?ag) 1))
                                                    (= (z ?bl) (z ?ag)))))) 
:effect (and (decrease (y ?ag) 1))
)
            
(:action move-east
    :parameters (?ag - agent)
    :precondition (and (agent-alive ?ag)
                        (not (exists (?bl - block) (and (block-present ?bl)
                                                        (= (x ?bl) (+ (x ?ag) 1))
                                                        (= (y ?bl) (y ?ag))
                                                        (= (z ?bl) (z ?ag)))))) 
    :effect (and (increase (x ?ag) 1))
    )
             
(:action move-west
    :parameters (?ag - agent)
    :precondition (and (agent-alive ?ag)
                    (not (exists (?bl - block) (and (block-present ?bl)
                                                    (= (x ?bl) (- (x ?ag) 1))
                                                    (= (y ?bl) (y ?ag))
                                                    (= (z ?bl) (z ?ag))))))
    :effect (and (decrease (x ?ag) 1))
)

;(:action pickup-diamond
; :parameters (?ag - agent ?i - diamond)
; :precondition (and (present ?i)
;                    (= (x ?i) (x ?ag))
;                    (= (y ?i) (y ?ag))
;                    (= (z ?i) (z ?ag)))
; :effect (and (increase (agent-num-diamond ?ag) 1)
;              (not (present ?i)))
;)


;(:action pickup-stick
; :parameters (?ag - agent ?i - stick)
; :precondition (and (present ?i)
;                    (= (x ?i) (x ?ag))
;                    (= (y ?i) (y ?ag))
;                    (= (z ?i) (z ?ag)))
; :effect (and (increase (agent-num-stick ?ag) 1)
;              (not (present ?i)))
;)


;(:action pickup-diamond-axe
; :parameters (?ag - agent ?i - diamond-axe)
; :precondition (and (present ?i)
;                    (= (x ?i) (x ?ag))
;                    (= (y ?i) (y ?ag))
;                    (= (z ?i) (z ?ag)))
; :effect (and (increase (agent-num-diamond-axe ?ag) 1)
;              (not (present ?i)))
;)


;(:action pickup-blue-dye
; :parameters (?ag - agent ?i - blue-dye)
; :precondition (and (present ?i)
;                    (= (x ?i) (x ?ag))
;                    (= (y ?i) (y ?ag))
;                    (= (z ?i) (z ?ag)))
; :effect (and (increase (agent-num-blue-dye ?ag) 1)
;              (not (present ?i)))
;)


;(:action drop-diamond
; :parameters (?ag - agent ?i - diamond)
; :precondition (and (>= (agent-num-diamond ?ag) 1)
;                    (not (present ?i)))
; :effect (and (present ?i)
;              (assign (x ?i) (x ?ag))
;              (assign (y ?i) (+ (y ?ag) 1))
;              (assign (z ?i) (z ?ag))
;              (decrease (agent-num-diamond ?ag) 1)
;         )
;)


;(:action drop-stick
; :parameters (?ag - agent ?i - stick)
; :precondition (and (>= (agent-num-stick ?ag) 1)
;                    (not (present ?i)))
; :effect (and (present ?i)
;              (assign (x ?i) (x ?ag))
;              (assign (y ?i) (+ (y ?ag) 1))
;              (assign (z ?i) (z ?ag))
;              (decrease (agent-num-stick ?ag) 1)
;         )
;)


;(:action drop-blue-dye
; :parameters (?ag - agent ?i - blue-dye)
; :precondition (and (>= (agent-num-blue-dye ?ag) 1)
;                    (not (present ?i)))
; :effect (and (present ?i)
;              (assign (x ?i) (x ?ag))
;              (assign (y ?i) (+ (y ?ag) 1))
;              (assign (z ?i) (z ?ag))
;              (decrease (agent-num-blue-dye ?ag) 1)
;         )
;)


(:action drop-wooden-block
 :parameters (?ag - agent ?b - wooden-block)
 :precondition (and (>= (agent-num-wooden-block ?ag) 1)
                    (not (block-present ?b)))
 :effect (and (block-present ?b)
              (assign (x ?b) (x ?ag))
              (assign (y ?b) (+ (y ?ag) 1))
              (assign (z ?b) (z ?ag))
              (decrease (agent-num-wooden-block ?ag) 1)
         )
)


;(:action drop-wooden-planks
; :parameters (?ag - agent ?b - wooden-planks)
; :precondition (and (>= (agent-num-wooden-planks ?ag) 1)
;                    (not (block-present ?b)))
; :effect (and (block-present ?b)
;              (assign (x ?b) (x ?ag))
;              (assign (y ?b) (+ (y ?ag) 1))
;              (assign (z ?b) (z ?ag))
;              (decrease (agent-num-wooden-planks ?ag) 1)
;         )
;)


(:action drop-wool-block
 :parameters (?ag - agent ?b - wool-block)
 :precondition (and (>= (agent-num-wool-block ?ag) 1)
                    (not (block-present ?b)))
 :effect (and (block-present ?b)
              (assign (x ?b) (x ?ag))
              (assign (y ?b) (+ (y ?ag) 1))
              (assign (z ?b) (z ?ag))
              (decrease (agent-num-wool-block ?ag) 1)
         )
)


(:action drop-bed
 :parameters (?ag - agent ?b - bed)
 :precondition (and (>= (agent-num-bed ?ag) 1)
                    (not (block-present ?b)))
 :effect (and (block-present ?b)
              (assign (x ?b) (x ?ag))
              (assign (y ?b) (+ (y ?ag) 1))
              (assign (z ?b) (z ?ag))
              (decrease (agent-num-bed ?ag) 1)
         )
)


(:action apply-blue-dye
 :parameters (?ag - agent ?woolb - wool-block)
 :precondition (and (not (block-present ?woolb)) (>= (agent-num-wool-block ?ag) 1) (>= (agent-num-blue-dye ?ag) 1))
 :effect (and (decrease (agent-num-blue-dye ?ag) 1) (assign (wool-color ?woolb) 1)))



(:action craft-bed-blue-dye
 :parameters (?ag - agent ?woolb1 - wool-block ?woolb2 - wool-block ?woolb3 - wool-block ?bd - bed)
 :precondition (and (not (block-present ?woolb1)) (not (block-present ?woolb2)) (not (block-present ?woolb3)) 
                (= (wool-color ?woolb1) 1) (= (wool-color ?woolb2) 1) (= (wool-color ?woolb3) 1) 
                (not (= ?woolb1 ?woolb2)) (not (= ?woolb1 ?woolb3)) (not (= ?woolb2 ?woolb3))
                (not (block-present ?bd)) (>= (agent-num-wool-block ?ag) 3) (>= (agent-num-wooden-planks ?ag) 3))
 :effect (and (decrease (agent-num-wooden-planks ?ag) 3) (decrease (agent-num-wool-block ?ag) 3) (increase (agent-num-bed ?ag) 1) (assign (bed-color ?bd) 1)))



;(:action craft-diamond-axe
;    :parameters ( ?ag - agent )
;    :precondition ( and
;                      ( >= (agent-num-stick ?ag) 2 )
;                      ( >= (agent-num-diamond ?ag) 3 )
;                  )
;    :effect (and (increase (agent-num-diamond-axe ?ag) 1)
;        (decrease (agent-num-stick ?ag) 2)
;        (decrease (agent-num-diamond ?ag) 3))
;
;)

(:action craft-wooden-planks
 :parameters (?ag - agent ?wb - wooden-block)
 :precondition (and (not (block-present ?wb)) (>= (agent-num-wooden-block ?ag) 1) )
 :effect (and (decrease (agent-num-wooden-block ?ag) 1) (increase (agent-num-wooden-planks ?ag) 4)))

(:action craft-blue-dye
    :parameters ( ?ag - agent )
    :precondition ( and
                      ( >= (agent-num-orchid-flower ?ag) 1 )
                  )
    :effect (and (increase (agent-num-blue-dye ?ag) 1)
        (decrease (agent-num-orchid-flower ?ag) 1))

)

(:action hit-wooden-block
    :parameters (?ag - agent ?b - wooden-block)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (z ?ag))
                        (block-present ?b)
                        (< (block-hits ?b) 2)
                        ( >= ( agent-num-diamond-axe ?ag ) 1 ))
    :effect (and (increase (block-hits ?b) 1))
    )

(:action destroy-wooden-block
    :parameters (?ag - agent ?b - wooden-block)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (z ?ag))
                        (block-present ?b)
                        (= (block-hits ?b) 2)
                        ( >= ( agent-num-diamond-axe ?ag ) 1 ))
    :effect (and (not (block-present ?b))
                 (increase (agent-num-wooden-block ?ag) 1)
                 (assign (block-hits ?b) 0)
            )
    )

;(:action hit-wooden-planks
;    :parameters (?ag - agent ?b - wooden-planks)
;    :precondition (and (= (x ?b) (x ?ag))
;                        (= (y ?b) (+ (y ?ag) 1))
;                        (= (z ?b) (z ?ag))
;                        (block-present ?b)
;                        (< (block-hits ?b) 2)
;                        ( >= ( agent-num-diamond-axe ?ag ) 1 ))
;    :effect (and (increase (block-hits ?b) 1))
;    )

;(:action destroy-wooden-planks
;    :parameters (?ag - agent ?b - wooden-planks)
;    :precondition (and (= (x ?b) (x ?ag))
;                        (= (y ?b) (+ (y ?ag) 1))
;                        (= (z ?b) (z ?ag))
;                        (block-present ?b)
;                        (= (block-hits ?b) 2)
;                        ( >= ( agent-num-diamond-axe ?ag ) 1 ))
;    :effect (and (not (block-present ?b))
;                 (increase (agent-num-wooden-planks ?ag) 1)
;                 (assign (block-hits ?b) 0)
;            )
;    )

(:action hit-bed
    :parameters (?ag - agent ?b - bed)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (z ?ag))
                        (block-present ?b)
                        (< (block-hits ?b) 2)
                        ( >= ( agent-num-diamond-axe ?ag ) 1 ))
    :effect (and (increase (block-hits ?b) 1))
    )

(:action destroy-bed
    :parameters (?ag - agent ?b - bed)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (z ?ag))
                        (block-present ?b)
                        (= (block-hits ?b) 2)
                        ( >= ( agent-num-diamond-axe ?ag ) 1 ))
    :effect (and (not (block-present ?b))
                 (increase (agent-num-bed ?ag) 1)
                 (assign (block-hits ?b) 0)
            )
    )

(:action destroy-orchid-flower
    :parameters (?ag - agent ?b - orchid-flower)
    :precondition (and (= (x ?b) (x ?ag))
                        (= (y ?b) (+ (y ?ag) 1))
                        (= (z ?b) (z ?ag))
                        (present ?b)
                        (= (item-hits ?b) 0))
    :effect (and (not (present ?b))
                 (increase (agent-num-orchid-flower ?ag) 1)
                 (assign (item-hits ?b) 0)
            )
    )

;(:action destroy-oak-sapling
;    :parameters (?ag - agent ?b - oak-sapling)
;    :precondition (and (= (x ?b) (x ?ag))
;                        (= (y ?b) (+ (y ?ag) 1))
;                        (= (z ?b) (z ?ag))
;                        (present ?b)
;                        (= (item-hits ?b) 0))
;    :effect (and (not (present ?b))
;                 (increase (agent-num-oak-sapling ?ag) 1)
;                 (assign (item-hits ?b) 0)
;            )
;    )

;(:action destroy-birch-sapling
;    :parameters (?ag - agent ?b - birch-sapling)
;    :precondition (and (= (x ?b) (x ?ag))
;                        (= (y ?b) (+ (y ?ag) 1))
;                        (= (z ?b) (z ?ag))
;                        (present ?b)
;                        (= (item-hits ?b) 0))
;    :effect (and (not (present ?b))
;                 (increase (agent-num-birch-sapling ?ag) 1)
;                 (assign (item-hits ?b) 0)
;            )
;    )

)