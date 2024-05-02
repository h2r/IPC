(define (problem MINECRAFTCONTRIVED-3)
    (:domain minecraft-bedmaking)


(:objects
	steve - agent
	old-pointy - diamond-axe
	of0 of1 of2 - orchid-flower
	wb0 wb1 - wooden-block
	woolb1 woolb2 woolb3 - wool-block
	bed1 - bed
)


(:init
	(agent-alive steve)
	(= (x wb0) 7)
	(= (y wb0) 12)
	(= (z wb0) 0)
	(block-present wb0)
	(= (x wb1) 7)
	(= (y wb1) 13)
	(= (z wb1) 0)
	(block-present wb1)
	(= (x steve) 3)
	(= (y steve) 0)
	(= (z steve) 0)
	( = ( agent-num-diamond steve ) 0 )
	( = ( agent-num-stick steve ) 0 )
	( = ( agent-num-diamond-axe steve ) 1 )
	( = ( agent-num-blue-dye steve ) 0 )
	( = ( agent-num-wool-block steve ) 3 )
	( = ( block-hits wb0 ) 0 )
	( = ( block-hits wb1 ) 0 )
	(= (agent-num-wooden-block steve) 0)
	(= (agent-num-wooden-planks steve) 0)
	( = ( block-hits woolb1 ) 0 )
	(not (block-present woolb1))
	( = ( wool-color woolb1 ) 0 )
	( = ( block-hits woolb2 ) 0 )
	(not (block-present woolb2))
	( = ( wool-color woolb2 ) 0 )
	( = ( block-hits woolb3 ) 0 )
	(not (block-present woolb3))
	( = ( wool-color woolb3 ) 0 )
	(= (x bed1) 0)
	(= (y bed1) 0)
	(= (z bed1) 0)
	( = ( block-hits bed1 ) 0 )
	( = ( bed-color bed1 ) 0 )
	(not (block-present bed1))
	(= (agent-num-bed steve) 0)
	( = ( item-hits of0 ) 0 )
	( = ( item-hits of1 ) 0 )
	( = ( item-hits of2 ) 0 )
	(= (agent-num-orchid-flower steve) 0)
	(= (x old-pointy) 0)
	(= (y old-pointy) 0)
	(= (z old-pointy) 0)
	( not ( present old-pointy ) )
	(= (x of0) 4)
	(= (y of0) 4)
	(= (z of0) 0)
	( present of0 )
	(= (x of1) 5)
	(= (y of1) 4)
	(= (z of1) 0)
	( present of1 )
	(= (x of2) 6)
	(= (y of2) 4)
	(= (z of2) 0)
	( present of2 )
)


(:goal (and 
                    (= (x bed1) 7)
                    (= (y bed1) 9)
                    (= (z bed1) 0)
                    (= (bed-color bed1) 1)
            
       (not (block-present wb0))
                (block-present wb1))
)


)