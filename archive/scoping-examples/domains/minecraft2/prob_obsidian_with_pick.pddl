(define (problem MINECRAFTCONTRIVED-1)
    (:domain minecraft-contrived)


(:objects
	obsidian0 obsidian1 - obsidian-block
	steve - agent
	old-pointy - diamond-pickaxe
	dmd0 dmd1 dmd2 - diamond
	stick0 stick1 - stick
	flint0 flint1 flint2 - flint
	iron-ore0 - iron-ore
	coal0 - coal
	iron-ingot0 - iron-ingot
	netherportal0 - netherportal
	flint-and-steel0 - flint-and-steel
)


(:init
	(agent-alive steve)
	(= (x steve) 0)
	(= (y steve) 0)
	(= (z steve) 0)
	( = ( agent-num-wool steve ) 0 )
	( = ( agent-num-diamond steve ) 0 )
	( = ( agent-num-stick steve ) 0 )
	( = ( agent-num-diamond-pickaxe steve ) 1 )
	( = ( agent-num-apple steve ) 0 )
	( = ( agent-num-potato steve ) 0 )
	( = ( agent-num-rabbit steve ) 0 )
	( = ( agent-num-orchid-flower steve ) 0 )
	( = ( agent-num-daisy-flower steve ) 0 )
	( = ( agent-num-flint steve ) 0 )
	( = ( agent-num-coal steve ) 0 )
	( = ( agent-num-iron-ore steve ) 0 )
	( = ( agent-num-iron-ingot steve ) 0 )
	( = ( agent-num-flint-and-steel steve ) 0 )
	(= (x obsidian0) 11)
	(= (y obsidian0) 8)
	(= (z obsidian0) 1)
	(= (x obsidian1) 10)
	(= (y obsidian1) 8)
	(= (z obsidian1) 0)
	( = ( block-hits obsidian0 ) 0 )
	( = ( block-hits obsidian1 ) 0 )
	(= (agent-num-obsidian-block steve) 0)
	(= (x old-pointy) 0)
	(= (y old-pointy) 0)
	(= (z old-pointy) 0)
	( not ( present old-pointy ) )
	(= (x stick0) 1)
	(= (y stick0) 0)
	(= (z stick0) 0)
	( present stick0 )
	(= (x stick1) 1)
	(= (y stick1) 1)
	(= (z stick1) 0)
	( present stick1 )
	(= (x flint0) 8)
	(= (y flint0) 0)
	(= (z flint0) 0)
	( present flint0 )
	(= (x flint1) 8)
	(= (y flint1) 1)
	(= (z flint1) 0)
	( present flint1 )
	(= (x flint2) 8)
	(= (y flint2) 2)
	(= (z flint2) 0)
	( present flint2 )
	(= (x iron-ore0) 10)
	(= (y iron-ore0) 0)
	(= (z iron-ore0) 0)
	( present iron-ore0 )
	(= (x coal0) 9)
	(= (y coal0) 0)
	(= (z coal0) 0)
	( present coal0 )
	(= (x dmd0) 2)
	(= (y dmd0) 0)
	(= (z dmd0) 0)
	(present dmd0)
	(= (x dmd1) 2)
	(= (y dmd1) 1)
	(= (z dmd1) 0)
	(present dmd1)
	(= (x dmd2) 2)
	(= (y dmd2) 2)
	(= (z dmd2) 0)
	(present dmd2)
	(= (x iron-ingot0) 0)
	(= (y iron-ingot0) 0)
	(= (z iron-ingot0) 0)
	(not ( present iron-ingot0 ))
	(= (x flint-and-steel0) 0)
	(= (y flint-and-steel0) 0)
	(= (z flint-and-steel0) 0)
	(not ( present flint-and-steel0 ))
	(= (x netherportal0) 0)
	(= (y netherportal0) 0)
	(= (z netherportal0) 0)
	(not ( present netherportal0 ))
	(block-present obsidian0)
	(block-present obsidian1)
)


(:goal (and
                (not (block-present obsidian0 ))
                (= (x steve) 3)
                (= (y steve) 4)
                (= (z steve) 0)
            )
        )
        


; State space size if we allow any object to have any z > 764455497855367032


; Conventional state space size = 5.959296202492212e+43


)