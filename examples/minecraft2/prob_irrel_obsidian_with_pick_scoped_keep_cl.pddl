(define (problem MINECRAFTCONTRIVED-1)
    (:domain minecraft-contrived)


(:objects
	obsidian0 obsidian1 - obsidian-block
	steve - agent
;	old-pointy - diamond-pickaxe
;	dmd0 dmd1 dmd2 - diamond
;	stick0 stick1 - stick
;	flint0 flint1 flint2 - flint
;	iron-ore0 - iron-ore
;	coal0 - coal
;	iron-ingot0 - iron-ingot
;	netherportal0 - netherportal
;	flint-and-steel0 - flint-and-steel
;	apple1 apple2 apple3 - apple
;	potato1 potato2 potato3 potato4 potato5 - potato
;	orchid-flower1 orchid-flower2 orchid-flower3 orchid-flower4 orchid-flower5 - orchid-flower
;	daisy-flower1 daisy-flower2 daisy-flower3 daisy-flower4 daisy-flower5 - daisy-flower
;	rabbit1 rabbit2 rabbit3 rabbit4 rabbit5 - rabbit
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
	(= (y obsidian0) 7)
	(= (z obsidian0) 1)
	(= (x obsidian1) 10)
	(= (y obsidian1) 7)
	(= (z obsidian1) 0)
	( = ( block-hits obsidian0 ) 0 )
	( = ( block-hits obsidian1 ) 0 )
	(= (agent-num-obsidian-block steve) 0)
;	(= (x old-pointy) 0)
;	(= (y old-pointy) 0)
;	(= (z old-pointy) 0)
;	( not ( present old-pointy ) )
;	(= (x stick0) 1)
;	(= (y stick0) 0)
;	(= (z stick0) 0)
;	( present stick0 )
;	(= (x stick1) 1)
;	(= (y stick1) 1)
;	(= (z stick1) 0)
;	( present stick1 )
;	(= (x flint0) 8)
;	(= (y flint0) 0)
;	(= (z flint0) 0)
;	( present flint0 )
;	(= (x flint1) 8)
;	(= (y flint1) 1)
;	(= (z flint1) 0)
;	( present flint1 )
;	(= (x flint2) 8)
;	(= (y flint2) 2)
;	(= (z flint2) 0)
;	( present flint2 )
;	(= (x iron-ore0) 10)
;	(= (y iron-ore0) 0)
;	(= (z iron-ore0) 0)
;	( present iron-ore0 )
;	(= (x coal0) 9)
;	(= (y coal0) 0)
;	(= (z coal0) 0)
;	( present coal0 )
;	(= (x dmd0) 2)
;	(= (y dmd0) 0)
;	(= (z dmd0) 0)
;	(present dmd0)
;	(= (x dmd1) 2)
;	(= (y dmd1) 1)
;	(= (z dmd1) 0)
;	(present dmd1)
;	(= (x dmd2) 2)
;	(= (y dmd2) 2)
;	(= (z dmd2) 0)
;	(present dmd2)
;	(= (x iron-ingot0) 0)
;	(= (y iron-ingot0) 0)
;	(= (z iron-ingot0) 0)
;	(not ( present iron-ingot0 ))
;	(= (x flint-and-steel0) 0)
;	(= (y flint-and-steel0) 0)
;	(= (z flint-and-steel0) 0)
;	(not ( present flint-and-steel0 ))
;	(= (x netherportal0) 0)
;	(= (y netherportal0) 0)
;	(= (z netherportal0) 0)
;	(not ( present netherportal0 ))
;	(= (x apple1) 3)
;	(= (y apple1) 0)
;	(= (z apple1) 0)
;	( present apple1 )
;	(= (x apple2) 3)
;	(= (y apple2) 1)
;	(= (z apple2) 0)
;	( present apple2 )
;	(= (x apple3) 3)
;	(= (y apple3) 2)
;	(= (z apple3) 0)
;	( present apple3 )
;	(= (x potato1) 4)
;	(= (y potato1) 0)
;	(= (z potato1) 0)
;	( present potato1 )
;	(= (x potato2) 4)
;	(= (y potato2) 1)
;	(= (z potato2) 0)
;	( present potato2 )
;	(= (x potato3) 4)
;	(= (y potato3) 2)
;	(= (z potato3) 0)
;	( present potato3 )
;	(= (x potato4) 4)
;	(= (y potato4) 3)
;	(= (z potato4) 0)
;	( present potato4 )
;	(= (x potato5) 4)
;	(= (y potato5) 4)
;	(= (z potato5) 0)
;	( present potato5 )
;	(= (x daisy-flower1) 5)
;	(= (y daisy-flower1) 0)
;	(= (z daisy-flower1) 0)
;	( present daisy-flower1 )
;	(= (x daisy-flower2) 5)
;	(= (y daisy-flower2) 1)
;	(= (z daisy-flower2) 0)
;	( present daisy-flower2 )
;	(= (x daisy-flower3) 5)
;	(= (y daisy-flower3) 2)
;	(= (z daisy-flower3) 0)
;	( present daisy-flower3 )
;	(= (x daisy-flower4) 5)
;	(= (y daisy-flower4) 3)
;	(= (z daisy-flower4) 0)
;	( present daisy-flower4 )
;	(= (x daisy-flower5) 5)
;	(= (y daisy-flower5) 4)
;	(= (z daisy-flower5) 0)
;	( present daisy-flower5 )
;	(= (x orchid-flower1) 6)
;	(= (y orchid-flower1) 0)
;	(= (z orchid-flower1) 0)
;	( present orchid-flower1 )
;	(= (x orchid-flower2) 6)
;	(= (y orchid-flower2) 1)
;	(= (z orchid-flower2) 0)
;	( present orchid-flower2 )
;	(= (x orchid-flower3) 6)
;	(= (y orchid-flower3) 2)
;	(= (z orchid-flower3) 0)
;	( present orchid-flower3 )
;	(= (x orchid-flower4) 6)
;	(= (y orchid-flower4) 3)
;	(= (z orchid-flower4) 0)
;	( present orchid-flower4 )
;	(= (x orchid-flower5) 6)
;	(= (y orchid-flower5) 4)
;	(= (z orchid-flower5) 0)
;	( present orchid-flower5 )
;	(= (x rabbit1) 7)
;	(= (y rabbit1) 0)
;	(= (z rabbit1) 0)
;	( present rabbit1 )
;	(= (x rabbit2) 7)
;	(= (y rabbit2) 1)
;	(= (z rabbit2) 0)
;	( present rabbit2 )
;	(= (x rabbit3) 7)
;	(= (y rabbit3) 2)
;	(= (z rabbit3) 0)
;	( present rabbit3 )
;	(= (x rabbit4) 7)
;	(= (y rabbit4) 3)
;	(= (z rabbit4) 0)
;	( present rabbit4 )
;	(= (x rabbit5) 7)
;	(= (y rabbit5) 4)
;	(= (z rabbit5) 0)
;	( present rabbit5 )
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
        


)