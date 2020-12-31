(define (problem MINECRAFTCONTRIVED-3)
    (:domain minecraft-bedmaking)


(:objects
	steve - agent
	old-pointy - diamond-axe
	of0 of1 of2 - orchid-flower
	wb0 wb1 - wooden-block
	woolb1 woolb2 woolb3 woolb4 - wool-block
	bed1 - bed
	dmd0 dmd1 dmd2 dmd3 dmd4 - diamond
	stick0 stick1 stick2 stick3 stick4 - stick
	bs0 bs1 bs2 bs3 bs4 bs5 bs6 bs7 bs8 bs9 bs10 bs11 bs12 bs13 bs14 bs15 bs16 bs17 bs18 bs19 - birch-sapling
	os0 os1 os2 os3 os4 os5 os6 os7 os8 os9 os10 os11 - oak-sapling
)


(:init
	(agent-alive steve)
	(= (x wb0) 7)
	(= (y wb0) 7)
	(= (z wb0) 0)
	(block-present wb0)
	(= (x wb1) 7)
	(= (y wb1) 8)
	(= (z wb1) 0)
	(block-present wb1)
	(= (x steve) 7)
	(= (y steve) 1)
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
	( = ( wool-color woolb2 ) 1 )
	( = ( block-hits woolb3 ) 0 )
	(not (block-present woolb3))
	( = ( wool-color woolb3 ) 1 )
	( = ( block-hits woolb4 ) 0 )
	(not (block-present woolb4))
	( = ( wool-color woolb4 ) 1 )
	(= (agent-num-wool-block steve) 3)
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
	( = ( item-hits bs0 ) 0 )
	( = ( item-hits bs1 ) 0 )
	( = ( item-hits bs2 ) 0 )
	( = ( item-hits bs3 ) 0 )
	( = ( item-hits bs4 ) 0 )
	( = ( item-hits bs5 ) 0 )
	( = ( item-hits bs6 ) 0 )
	( = ( item-hits bs7 ) 0 )
	( = ( item-hits bs8 ) 0 )
	( = ( item-hits bs9 ) 0 )
	( = ( item-hits bs10 ) 0 )
	( = ( item-hits bs11 ) 0 )
	( = ( item-hits bs12 ) 0 )
	( = ( item-hits bs13 ) 0 )
	( = ( item-hits bs14 ) 0 )
	( = ( item-hits bs15 ) 0 )
	( = ( item-hits bs16 ) 0 )
	( = ( item-hits bs17 ) 0 )
	( = ( item-hits bs18 ) 0 )
	( = ( item-hits bs19 ) 0 )
	(= (agent-num-birch-sapling steve) 0)
	( = ( item-hits os0 ) 0 )
	( = ( item-hits os1 ) 0 )
	( = ( item-hits os2 ) 0 )
	( = ( item-hits os3 ) 0 )
	( = ( item-hits os4 ) 0 )
	( = ( item-hits os5 ) 0 )
	( = ( item-hits os6 ) 0 )
	( = ( item-hits os7 ) 0 )
	( = ( item-hits os8 ) 0 )
	( = ( item-hits os9 ) 0 )
	( = ( item-hits os10 ) 0 )
	( = ( item-hits os11 ) 0 )
	(= (agent-num-oak-sapling steve) 0)
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
	(= (x stick0) 0)
	(= (y stick0) 2)
	(= (z stick0) 0)
	( present stick0 )
	(= (x stick1) 0)
	(= (y stick1) 3)
	(= (z stick1) 0)
	( present stick1 )
	(= (x stick2) 0)
	(= (y stick2) 4)
	(= (z stick2) 0)
	( present stick2 )
	(= (x stick3) 0)
	(= (y stick3) 5)
	(= (z stick3) 0)
	( present stick3 )
	(= (x stick4) 0)
	(= (y stick4) 6)
	(= (z stick4) 0)
	( present stick4 )
	(= (x dmd0) 1)
	(= (y dmd0) 2)
	(= (z dmd0) 0)
	(present dmd0)
	(= (x dmd1) 1)
	(= (y dmd1) 3)
	(= (z dmd1) 0)
	(present dmd1)
	(= (x dmd2) 1)
	(= (y dmd2) 4)
	(= (z dmd2) 0)
	(present dmd2)
	(= (x dmd3) 1)
	(= (y dmd3) 5)
	(= (z dmd3) 0)
	(present dmd3)
	(= (x dmd4) 1)
	(= (y dmd4) 6)
	(= (z dmd4) 0)
	(present dmd4)
	(= (x bs0) 2)
	(= (y bs0) 6)
	(= (z bs0) 0)
	(present bs0)
	(= (x bs1) 3)
	(= (y bs1) 6)
	(= (z bs1) 0)
	(present bs1)
	(= (x bs2) 4)
	(= (y bs2) 6)
	(= (z bs2) 0)
	(present bs2)
	(= (x bs3) 5)
	(= (y bs3) 6)
	(= (z bs3) 0)
	(present bs3)
	(= (x bs4) 6)
	(= (y bs4) 6)
	(= (z bs4) 0)
	(present bs4)
	(= (x bs5) 7)
	(= (y bs5) 6)
	(= (z bs5) 0)
	(present bs5)
	(= (x bs6) 8)
	(= (y bs6) 6)
	(= (z bs6) 0)
	(present bs6)
	(= (x bs7) 8)
	(= (y bs7) 5)
	(= (z bs7) 0)
	(present bs7)
	(= (x bs8) 8)
	(= (y bs8) 4)
	(= (z bs8) 0)
	(present bs8)
	(= (x bs9) 8)
	(= (y bs9) 3)
	(= (z bs9) 0)
	(present bs9)
	(= (x bs10) 8)
	(= (y bs10) 2)
	(= (z bs10) 0)
	(present bs10)
	(= (x bs11) 7)
	(= (y bs11) 2)
	(= (z bs11) 0)
	(present bs11)
	(= (x bs12) 6)
	(= (y bs12) 2)
	(= (z bs12) 0)
	(present bs12)
	(= (x bs13) 5)
	(= (y bs13) 2)
	(= (z bs13) 0)
	(present bs13)
	(= (x bs14) 4)
	(= (y bs14) 2)
	(= (z bs14) 0)
	(present bs14)
	(= (x bs15) 3)
	(= (y bs15) 2)
	(= (z bs15) 0)
	(present bs15)
	(= (x bs16) 2)
	(= (y bs16) 2)
	(= (z bs16) 0)
	(present bs16)
	(= (x bs17) 2)
	(= (y bs17) 3)
	(= (z bs17) 0)
	(present bs17)
	(= (x bs18) 2)
	(= (y bs18) 4)
	(= (z bs18) 0)
	(present bs18)
	(= (x bs19) 2)
	(= (y bs19) 5)
	(= (z bs19) 0)
	(present bs19)
	(= (x os0) 3)
	(= (y os0) 5)
	(= (z os0) 0)
	( present os0 )
	(= (x os1) 4)
	(= (y os1) 5)
	(= (z os1) 0)
	( present os1 )
	(= (x os2) 5)
	(= (y os2) 5)
	(= (z os2) 0)
	( present os2 )
	(= (x os3) 6)
	(= (y os3) 5)
	(= (z os3) 0)
	( present os3 )
	(= (x os4) 7)
	(= (y os4) 5)
	(= (z os4) 0)
	( present os4 )
	(= (x os5) 7)
	(= (y os5) 4)
	(= (z os5) 0)
	( present os5 )
	(= (x os6) 7)
	(= (y os6) 3)
	(= (z os6) 0)
	( present os6 )
	(= (x os7) 6)
	(= (y os7) 3)
	(= (z os7) 0)
	( present os7 )
	(= (x os8) 5)
	(= (y os8) 3)
	(= (z os8) 0)
	( present os8 )
	(= (x os9) 4)
	(= (y os9) 3)
	(= (z os9) 0)
	( present os9 )
	(= (x os10) 3)
	(= (y os10) 3)
	(= (z os10) 0)
	( present os10 )
	(= (x os11) 3)
	(= (y os11) 4)
	(= (z os11) 0)
	( present os11 )
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