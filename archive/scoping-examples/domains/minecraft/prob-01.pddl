(define (problem MINECRAFTCONTRIVED-1)
    (:domain minecraft-contrived)


obsidian0 obsidian1 - obsidian-block
steve - agent


(:init
	(= (x Steve) 0)
	(= (y Steve) 0)
	(= (z Steve) 0)
	(= (agent-num-apples Steve) 0)
	(= (agent-num-potatoes Steve) 0)
	(= (agent-num-orchids Steve) 0)
	(= (agent-num-daisies Steve) 0)
	(= (agent-num-raw-rabbits Steve) 0)
	(= (agent-num-cooked-rabbits Steve) 0)
	(= (x obsidian0) 0)
	(= (y obsidian0) 3)
	(= (z obsidian0) 1)
	(= (x obsidian1) 0)
	(= (y obsidian1) 3)
	(= (z obsidian1) 2)
	(block-present {s})
	(block-present {s})
)


(:goal (and
            (not (block-present {tgt_obsidian} ))
        )
    )

    


)