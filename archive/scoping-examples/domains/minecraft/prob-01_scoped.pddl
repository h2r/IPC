(define (problem MINECRAFTCONTRIVED-1)
(:domain minecraft-contrived)


(:objects
	steve - agent
	ap0 - apple
)


(:init
	(= (agent-x Steve) 0)
	(= (agent-y Steve) 0)
	(= (agent-z Steve) 0)
	( agent-has-pickaxe Steve )
	(= (agent-num-apples Steve) 0)
	(= (agent-num-potatoes Steve) 0)
	(= (agent-num-orchids Steve) 0)
	(= (agent-num-daisies Steve) 0)
	(= (agent-num-raw-rabbits Steve) 0)
	(= (agent-num-cooked-rabbits Steve) 0)
	
	
	(= (apple-x ap0) 0)
	(= (apple-y ap0) 1)
	(= (apple-z ap0) 0)
	( apple-present ap0 )
)


(:goal (and
        (> (agent-num-apples steve) 0)
    )
)




)