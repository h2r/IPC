# List of Domains we've looked into for Scoping with FD
## Failures
- Spider (conditional effects)
- Miconic (too simple - FD just scopes everything)
- Rovers (z3 can't simplify preconds down enough for substantial scoping, and FD scopes most of the stuff itself)
- Movie (too simple)
- Woodworking (has metric)
- Barman (too complicated and interconnected)
- ChildSnack (too complicated and interconnected)
- Agricola (Action costs)
- DataNetwork (Functions)
- Freecell (too interconnected)
- Grid (too complicated and interconnected)
- Hiking (too interconnected...)
- Termes (too interconnected...)
- Storage (too simple)
- Snakes (too complex)
- Parking (too complex)
- Mystery (too strange)

## Could work, but not immediately obvious how
- Depot (the transition dynamics allow the agent to drive from one place to another easily, so hard to exploit this to introduce conditional irrelevance)
- Gripper (due to the way the translator converts to SAS+, there is one pvar per gripper state [i.e, one pvar that says gripper1 could be holding any one of the 55 possible balls, for example]. Basically all actions affect this, and so this is an 'interconnected' domain...)
- Blocks (similar issue to gripper)
- Tidybot (problem 1 breaks z3's recursion depth)
- TPP (looks too simple, didn't put too much effort into checking)

## Working
- Logistics
- Satellite
- Driverlog
- ZenoTravel
- NoMystery
