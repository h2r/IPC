# List of Domains we've looked into for Scoping with FD
## Failures
- Spider (conditional effects)
- Miconic (too simple - FD just scopes everything)
- Rovers (z3 can't simplify preconds down enough for substantial scoping, and FD scopes most of the stuff itself)
- Movie (too simple)

## Could work, but not immediately obvious how
- Depot (the transition dynamics allow the agent to drive from one place to another easily, so hard to exploit this to introduce conditional irrelevance)
- Gripper (due to the way the translator converts to SAS+, there is one pvar per gripper state [i.e, one pvar that says gripper1 could be holding any one of the 55 possible balls, for example]. Basically all actions affect this, and so this is an 'interconnected' domain...)
- Blocks (similar issue to gripper)

## Working
- Logistics (on prob15, Scoped FD currently outperforms FD even with the LM-Cut heuristic!)
- Satellite (on prob5, ScopedFD performs as well as FD with LM-Cut. You can increase problem difficulty arbitrarily by adding satellites with random instruments and then having their goal condition already fulfilled in the initial state)
- Driverlog (on prob01, without me doing ANYTHING, scoping just worked by default. It seems that adding packages that are already in the right position creates conditional irrelevance! We can blow this up arbitrarily to win even with LM-Cut heuristic!)