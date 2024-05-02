(define (problem MINECRAFTCONTRIVED-1)
(:domain minecraft-contrived)


(:objects
	steve - agent
	ap0 - apple
	bed0 bed1 bed2 bed3 bed4 bed5 bed6 bed7 bed8 bed9 bed10 bed11 bed12 bed13 bed14 bed15 bed16 bed17 bed18 bed19 bed20 bed21 bed22 bed23 bed24 bed25 bed26 bed27 bed28 bed29 bed30 bed31 bed32 bed33 bed34 bed35 bed36 bed37 bed38 bed39 bed40 bed41 bed42 bed43 bed44 bed45 bed46 bed47 bed48 bed49 bed50 bed51 bed52 bed53 bed54 bed55 bed56 bed57 bed58 bed59 bed60 bed61 bed62 bed63 bed64 bed65 bed66 bed67 bed68 bed69 bed70 bed71 bed72 bed73 bed74 bed75 bed76 bed77 bed78 bed79 bed80 bed81 bed82 bed83 bed84 bed85 bed86 bed87 bed88 bed89 bed90 bed91 bed92 bed93 bed94 bed95 bed96 bed97 bed98 bed99 bed100 bed101 bed102 bed103 bed104 bed105 bed106 bed107 bed108 bed109 bed110 bed111 bed112 bed113 bed114 bed115 bed116 bed117 bed118 bed119 bed120 bed121 bed122 bed123 bed124 bed125 bed126 bed127 bed128 bed129 bed130 bed131 bed132 bed133 bed134 bed135 bed136 bed137 bed138 bed139 bed140 bed141 bed142 bed143 bed144 bed145 bed146 bed147 bed148 bed149 - bedrock-block
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
	
	(= (bl-x bed0) -1)
	(= (bl-y bed0) -1)
	(= (bl-z bed0) 3)
	(block-present bed0)
	
	(= (bl-x bed1) -1)
	(= (bl-y bed1) -1)
	(= (bl-z bed1) -1)
	(block-present bed1)
	
	(= (bl-x bed2) -1)
	(= (bl-y bed2) 0)
	(= (bl-z bed2) 3)
	(block-present bed2)
	
	(= (bl-x bed3) -1)
	(= (bl-y bed3) 0)
	(= (bl-z bed3) -1)
	(block-present bed3)
	
	(= (bl-x bed4) -1)
	(= (bl-y bed4) 1)
	(= (bl-z bed4) 3)
	(block-present bed4)
	
	(= (bl-x bed5) -1)
	(= (bl-y bed5) 1)
	(= (bl-z bed5) -1)
	(block-present bed5)
	
	(= (bl-x bed6) -1)
	(= (bl-y bed6) 2)
	(= (bl-z bed6) 3)
	(block-present bed6)
	
	(= (bl-x bed7) -1)
	(= (bl-y bed7) 2)
	(= (bl-z bed7) -1)
	(block-present bed7)
	
	(= (bl-x bed8) -1)
	(= (bl-y bed8) 3)
	(= (bl-z bed8) 3)
	(block-present bed8)
	
	(= (bl-x bed9) -1)
	(= (bl-y bed9) 3)
	(= (bl-z bed9) -1)
	(block-present bed9)
	
	(= (bl-x bed10) 0)
	(= (bl-y bed10) -1)
	(= (bl-z bed10) 3)
	(block-present bed10)
	
	(= (bl-x bed11) 0)
	(= (bl-y bed11) -1)
	(= (bl-z bed11) -1)
	(block-present bed11)
	
	(= (bl-x bed12) 0)
	(= (bl-y bed12) 0)
	(= (bl-z bed12) 3)
	(block-present bed12)
	
	(= (bl-x bed13) 0)
	(= (bl-y bed13) 0)
	(= (bl-z bed13) -1)
	(block-present bed13)
	
	(= (bl-x bed14) 0)
	(= (bl-y bed14) 1)
	(= (bl-z bed14) 3)
	(block-present bed14)
	
	(= (bl-x bed15) 0)
	(= (bl-y bed15) 1)
	(= (bl-z bed15) -1)
	(block-present bed15)
	
	(= (bl-x bed16) 0)
	(= (bl-y bed16) 2)
	(= (bl-z bed16) 3)
	(block-present bed16)
	
	(= (bl-x bed17) 0)
	(= (bl-y bed17) 2)
	(= (bl-z bed17) -1)
	(block-present bed17)
	
	(= (bl-x bed18) 0)
	(= (bl-y bed18) 3)
	(= (bl-z bed18) 3)
	(block-present bed18)
	
	(= (bl-x bed19) 0)
	(= (bl-y bed19) 3)
	(= (bl-z bed19) -1)
	(block-present bed19)
	
	(= (bl-x bed20) 1)
	(= (bl-y bed20) -1)
	(= (bl-z bed20) 3)
	(block-present bed20)
	
	(= (bl-x bed21) 1)
	(= (bl-y bed21) -1)
	(= (bl-z bed21) -1)
	(block-present bed21)
	
	(= (bl-x bed22) 1)
	(= (bl-y bed22) 0)
	(= (bl-z bed22) 3)
	(block-present bed22)
	
	(= (bl-x bed23) 1)
	(= (bl-y bed23) 0)
	(= (bl-z bed23) -1)
	(block-present bed23)
	
	(= (bl-x bed24) 1)
	(= (bl-y bed24) 1)
	(= (bl-z bed24) 3)
	(block-present bed24)
	
	(= (bl-x bed25) 1)
	(= (bl-y bed25) 1)
	(= (bl-z bed25) -1)
	(block-present bed25)
	
	(= (bl-x bed26) 1)
	(= (bl-y bed26) 2)
	(= (bl-z bed26) 3)
	(block-present bed26)
	
	(= (bl-x bed27) 1)
	(= (bl-y bed27) 2)
	(= (bl-z bed27) -1)
	(block-present bed27)
	
	(= (bl-x bed28) 1)
	(= (bl-y bed28) 3)
	(= (bl-z bed28) 3)
	(block-present bed28)
	
	(= (bl-x bed29) 1)
	(= (bl-y bed29) 3)
	(= (bl-z bed29) -1)
	(block-present bed29)
	
	(= (bl-x bed30) 2)
	(= (bl-y bed30) -1)
	(= (bl-z bed30) 3)
	(block-present bed30)
	
	(= (bl-x bed31) 2)
	(= (bl-y bed31) -1)
	(= (bl-z bed31) -1)
	(block-present bed31)
	
	(= (bl-x bed32) 2)
	(= (bl-y bed32) 0)
	(= (bl-z bed32) 3)
	(block-present bed32)
	
	(= (bl-x bed33) 2)
	(= (bl-y bed33) 0)
	(= (bl-z bed33) -1)
	(block-present bed33)
	
	(= (bl-x bed34) 2)
	(= (bl-y bed34) 1)
	(= (bl-z bed34) 3)
	(block-present bed34)
	
	(= (bl-x bed35) 2)
	(= (bl-y bed35) 1)
	(= (bl-z bed35) -1)
	(block-present bed35)
	
	(= (bl-x bed36) 2)
	(= (bl-y bed36) 2)
	(= (bl-z bed36) 3)
	(block-present bed36)
	
	(= (bl-x bed37) 2)
	(= (bl-y bed37) 2)
	(= (bl-z bed37) -1)
	(block-present bed37)
	
	(= (bl-x bed38) 2)
	(= (bl-y bed38) 3)
	(= (bl-z bed38) 3)
	(block-present bed38)
	
	(= (bl-x bed39) 2)
	(= (bl-y bed39) 3)
	(= (bl-z bed39) -1)
	(block-present bed39)
	
	(= (bl-x bed40) 3)
	(= (bl-y bed40) -1)
	(= (bl-z bed40) 3)
	(block-present bed40)
	
	(= (bl-x bed41) 3)
	(= (bl-y bed41) -1)
	(= (bl-z bed41) -1)
	(block-present bed41)
	
	(= (bl-x bed42) 3)
	(= (bl-y bed42) 0)
	(= (bl-z bed42) 3)
	(block-present bed42)
	
	(= (bl-x bed43) 3)
	(= (bl-y bed43) 0)
	(= (bl-z bed43) -1)
	(block-present bed43)
	
	(= (bl-x bed44) 3)
	(= (bl-y bed44) 1)
	(= (bl-z bed44) 3)
	(block-present bed44)
	
	(= (bl-x bed45) 3)
	(= (bl-y bed45) 1)
	(= (bl-z bed45) -1)
	(block-present bed45)
	
	(= (bl-x bed46) 3)
	(= (bl-y bed46) 2)
	(= (bl-z bed46) 3)
	(block-present bed46)
	
	(= (bl-x bed47) 3)
	(= (bl-y bed47) 2)
	(= (bl-z bed47) -1)
	(block-present bed47)
	
	(= (bl-x bed48) 3)
	(= (bl-y bed48) 3)
	(= (bl-z bed48) 3)
	(block-present bed48)
	
	(= (bl-x bed49) 3)
	(= (bl-y bed49) 3)
	(= (bl-z bed49) -1)
	(block-present bed49)
	
	(= (bl-x bed50) -1)
	(= (bl-y bed50) -1)
	(= (bl-z bed50) -1)
	(block-present bed50)
	
	(= (bl-x bed51) -1)
	(= (bl-y bed51) 3)
	(= (bl-z bed51) -1)
	(block-present bed51)
	
	(= (bl-x bed52) -1)
	(= (bl-y bed52) -1)
	(= (bl-z bed52) 0)
	(block-present bed52)
	
	(= (bl-x bed53) -1)
	(= (bl-y bed53) 3)
	(= (bl-z bed53) 0)
	(block-present bed53)
	
	(= (bl-x bed54) -1)
	(= (bl-y bed54) -1)
	(= (bl-z bed54) 1)
	(block-present bed54)
	
	(= (bl-x bed55) -1)
	(= (bl-y bed55) 3)
	(= (bl-z bed55) 1)
	(block-present bed55)
	
	(= (bl-x bed56) -1)
	(= (bl-y bed56) -1)
	(= (bl-z bed56) 2)
	(block-present bed56)
	
	(= (bl-x bed57) -1)
	(= (bl-y bed57) 3)
	(= (bl-z bed57) 2)
	(block-present bed57)
	
	(= (bl-x bed58) -1)
	(= (bl-y bed58) -1)
	(= (bl-z bed58) 3)
	(block-present bed58)
	
	(= (bl-x bed59) -1)
	(= (bl-y bed59) 3)
	(= (bl-z bed59) 3)
	(block-present bed59)
	
	(= (bl-x bed60) 0)
	(= (bl-y bed60) -1)
	(= (bl-z bed60) -1)
	(block-present bed60)
	
	(= (bl-x bed61) 0)
	(= (bl-y bed61) 3)
	(= (bl-z bed61) -1)
	(block-present bed61)
	
	(= (bl-x bed62) 0)
	(= (bl-y bed62) -1)
	(= (bl-z bed62) 0)
	(block-present bed62)
	
	(= (bl-x bed63) 0)
	(= (bl-y bed63) 3)
	(= (bl-z bed63) 0)
	(block-present bed63)
	
	(= (bl-x bed64) 0)
	(= (bl-y bed64) -1)
	(= (bl-z bed64) 1)
	(block-present bed64)
	
	(= (bl-x bed65) 0)
	(= (bl-y bed65) 3)
	(= (bl-z bed65) 1)
	(block-present bed65)
	
	(= (bl-x bed66) 0)
	(= (bl-y bed66) -1)
	(= (bl-z bed66) 2)
	(block-present bed66)
	
	(= (bl-x bed67) 0)
	(= (bl-y bed67) 3)
	(= (bl-z bed67) 2)
	(block-present bed67)
	
	(= (bl-x bed68) 0)
	(= (bl-y bed68) -1)
	(= (bl-z bed68) 3)
	(block-present bed68)
	
	(= (bl-x bed69) 0)
	(= (bl-y bed69) 3)
	(= (bl-z bed69) 3)
	(block-present bed69)
	
	(= (bl-x bed70) 1)
	(= (bl-y bed70) -1)
	(= (bl-z bed70) -1)
	(block-present bed70)
	
	(= (bl-x bed71) 1)
	(= (bl-y bed71) 3)
	(= (bl-z bed71) -1)
	(block-present bed71)
	
	(= (bl-x bed72) 1)
	(= (bl-y bed72) -1)
	(= (bl-z bed72) 0)
	(block-present bed72)
	
	(= (bl-x bed73) 1)
	(= (bl-y bed73) 3)
	(= (bl-z bed73) 0)
	(block-present bed73)
	
	(= (bl-x bed74) 1)
	(= (bl-y bed74) -1)
	(= (bl-z bed74) 1)
	(block-present bed74)
	
	(= (bl-x bed75) 1)
	(= (bl-y bed75) 3)
	(= (bl-z bed75) 1)
	(block-present bed75)
	
	(= (bl-x bed76) 1)
	(= (bl-y bed76) -1)
	(= (bl-z bed76) 2)
	(block-present bed76)
	
	(= (bl-x bed77) 1)
	(= (bl-y bed77) 3)
	(= (bl-z bed77) 2)
	(block-present bed77)
	
	(= (bl-x bed78) 1)
	(= (bl-y bed78) -1)
	(= (bl-z bed78) 3)
	(block-present bed78)
	
	(= (bl-x bed79) 1)
	(= (bl-y bed79) 3)
	(= (bl-z bed79) 3)
	(block-present bed79)
	
	(= (bl-x bed80) 2)
	(= (bl-y bed80) -1)
	(= (bl-z bed80) -1)
	(block-present bed80)
	
	(= (bl-x bed81) 2)
	(= (bl-y bed81) 3)
	(= (bl-z bed81) -1)
	(block-present bed81)
	
	(= (bl-x bed82) 2)
	(= (bl-y bed82) -1)
	(= (bl-z bed82) 0)
	(block-present bed82)
	
	(= (bl-x bed83) 2)
	(= (bl-y bed83) 3)
	(= (bl-z bed83) 0)
	(block-present bed83)
	
	(= (bl-x bed84) 2)
	(= (bl-y bed84) -1)
	(= (bl-z bed84) 1)
	(block-present bed84)
	
	(= (bl-x bed85) 2)
	(= (bl-y bed85) 3)
	(= (bl-z bed85) 1)
	(block-present bed85)
	
	(= (bl-x bed86) 2)
	(= (bl-y bed86) -1)
	(= (bl-z bed86) 2)
	(block-present bed86)
	
	(= (bl-x bed87) 2)
	(= (bl-y bed87) 3)
	(= (bl-z bed87) 2)
	(block-present bed87)
	
	(= (bl-x bed88) 2)
	(= (bl-y bed88) -1)
	(= (bl-z bed88) 3)
	(block-present bed88)
	
	(= (bl-x bed89) 2)
	(= (bl-y bed89) 3)
	(= (bl-z bed89) 3)
	(block-present bed89)
	
	(= (bl-x bed90) 3)
	(= (bl-y bed90) -1)
	(= (bl-z bed90) -1)
	(block-present bed90)
	
	(= (bl-x bed91) 3)
	(= (bl-y bed91) 3)
	(= (bl-z bed91) -1)
	(block-present bed91)
	
	(= (bl-x bed92) 3)
	(= (bl-y bed92) -1)
	(= (bl-z bed92) 0)
	(block-present bed92)
	
	(= (bl-x bed93) 3)
	(= (bl-y bed93) 3)
	(= (bl-z bed93) 0)
	(block-present bed93)
	
	(= (bl-x bed94) 3)
	(= (bl-y bed94) -1)
	(= (bl-z bed94) 1)
	(block-present bed94)
	
	(= (bl-x bed95) 3)
	(= (bl-y bed95) 3)
	(= (bl-z bed95) 1)
	(block-present bed95)
	
	(= (bl-x bed96) 3)
	(= (bl-y bed96) -1)
	(= (bl-z bed96) 2)
	(block-present bed96)
	
	(= (bl-x bed97) 3)
	(= (bl-y bed97) 3)
	(= (bl-z bed97) 2)
	(block-present bed97)
	
	(= (bl-x bed98) 3)
	(= (bl-y bed98) -1)
	(= (bl-z bed98) 3)
	(block-present bed98)
	
	(= (bl-x bed99) 3)
	(= (bl-y bed99) 3)
	(= (bl-z bed99) 3)
	(block-present bed99)
	
	(= (bl-x bed100) -1)
	(= (bl-y bed100) -1)
	(= (bl-z bed100) -1)
	(block-present bed100)
	
	(= (bl-x bed101) 3)
	(= (bl-y bed101) -1)
	(= (bl-z bed101) -1)
	(block-present bed101)
	
	(= (bl-x bed102) -1)
	(= (bl-y bed102) 0)
	(= (bl-z bed102) -1)
	(block-present bed102)
	
	(= (bl-x bed103) 3)
	(= (bl-y bed103) 0)
	(= (bl-z bed103) -1)
	(block-present bed103)
	
	(= (bl-x bed104) -1)
	(= (bl-y bed104) 1)
	(= (bl-z bed104) -1)
	(block-present bed104)
	
	(= (bl-x bed105) 3)
	(= (bl-y bed105) 1)
	(= (bl-z bed105) -1)
	(block-present bed105)
	
	(= (bl-x bed106) -1)
	(= (bl-y bed106) 2)
	(= (bl-z bed106) -1)
	(block-present bed106)
	
	(= (bl-x bed107) 3)
	(= (bl-y bed107) 2)
	(= (bl-z bed107) -1)
	(block-present bed107)
	
	(= (bl-x bed108) -1)
	(= (bl-y bed108) 3)
	(= (bl-z bed108) -1)
	(block-present bed108)
	
	(= (bl-x bed109) 3)
	(= (bl-y bed109) 3)
	(= (bl-z bed109) -1)
	(block-present bed109)
	
	(= (bl-x bed110) -1)
	(= (bl-y bed110) -1)
	(= (bl-z bed110) 0)
	(block-present bed110)
	
	(= (bl-x bed111) 3)
	(= (bl-y bed111) -1)
	(= (bl-z bed111) 0)
	(block-present bed111)
	
	(= (bl-x bed112) -1)
	(= (bl-y bed112) 0)
	(= (bl-z bed112) 0)
	(block-present bed112)
	
	(= (bl-x bed113) 3)
	(= (bl-y bed113) 0)
	(= (bl-z bed113) 0)
	(block-present bed113)
	
	(= (bl-x bed114) -1)
	(= (bl-y bed114) 1)
	(= (bl-z bed114) 0)
	(block-present bed114)
	
	(= (bl-x bed115) 3)
	(= (bl-y bed115) 1)
	(= (bl-z bed115) 0)
	(block-present bed115)
	
	(= (bl-x bed116) -1)
	(= (bl-y bed116) 2)
	(= (bl-z bed116) 0)
	(block-present bed116)
	
	(= (bl-x bed117) 3)
	(= (bl-y bed117) 2)
	(= (bl-z bed117) 0)
	(block-present bed117)
	
	(= (bl-x bed118) -1)
	(= (bl-y bed118) 3)
	(= (bl-z bed118) 0)
	(block-present bed118)
	
	(= (bl-x bed119) 3)
	(= (bl-y bed119) 3)
	(= (bl-z bed119) 0)
	(block-present bed119)
	
	(= (bl-x bed120) -1)
	(= (bl-y bed120) -1)
	(= (bl-z bed120) 1)
	(block-present bed120)
	
	(= (bl-x bed121) 3)
	(= (bl-y bed121) -1)
	(= (bl-z bed121) 1)
	(block-present bed121)
	
	(= (bl-x bed122) -1)
	(= (bl-y bed122) 0)
	(= (bl-z bed122) 1)
	(block-present bed122)
	
	(= (bl-x bed123) 3)
	(= (bl-y bed123) 0)
	(= (bl-z bed123) 1)
	(block-present bed123)
	
	(= (bl-x bed124) -1)
	(= (bl-y bed124) 1)
	(= (bl-z bed124) 1)
	(block-present bed124)
	
	(= (bl-x bed125) 3)
	(= (bl-y bed125) 1)
	(= (bl-z bed125) 1)
	(block-present bed125)
	
	(= (bl-x bed126) -1)
	(= (bl-y bed126) 2)
	(= (bl-z bed126) 1)
	(block-present bed126)
	
	(= (bl-x bed127) 3)
	(= (bl-y bed127) 2)
	(= (bl-z bed127) 1)
	(block-present bed127)
	
	(= (bl-x bed128) -1)
	(= (bl-y bed128) 3)
	(= (bl-z bed128) 1)
	(block-present bed128)
	
	(= (bl-x bed129) 3)
	(= (bl-y bed129) 3)
	(= (bl-z bed129) 1)
	(block-present bed129)
	
	(= (bl-x bed130) -1)
	(= (bl-y bed130) -1)
	(= (bl-z bed130) 2)
	(block-present bed130)
	
	(= (bl-x bed131) 3)
	(= (bl-y bed131) -1)
	(= (bl-z bed131) 2)
	(block-present bed131)
	
	(= (bl-x bed132) -1)
	(= (bl-y bed132) 0)
	(= (bl-z bed132) 2)
	(block-present bed132)
	
	(= (bl-x bed133) 3)
	(= (bl-y bed133) 0)
	(= (bl-z bed133) 2)
	(block-present bed133)
	
	(= (bl-x bed134) -1)
	(= (bl-y bed134) 1)
	(= (bl-z bed134) 2)
	(block-present bed134)
	
	(= (bl-x bed135) 3)
	(= (bl-y bed135) 1)
	(= (bl-z bed135) 2)
	(block-present bed135)
	
	(= (bl-x bed136) -1)
	(= (bl-y bed136) 2)
	(= (bl-z bed136) 2)
	(block-present bed136)
	
	(= (bl-x bed137) 3)
	(= (bl-y bed137) 2)
	(= (bl-z bed137) 2)
	(block-present bed137)
	
	(= (bl-x bed138) -1)
	(= (bl-y bed138) 3)
	(= (bl-z bed138) 2)
	(block-present bed138)
	
	(= (bl-x bed139) 3)
	(= (bl-y bed139) 3)
	(= (bl-z bed139) 2)
	(block-present bed139)
	
	(= (bl-x bed140) -1)
	(= (bl-y bed140) -1)
	(= (bl-z bed140) 3)
	(block-present bed140)
	
	(= (bl-x bed141) 3)
	(= (bl-y bed141) -1)
	(= (bl-z bed141) 3)
	(block-present bed141)
	
	(= (bl-x bed142) -1)
	(= (bl-y bed142) 0)
	(= (bl-z bed142) 3)
	(block-present bed142)
	
	(= (bl-x bed143) 3)
	(= (bl-y bed143) 0)
	(= (bl-z bed143) 3)
	(block-present bed143)
	
	(= (bl-x bed144) -1)
	(= (bl-y bed144) 1)
	(= (bl-z bed144) 3)
	(block-present bed144)
	
	(= (bl-x bed145) 3)
	(= (bl-y bed145) 1)
	(= (bl-z bed145) 3)
	(block-present bed145)
	
	(= (bl-x bed146) -1)
	(= (bl-y bed146) 2)
	(= (bl-z bed146) 3)
	(block-present bed146)
	
	(= (bl-x bed147) 3)
	(= (bl-y bed147) 2)
	(= (bl-z bed147) 3)
	(block-present bed147)
	
	(= (bl-x bed148) -1)
	(= (bl-y bed148) 3)
	(= (bl-z bed148) 3)
	(block-present bed148)
	
	(= (bl-x bed149) 3)
	(= (bl-y bed149) 3)
	(= (bl-z bed149) 3)
	(block-present bed149)
)


(:goal (and
        (> (agent-num-apples steve) 0)
    )
)




)