;   1 2 3 4 5
; 1 R .|T . G
; 2 . .|. . .
; 3 . . . . .
; 4 .|. .|. .
; 5 Y|. .|B .


(define (problem taxi_grid1)
  (:domain taxi_grid)
  (:objects
   taxi1 - taxi
   x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 - position
   joseph smoov curly george damian sabrina darryl furry bruce clark diana kara - person
   full threequarter half_full onequarter empty - fuel)

  (:init
   (inc x1 x2) (inc x2 x3) (inc x3 x4) (inc x4 x5)
   (inc y1 y2) (inc y2 y3) (inc y3 y4) (inc y4 y5)
   (eastwall x1 y4)
   (eastwall x1 y5)
   (eastwall x2 y1)
   (eastwall x2 y2)
   (eastwall x3 y4)
   (eastwall x3 y5)
   (usefuel full threequarter)
   (usefuel threequarter half_full)
   (usefuel half_full onequarter)
   (usefuel onequarter empty)
   (fillupfuel full)

   (tlocation taxi1 x3 y1)
   (tfuel taxi1 full)
   (plocation joseph x3 y5)
   (outsidetaxi joseph)
   (dest joseph x4 y5)
   (plocation smoov x3 y1)
   (insidetaxi smoov taxi1)
   (tfull taxi1)
   (dest smoov x1 y1)
   (plocation curly x1 y1)
   (outsidetaxi curly)
   (dest curly x2 y1)
   (plocation george x1 y2)
   (outsidetaxi george)
   (dest george x1 y1)
   (plocation damian x2 y2)
   (outsidetaxi damian)
   (dest damian x1 y1)
   (plocation sabrina x3 y3)
   (outsidetaxi sabrina)
   (dest sabrina x3 y4)
   (plocation darryl x5 y5)
   (outsidetaxi darryl)
   (dest darryl x3 y4)
   (plocation furry x1 y3)
   (outsidetaxi furry)
   (dest furry x5 y1)
   (plocation bruce x4 y3)
   (outsidetaxi bruce)
   (dest bruce x4 y3)
   (plocation clark x1 y2)
   (outsidetaxi clark)
   (dest clark x5 y5)
   (plocation diana x3 y2)
   (outsidetaxi diana)
   (dest diana x2 y3)
   (plocation kara x3 y5)
   (outsidetaxi kara)
   (dest kara x4 y5)
   )

  (:goal
   (and (plocation joseph x4 y5)
	(outsidetaxi joseph))
  )
)
