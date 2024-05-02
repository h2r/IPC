(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0
	instrument0
	instrument1
	instrument2
	satellite1
	instrument3
	satellite2
	instrument4
	instrument5
	satellite3
	instrument6
	instrument7
	image2
	image1
	image0
	image3
	Star3
	GroundStation2
	Star1
	GroundStation4
	GroundStation0
	Phenomenon5
	Star6
	Star7
	Planet8
	Planet9
	Planet10
	Planet11
	; Irrelevant Items
	satellite4
	satellite5
	satellite6
	instrument8
	instrument9
	instrument10
	instrument11
	instrument12
	instrument13
	

)
(:init
	(satellite satellite0)
	(instrument instrument0)
	(supports instrument0 image1)
	(supports instrument0 image3)
	(calibration_target instrument0 Star1)
	(instrument instrument1)
	(supports instrument1 image3)
	(calibration_target instrument1 GroundStation0)
	(instrument instrument2)
	(supports instrument2 image0)
	(calibration_target instrument2 GroundStation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star6)
	(satellite satellite1)
	(instrument instrument3)
	(supports instrument3 image0)
	(supports instrument3 image2)
	(calibration_target instrument3 GroundStation4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation0)
	(satellite satellite2)
	(instrument instrument4)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 Star1)
	(instrument instrument5)
	(supports instrument5 image2)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 Star1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star6)
	(satellite satellite3)
	(instrument instrument6)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 GroundStation4)
	(instrument instrument7)
	(supports instrument7 image3)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(calibration_target instrument7 GroundStation0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 GroundStation2)
	(mode image2)
	(mode image1)
	(mode image0)
	(mode image3)
	(direction Star3)
	(direction GroundStation2)
	(direction Star1)
	(direction GroundStation4)
	(direction GroundStation0)
	(direction Phenomenon5)
	(direction Star6)
	(direction Star7)
	(direction Planet8)
	(direction Planet9)
	(direction Planet10)
	(direction Planet11)
	; Conditionally irrelevant propositions
	(instrument instrument8)
	(instrument instrument9)
	(instrument instrument10)
	(instrument instrument11)
	(instrument instrument12)
	(instrument instrument13)	
	(satellite satellite4)
	(power_avail satellite4)
	(pointing satellite4 Star3)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(satellite satellite5)
	(power_avail satellite5)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(pointing satellite5 GroundStation2)
	(satellite satellite6)
	(power_avail satellite6)
	(pointing satellite6 Star1)
	(on_board instrument12 satellite6)
	(on_board instrument13 satellite6)
)
(:goal (and
		(pointing satellite1 Star1)
		(pointing satellite2 Phenomenon5)
		(have_image Phenomenon5 image0)
		(have_image Star6 image1)
		(have_image Star7 image0)
		(have_image Planet8 image0)
		(have_image Planet9 image3)
		(have_image Planet10 image0)
		(have_image Planet11 image2)
		; Causally linked goal facts
		(pointing satellite4 Star3)
		(pointing satellite5 GroundStation2)
		(pointing satellite6 Star1)
	   )
)

)
