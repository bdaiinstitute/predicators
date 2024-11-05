(define (problem myproblem) (:domain mydomain)
  (:objects
    block0 - block
    block1 - block
    block2 - block
    block3 - block
    block4 - block
    robby - robot
  )
  (:init
    (Clear block4)
    (GripperOpen robby)
    (On block1 block0)
    (On block2 block1)
    (On block3 block2)
    (On block4 block3)
    (OnTable block0)
  )
  (:goal (or (On block1 block0)
    (On block2 block1)
    (On block4 block3)
    (OnTable block0)
    (OnTable block3)))
)