(define (domain mydomain)
  (:requirements :typing)
  (:types block robot)

  (:predicates
    (Clear ?x0 - block)
    (GripperOpen ?x0 - robot)
    (Holding ?x0 - block)
    (On ?x0 - block ?x1 - block)
    (OnTable ?x0 - block)
  )

  (:action PickFromTable
    :parameters (?block - block ?robot - robot)
    :precondition (and (Clear ?block)
        (GripperOpen ?robot)
        (OnTable ?block))
    :effect (and (Holding ?block)
        (not (Clear ?block))
        (not (GripperOpen ?robot))
        (not (OnTable ?block)))
  )

  (:action PutOnTable
    :parameters (?block - block ?robot - robot)
    :precondition (and (Holding ?block))
    :effect (and (Clear ?block)
        (GripperOpen ?robot)
        (OnTable ?block)
        (not (Holding ?block)))
  )

  (:action Stack
    :parameters (?block - block ?otherblock - block ?robot - robot)
    :precondition (and (Clear ?otherblock)
        (Holding ?block))
    :effect (and (Clear ?block)
        (GripperOpen ?robot)
        (On ?block ?otherblock)
        (not (Clear ?otherblock))
        (not (Holding ?block)))
  )

  (:action Unstack
    :parameters (?block - block ?otherblock - block ?robot - robot)
    :precondition (and (Clear ?block)
        (GripperOpen ?robot)
        (On ?block ?otherblock))
    :effect (and (Clear ?otherblock)
        (Holding ?block)
        (not (Clear ?block))
        (not (GripperOpen ?robot))
        (not (On ?block ?otherblock)))
  )
)