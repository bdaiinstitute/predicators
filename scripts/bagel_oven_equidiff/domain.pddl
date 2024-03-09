(define (domain BagelOven)

(:predicates
    (NothingGrasped)
    
    (BagelGrasped)
    (OvenGrasped)
    (TrayGrasped)
    (TrayReadyToPush)
    
    (OvenClosed)
    (OvenOpen)
    
    (TrayInsideOven)
    (TrayPulledOut)
    
    (BagelOnTable)
    (BagelOnTray)
)

(:action GraspOven
    :parameters ()
    :precondition (NothingGrasped)
    :effect (and
        (OvenGrasped)
        (not (NothingGrasped))
    )
)

(:action OpenOven
    :parameters ()
    :precondition (and (OvenClosed) (OvenGrasped))
    :effect (and
        (OvenOpen)
        (not (OvenClosed))
        (NothingGrasped)
        (not (OvenGrasped))
    )
)

(:action CloseOven
    :parameters ()
    :precondition (and (OvenOpen) (TrayInsideOven) (OvenGrasped))
    :effect (and
        (OvenClosed)
        (not (OvenOpen))
        (NothingGrasped)
        (not (OvenGrasped))
    )
)

(:action GraspTray
    :parameters ()
    :precondition (and (NothingGrasped) (OvenOpen))
    :effect (and
        (TrayGrasped)
        (not (NothingGrasped))
    )
)

(:action PullOutTray
    :parameters ()
    :precondition (and (TrayInsideOven) (TrayGrasped))
    :effect (and
        (TrayPulledOut)
        (not (TrayInsideOven))
        (NothingGrasped)
        (not (TrayGrasped))
    )
)

(:action PrepareTrayPush
    :parameters ()
    :precondition (and (NothingGrasped) (TrayPulledOut))
    :effect (and
        (TrayReadyToPush)
        (not (NothingGrasped)) 
    )
)

(:action PushInTray
    :parameters ()
    :precondition (and (TrayPulledOut) (TrayReadyToPush))
    :effect (and
        (TrayInsideOven)
        (not (TrayPulledOut))
        (NothingGrasped)
        (not (TrayReadyToPush))
    )
)

(:action GraspBagel
    :parameters ()
    :precondition (and (BagelOnTable) (NothingGrasped))
    :effect (and
        (BagelGrasped)
        (not (NothingGrasped))
    )
)

(:action PlaceBagelOnTray
    :parameters ()
    :precondition (and (BagelGrasped) (TrayPulledOut))
    :effect (and
        (BagelOnTray)
        (NothingGrasped)
        (not (BagelGrasped))
    )
)

)
