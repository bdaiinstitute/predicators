(define (domain BagelOven)

(:predicates
    (NotHoldingBagel)
    (OvenClosed)
    (OvenOpen)
    (TrayInsideOven)
    (TrayPulledOut)
    (BagelGrasped)
    (BagelOnTable)
    (BagelOnTray)
)

(:action OpenOven
    :parameters ()
    :precondition (and (OvenClosed) (NotHoldingBagel))
    :effect (and
        (OvenOpen)
        (not (OvenClosed))
    )
)

(:action CloseOven
    :parameters ()
    :precondition (and (OvenOpen) (TrayInsideOven) (NotHoldingBagel))
    :effect (and
        (OvenClosed)
        (not (OvenOpen))
    )
)

(:action PullOutTray
    :parameters ()
    :precondition (and (TrayInsideOven) (OvenOpen) (NotHoldingBagel))
    :effect (and
        (TrayPulledOut)
        (not (TrayInsideOven))
    )
)

(:action PushInTray
    :parameters ()
    :precondition (and (TrayPulledOut) (NotHoldingBagel))
    :effect (and
        (TrayInsideOven)
        (not (TrayPulledOut))
    )
)

(:action PickBagelFromTable
    :parameters ()
    :precondition (and (BagelOnTable) (NotHoldingBagel))
    :effect (and
        (BagelGrasped)
        (not (BagelOnTable))
        (not (NotHoldingBagel))
    )
)

(:action PlaceBagelOnTray
    :parameters ()
    :precondition (and (BagelGrasped) (TrayPulledOut))
    :effect (and
        (BagelOnTray)
        (NotHoldingBagel)
        (not (BagelGrasped))
    )
)

)
