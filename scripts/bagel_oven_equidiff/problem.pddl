(define
    (problem BagelOven1)
    (:domain BagelOven)
    
    (:init
        (BagelOnTable)
        (TrayInsideOven)
        (OvenClosed)
        (NotHoldingBagel)
    )
    (:goal (and
            (TrayInsideOven)
            (OvenClosed)
            (BagelOnTray)
        )
    )
)
