(define
    (problem BagelOven1)
    (:domain BagelOven)
    
    (:init
        (BagelOnTable)
        (TrayInsideOven)
        (OvenClosed)
        (NothingGrasped)
    )
    (:goal (and
            (TrayInsideOven)
            (OvenClosed)
            (BagelOnTray)
            (NothingGrasped)
        )
    )
)
