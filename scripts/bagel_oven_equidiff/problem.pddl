(define
    (problem bagel-oven-problem1)
    (:domain bagel-oven)
    
    (:init
        (bagelontable)
        (trayinsideoven)
        (ovenclosed)
        (nothinggrasped)
    )
    (:goal (and
            (trayinsideoven)
            (ovenclosed)
            (bagelontray)
            (nothinggrasped)
        )
    )
)
