from env.models import TaskConfig


tasks = {

    "easy": TaskConfig(
        vehicles=2,
        signals=1,
        max_steps=40
    ),

    "medium": TaskConfig(
        vehicles=5,
        signals=2,
        max_steps=50
    ),

    "hard": TaskConfig(
        vehicles=8,
        signals=3,
        max_steps=60
    )

}