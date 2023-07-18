"""Precompute a PRM-ish database for free space moving in the kitchen env."""

import numpy as np

from predicators.envs.kitchen import KitchenEnv
from predicators import utils


def _main() -> None:
    utils.reset_config({
        "env": "kitchen",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
    })
    env = KitchenEnv()
    # Set up initial state with kettle out of the way.
    env.reset("train", 0)
    gym_env = env._gym_env

    joint_state, _ = gym_env.get_env_state()
    joint_state[23:26] = -100  # kettle, way off screen
    gym_env.sim.set_state(joint_state)
    gym_env.sim.forward()

    gym_env.step(np.zeros(9))
    gym_env.render()
    import time; time.sleep(5)


if __name__ == "__main__":
    _main()
