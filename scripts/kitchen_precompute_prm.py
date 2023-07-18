"""Precompute a PRM-ish database for free space moving in the kitchen env."""

import numpy as np

from predicators.envs.kitchen import KitchenEnv
from predicators import utils
from predicators.settings import CFG


class OrnsteinUhlenbeckActionNoise:
    """Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py"""
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    
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

    joint_lower_lim = gym_env.robot.robot_pos_bound[: 7, 0]
    joint_upper_lim = gym_env.robot.robot_pos_bound[: 7, 1]
    
    joint_state, _ = gym_env.get_env_state()
    joint_state[23:26] = -100  # kettle, way off screen
    gym_env.sim.set_state(joint_state)
    gym_env.sim.forward()

    # Sample random trajectories in 7 DOF space.
    rng = np.random.default_rng(CFG.seed)
    noise_scale = 0.1
    max_steps_per_traj = 1000
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(7), sigma=noise_scale * np.ones(7))
    noise.reset()
    for _ in range(max_steps_per_traj):
        delta_act = noise()
        current_pos = gym_env.sim.data.qpos[:7]
        main_act = np.clip(current_pos + delta_act, joint_lower_lim, joint_upper_lim)
        act = np.concatenate([main_act, gym_env.sim.data.qpos[7:9]])
        gym_env.step(act)
        gym_env.render()


if __name__ == "__main__":
    _main()
