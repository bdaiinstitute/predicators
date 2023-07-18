"""Precompute a PRM-ish database for free space moving in the kitchen env."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import networkx as nx
from typing import Any

from numpy.typing import NDArray

from predicators.envs.kitchen import KitchenEnv
from predicators import utils
from predicators.settings import CFG

matplotlib.use("TkAgg")



@dataclass(frozen=True)
class Pose:
    xyz: NDArray[np.float32]
    joints: NDArray[np.float32]

    def distance(self, other: Pose) -> float:
        """Distance to another pose."""
        return np.linalg.norm(self.joints - other.joints)
    
    def __hash__(self) -> int:
        return hash(tuple(self.joints))
    
    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Pose)
        return np.allclose(self.joints, other.joints)


class OrnsteinUhlenbeckActionNoise:
    """Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py"""
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None, seed=0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.rng = np.random.default_rng(seed)
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * self.rng.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


def _reset_gym_env(gym_env):
    # Set up initial state with kettle out of the way.
    gym_env.reset()
    joint_state, _ = gym_env.get_env_state()
    joint_state[23:26] = -100  # kettle, way off screen
    gym_env.sim.set_state(joint_state)
    gym_env.sim.forward()


def _get_pose_from_env(gym_env):
    xyz = gym_env.get_ee_pose().copy()
    joints = gym_env.sim.data.qpos[:7].copy()
    return Pose(xyz, joints)


def _add_pose_to_graph(pose, graph, distance_thresh):
    graph.add_node(pose)
    for other_pose in graph.nodes:
        distance = pose.distance(other_pose)
        if distance < distance_thresh:
            graph.add_edge(pose, other_pose, weight=distance)
    

def _main() -> None:
    utils.reset_config({
        "env": "kitchen",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
    })
    env = KitchenEnv()
    gym_env = env._gym_env

    joint_lower_lim = gym_env.robot.robot_pos_bound[: 7, 0]
    joint_upper_lim = gym_env.robot.robot_pos_bound[: 7, 1]
    
    # Sample random trajectories in 7 DOF space.
    noise_scale = 0.1
    max_steps_per_traj = 100
    num_trajs = 10
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(7), sigma=noise_scale)

    distance_thresh = 0.1
    graph = nx.Graph()
    all_poses = []

    for trial in range(num_trajs):
        print(f"Starting trajectory {trial}")
        _reset_gym_env(gym_env)
        noise.reset()
        for _ in range(max_steps_per_traj):
            delta_act = noise()
            current_pos = gym_env.sim.data.qpos[:7]
            main_act = np.clip(current_pos + delta_act, joint_lower_lim, joint_upper_lim)
            act = np.concatenate([main_act, gym_env.sim.data.qpos[7:9]])
            gym_env.step(act)
            pose = _get_pose_from_env(gym_env)
            _add_pose_to_graph(pose, graph, distance_thresh)
            all_poses.append(pose)
            # TODO quit if something changes...

    _reset_gym_env(gym_env)
    init_pose = _get_pose_from_env(gym_env)
    _add_pose_to_graph(init_pose, graph, distance_thresh)
    target_xyz = all_poses[-1].xyz  # arbitary
    target = min(all_poses, key=lambda p: np.linalg.norm(p.xyz - target_xyz))
    print("init:", init_pose)
    print("target_xyz:", target_xyz)
    print("target:", target)

    path = nx.shortest_path(graph, init_pose, target, weight="weight")

    for pose in path:
        main_act = pose.joints
        act = np.concatenate([main_act, gym_env.sim.data.qpos[7:9]])
        gym_env.step(act)
        gym_env.render()
    
    final_pose = _get_pose_from_env(gym_env)
    print("Distance to target xyz:", np.linalg.norm(final_pose.xyz - target_xyz))

    # xs, ys, zs = np.transpose(visited_ee)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xs, ys, zs)
    # plt.show()



if __name__ == "__main__":
    _main()
