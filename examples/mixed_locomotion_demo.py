from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys

# Ensure the repository root is importable and headless rendering is disabled.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault('MUJOCO_GL', 'disable')

import matplotlib.pyplot as plt
import numpy as np

from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly
from flybody.tasks.mixed_locomotion import MixedLocomotionImitation
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory


@dataclass
class SyntheticWalkSnippet:
    qpos: np.ndarray
    qvel: np.ndarray
    root2site: np.ndarray
    joint_quat: np.ndarray


class SyntheticWalkingLoader:
    """Procedurally generates a short walking snippet for demonstrations."""

    def __init__(self, mocap_joint_names: list[str], mocap_site_names: list[str],
                 n_steps: int = 240, stride: float = 0.0015):
        self._mocap_joint_names = mocap_joint_names
        self._mocap_site_names = mocap_site_names
        self._n_steps = n_steps
        self._stride = stride
        self._snippet = self._build_snippet()

    def _build_snippet(self) -> SyntheticWalkSnippet:
        total_dofs = 7 + len(self._mocap_joint_names)
        vel_dofs = 6 + len(self._mocap_joint_names)

        walk_qpos = np.zeros((self._n_steps, total_dofs), dtype=np.float32)
        walk_qvel = np.zeros((self._n_steps, vel_dofs), dtype=np.float32)
        root2site = np.zeros((self._n_steps, len(self._mocap_site_names), 3),
                             dtype=np.float32)
        joint_quat = np.zeros((self._n_steps, len(self._mocap_joint_names), 4),
                              dtype=np.float32)
        joint_quat[..., 0] = 1.0  # Identity orientation for all joints.

        for step in range(self._n_steps):
            phase = 2 * math.pi * step / self._n_steps
            walk_qpos[step, 0] = self._stride * step
            walk_qpos[step, 2] = 0.1278 + 0.0015 * math.sin(phase)
            walk_qpos[step, 3] = 1.0  # Identity quaternion component w.
            walk_qvel[step, 0] = self._stride / max(self._n_steps * 2e-3, 1e-6)

            swing = 0.35 * math.sin(phase)
            anti_swing = 0.35 * math.sin(phase + math.pi)
            for i, name in enumerate(self._mocap_joint_names):
                offset = swing if 'left' in name else anti_swing
                walk_qpos[step, 7 + i] = offset
                walk_qvel[step, 6 + i] = 2 * math.pi * 0.35 * math.cos(phase)

        return SyntheticWalkSnippet(
            qpos=walk_qpos,
            qvel=walk_qvel,
            root2site=root2site,
            joint_quat=joint_quat,
        )

    def get_trajectory(self, traj_idx: int | None = None):  # pylint: disable=unused-argument
        return {
            'qpos': self._snippet.qpos,
            'qvel': self._snippet.qvel,
            'root2site': self._snippet.root2site,
            'joint_quat': self._snippet.joint_quat,
        }

    def get_joint_names(self):
        return self._mocap_joint_names

    def get_site_names(self):
        return self._mocap_site_names


class SyntheticFlightLoader:
    """Reuse the constant-speed generator to mimic a take-off segment."""

    def __init__(self, n_steps: int = 240, speed: float = 8.0,
                 height: float = 0.12):
        qpos, qvel = constant_speed_trajectory(
            n_steps=n_steps,
            speed=speed,
            init_pos=(0.0, 0.0, height),
            body_rot_angle_y=-47.5,
            control_timestep=2e-4,
        )
        self._com_qpos = qpos
        self._com_qvel = qvel

    def get_trajectory(self, traj_idx: int | None = None):  # pylint: disable=unused-argument
        return self._com_qpos, self._com_qvel


class DemoWingBeat(WingBeatPatternGenerator):
    """Wing beat generator with deterministic phase for plotting."""

    def reset(self, initial_phase: float = 0.0, return_qvel: bool = False):  # type: ignore[override]
        return super().reset(initial_phase=initial_phase, return_qvel=return_qvel)


def build_environment():
    walker = fruitfly.FruitFly
    arena = floors.Floor()

    mocap_joint_names = [
        'coxa_abduct_T1_left', 'coxa_twist_T1_left', 'coxa_T1_left',
        'femur_twist_T1_left', 'femur_T1_left', 'tibia_T1_left',
        'tarsus_T1_left', 'coxa_abduct_T1_right', 'coxa_twist_T1_right',
        'coxa_T1_right', 'femur_twist_T1_right', 'femur_T1_right',
        'tibia_T1_right', 'tarsus_T1_right'
    ]
    mocap_site_names = [
        'tarsus_T1_left', 'tarsus_T1_right',
        'tarsus_T2_left', 'tarsus_T2_right',
        'tarsus_T3_left', 'tarsus_T3_right',
    ]

    walk_loader = SyntheticWalkingLoader(mocap_joint_names, mocap_site_names)
    flight_loader = SyntheticFlightLoader()
    wbpg = DemoWingBeat()

    time_limit = 0.48 + 0.48
    task = MixedLocomotionImitation(
        walker=walker,
        arena=arena,
        walk_traj_generator=walk_loader,
        flight_traj_generator=flight_loader,
        wbpg=wbpg,
        mocap_joint_names=mocap_joint_names,
        mocap_site_names=mocap_site_names,
        walk_phase_duration=0.48,
        flight_phase_duration=0.48,
        terminal_com_dist=1.0,
        trajectory_sites=False,
        mode_control=True,
        joint_filter=0.01,
        force_actuators=False,
        future_steps=16,
    )

    env = composer.Environment(
        time_limit=time_limit,
        task=task,
        strip_singleton_obs_buffer_dim=True,
    )
    return env


def rollout(env, control_phase: np.ndarray | None = None):
    action_spec = env.action_spec()

    timestep = env.reset()
    physics = env.physics
    data = {
        'time': [physics.time()],
        'mode': [0.0],
        'com_ref_z': [env.task._ref_qpos[0, 2]],
        'com_model_z': [physics.named.data.subtree_com['walker/', 2]],
        'x_ref': [env.task._ref_qpos[0, 0]],
        'x_model': [physics.named.data.subtree_com['walker/', 0]],
    }

    step = 0
    while not timestep.last():
        action = np.zeros(action_spec.shape, dtype=np.float32)
        if env.task._mode_user_index is not None:
            if control_phase is not None:
                action[env.task._mode_user_index] = control_phase[step]
            else:
                action[env.task._mode_user_index] = -1.0 if step < env.task._walk_steps else 1.0

        timestep = env.step(action)
        step += 1

        data['time'].append(physics.time())
        data['mode'].append(env.task._current_phase.phase == 'flight')
        data['com_ref_z'].append(env.task._ref_qpos[min(step, env.task._combined_steps), 2])
        data['com_model_z'].append(
            physics.named.data.subtree_com['walker/', 2])
        data['x_ref'].append(env.task._ref_qpos[min(step, env.task._combined_steps), 0])
        data['x_model'].append(
            physics.named.data.subtree_com['walker/', 0])

    return data


def plot_results(data: dict[str, list[float]], output_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(data['time'], data['x_ref'], label='Reference COM x')
    axes[0].plot(data['time'], data['x_model'], label='Walker COM x')
    axes[0].set_ylabel('x position (m)')
    axes[0].legend(loc='upper left')

    axes[1].plot(data['time'], data['com_ref_z'], label='Reference COM z')
    axes[1].plot(data['time'], data['com_model_z'], label='Walker COM z')
    axes[1].fill_between(data['time'], 0, 1,
                         where=np.asarray(data['mode']) > 0.5,
                         alpha=0.2, transform=axes[1].get_xaxis_transform(),
                         color='tab:orange', label='Flight phase')
    axes[1].set_ylabel('z position (m)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    env = build_environment()
    data = rollout(env)
    output_path = os.path.join(REPO_ROOT, 'docs', 'images', 'mixed_demo.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_results(data, output_path)
    print(f"Saved demo plot to {os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
