"""Unified walking and flight imitation task for fruit fly locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from dm_control.utils import rewards

from flybody.quaternions import quat_dist_short_arc, rotate_vec_with_quat
from flybody.tasks.base import FruitFlyTask
from flybody.tasks.constants import (
    _FLY_CONTROL_TIMESTEP,
    _FLY_PHYSICS_TIMESTEP,
    _TERMINAL_ANGVEL,
    _TERMINAL_HEIGHT,
    _TERMINAL_LINVEL,
    _WALK_CONTROL_TIMESTEP,
)
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.rewards import (
    get_reference_features,
    get_walker_features,
    reward_factors_deep_mimic,
)
from flybody.tasks.task_utils import (
    add_trajectory_sites,
    com2root,
    root2com,
    update_trajectory_sites,
)
from flybody.tasks.trajectory_loaders import (
    HDF5FlightTrajectoryLoader,
    HDF5WalkingTrajectoryLoader,
)
from flybody.utils import any_substr_in_str


@dataclass
class PhaseSchedule:
    """Simple helper describing the current locomotion phase."""

    phase: Literal["walk", "flight"]
    step_in_phase: int


class MixedLocomotionImitation(FruitFlyTask):
    """Task that unifies walking and flight imitation in a single environment."""

    def __init__(
        self,
        walker,
        arena,
        walk_traj_generator: HDF5WalkingTrajectoryLoader,
        flight_traj_generator: HDF5FlightTrajectoryLoader,
        wbpg: WingBeatPatternGenerator,
        mocap_joint_names: Sequence[str],
        mocap_site_names: Sequence[str],
        walk_phase_duration: float,
        flight_phase_duration: float,
        terminal_com_dist: float = 0.4,
        trajectory_sites: bool = True,
        mode_control: bool = True,
        joint_filter: float = 0.01,
        force_actuators: bool = False,
        future_steps: int = 16,
        observables_options: dict | None = None,
    ):
        """Create a task in which a single policy alternates walking and flight.

        Args:
            walker: Walker constructor to be used.
            arena: Arena instance.
            walk_traj_generator: Loader for walking reference trajectories.
            flight_traj_generator: Loader for flight reference trajectories.
            wbpg: Wing beat pattern generator.
            mocap_joint_names: Names of joints recorded in walking mocap data.
            mocap_site_names: Names of mocap sites used for imitation reward.
            walk_phase_duration: Duration (seconds) of the walking segment.
            flight_phase_duration: Duration (seconds) of the flight segment.
            terminal_com_dist: Terminate when the CoM deviates more than this
                distance from the reference ghost.
            trajectory_sites: Whether to render reference trajectory sites.
            mode_control: If ``True``, adds an extra user action that allows the
                policy to switch between walk/flight modes on the fly. When
                ``False`` the environment follows a preset walk-then-flight
                schedule and expects simultaneous control of wings and legs.
            joint_filter: Time constant for joint actuators.
            force_actuators: Whether to use force or position actuators for the
                body and legs. Wings always use force actuators.
            future_steps: Number of future reference steps exposed as
                observables.
            observables_options: Optional overrides for observables.
        """

        self._walk_phase_duration = walk_phase_duration
        self._flight_phase_duration = flight_phase_duration
        self._walk_traj_generator = walk_traj_generator
        self._flight_traj_generator = flight_traj_generator
        self._wbpg = wbpg
        self._terminal_com_dist = terminal_com_dist
        self._trajectory_sites = trajectory_sites
        self._mode_control = mode_control
        self._future_steps = future_steps

        # One user action is reserved for wing-beat frequency control. An
        # optional second channel toggles the locomotion mode.
        num_user_actions = 1 + (1 if mode_control else 0)
        time_limit = walk_phase_duration + flight_phase_duration

        super().__init__(
            walker=walker,
            arena=arena,
            time_limit=time_limit,
            use_legs=True,
            use_wings=True,
            use_mouth=False,
            use_antennae=False,
            physics_timestep=_FLY_PHYSICS_TIMESTEP,
            control_timestep=_FLY_CONTROL_TIMESTEP,
            joint_filter=joint_filter,
            adhesion_filter=0.007,
            force_actuators=force_actuators,
            add_ghost=True,
            ghost_visible_legs=False,
            future_steps=future_steps,
            num_user_actions=num_user_actions,
            observables_options=observables_options,
        )

        # Pre-compute useful action indices.
        self._wing_action_inds = tuple(self._walker._action_indices['wings'])
        self._leg_action_inds = tuple(self._walker._action_indices['legs'])
        self._adhesion_action_inds = tuple(
            self._walker._action_indices.get('adhesion', ()))
        user_indices = self._walker._action_indices['user']
        self._freq_user_index = user_indices[0] if user_indices else None
        self._mode_user_index = (user_indices[1]
                                 if mode_control and len(user_indices) > 1
                                 else None)

        # Gather wing and leg joints for initialization and regularisation.
        self._wing_joints = []
        for side in ['left', 'right']:
            for axis in ['yaw', 'roll', 'pitch']:
                joint = f'wing_{axis}_{side}'
                self._wing_joints.append(
                    self._walker.mjcf_model.find('joint', joint))

        self._leg_joints = []
        self._leg_springrefs = []
        for joint in self._walker.mjcf_model.find_all('joint'):
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus'], joint.name):
                springref = joint.springref or joint.dclass.joint.springref or 0.
                self._leg_joints.append(joint)
                self._leg_springrefs.append(springref)
        self._leg_springrefs = np.asarray(self._leg_springrefs)

        # Prepare mocap features for the walking segment.
        self._mocap_joints = [self._root_joint]
        for joint_name in mocap_joint_names:
            self._mocap_joints.append(
                self._walker.mjcf_model.find('joint', joint_name))

        self._mocap_sites = []
        for site_name in mocap_site_names:
            self._mocap_sites.append(
                self._walker.mjcf_model.find('site', site_name))

        # Observables for reference tracking.
        self._walker.observables.add_observable('ref_displacement',
                                                self.ref_displacement)
        self._walker.observables.add_observable('ref_root_quat',
                                                self.ref_root_quat)

        # Trajectory visualization helpers.
        if self._trajectory_sites:
            n_traj_sites = (
                round(self._time_limit / self.control_timestep) + 1) // 10
            add_trajectory_sites(self.root_entity, n_traj_sites, group=1)
            self._n_traj_sites = n_traj_sites
        else:
            self._n_traj_sites = 0

        # Internal buffers initialised later.
        self._walk_stride = int(round(_WALK_CONTROL_TIMESTEP /
                                      self.control_timestep))
        if self._walk_stride < 1:
            raise ValueError('Control timestep must be <= walking timestep.')

        self._combined_steps = 0
        self._walk_ref_qpos = None
        self._walk_ref_qvel = None
        self._flight_ref_qpos = None
        self._flight_ref_qvel = None
        self._walk_snippet = None
        self._flight_steps = 0
        self._walk_steps = 0
        self._ghost_offset_with_quat = np.hstack((self._ghost_offset, 4 * [0]))

    # ------------------------------------------------------------------
    # Episode setup

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        super().initialize_episode_mjcf(random_state)

        # Walking snippet.
        self._walk_snippet = self._walk_traj_generator.get_trajectory()
        max_walk_steps = (self._walk_snippet['qpos'].shape[0]
                          - self._future_steps - 1)
        requested_walk_steps = int(round(
            self._walk_phase_duration / _WALK_CONTROL_TIMESTEP))
        walk_steps_dataset = max(1, min(max_walk_steps, requested_walk_steps))
        walk_stop = walk_steps_dataset + self._future_steps + 1
        self._walk_ref_qpos = np.repeat(
            self._walk_snippet['qpos'][:walk_stop, :7],
            self._walk_stride,
            axis=0,
        )
        self._walk_ref_qvel = np.repeat(
            self._walk_snippet['qvel'][:walk_stop, :6],
            self._walk_stride,
            axis=0,
        )
        self._walk_steps = walk_steps_dataset * self._walk_stride

        # Flight trajectory.
        flight_qpos, flight_qvel = self._flight_traj_generator.get_trajectory()
        ghost_root_pos = com2root(flight_qpos[:, :3], flight_qpos[:, 3:])
        flight_root_qpos = np.concatenate((ghost_root_pos,
                                           flight_qpos[:, 3:7]),
                                          axis=1)
        max_flight_steps = flight_root_qpos.shape[0] - self._future_steps - 1
        requested_flight_steps = int(round(
            self._flight_phase_duration / self.control_timestep))
        flight_steps_dataset = max(1, min(max_flight_steps,
                                          requested_flight_steps))
        flight_stop = flight_steps_dataset + self._future_steps + 1
        self._flight_ref_qpos = flight_root_qpos[:flight_stop]
        self._flight_ref_qvel = flight_qvel[:flight_stop]
        self._flight_steps = flight_steps_dataset

        # Align flight reference with the end of the walking segment.
        walk_end_pos = self._walk_ref_qpos[self._walk_steps, :3]
        flight_offset = walk_end_pos - self._flight_ref_qpos[0, :3]
        self._flight_ref_qpos[:, :3] += flight_offset

        # Build the global reference buffers for observables.
        self._combined_steps = self._walk_steps + self._flight_steps
        total = self._combined_steps + self._future_steps + 1
        self._ref_qpos = np.zeros((total, 7))
        self._ref_qvel = np.zeros((total, 6))
        walk_len = min(self._walk_ref_qpos.shape[0], total)
        self._ref_qpos[:walk_len] = self._walk_ref_qpos[:walk_len]
        self._ref_qvel[:walk_len] = self._walk_ref_qvel[:walk_len]
        flight_len = min(self._flight_ref_qpos.shape[0], total - self._walk_steps)
        self._ref_qpos[self._walk_steps:self._walk_steps + flight_len] = (
            self._flight_ref_qpos[:flight_len])
        self._ref_qvel[self._walk_steps:self._walk_steps + flight_len] = (
            self._flight_ref_qvel[:flight_len])

        if self._combined_steps > 0:
            self._ref_qpos[self._combined_steps:] = self._ref_qpos[
                self._combined_steps - 1]
            self._ref_qvel[self._combined_steps:] = self._ref_qvel[
                self._combined_steps - 1]

        # Update trajectory sites for visualization.
        if self._trajectory_sites and self._n_traj_sites:
            update_trajectory_sites(self.root_entity, self._ref_qpos,
                                    self._n_traj_sites, self._combined_steps)

        # Prepare ghost offset for walking orientation.
        rotated_offset = rotate_vec_with_quat(self._ghost_offset,
                                              self._walk_ref_qpos[0, 3:7])
        rotated_offset[2] = self._ghost_offset[2]
        self._ghost_offset_with_quat = np.hstack((rotated_offset, 4 * [0]))

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

        # Set initial joint configuration from walking snippet.
        physics.bind(self._mocap_joints).qpos = self._walk_snippet['qpos'][0, :]
        if self._initialize_qvel:
            physics.bind(self._mocap_joints).qvel = self._walk_snippet['qvel'][0, :]

        # Wings start from the WBPG neutral pose.
        init_wing_qpos, init_wing_qvel = self._wbpg.reset(
            initial_phase=random_state.uniform(), return_qvel=True)
        physics.bind(self._wing_joints).qpos = init_wing_qpos
        physics.bind(self._wing_joints).qvel = init_wing_qvel

        # Legs are placed at their spring reference (retracted).
        physics.bind(self._leg_joints).qpos = self._leg_springrefs

        # Set ghost pose to the initial walking reference.
        ghost_qpos = self._walk_ref_qpos[0] + self._ghost_offset_with_quat
        self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
        self._ghost.set_velocity(physics,
                                 self._walk_ref_qvel[0, :3],
                                 self._walk_ref_qvel[0, 3:])

        self._current_phase = PhaseSchedule('walk', 0)
        self._reached_traj_end = False

    # ------------------------------------------------------------------
    # Simulation loop helpers

    def _get_phase(self, step: int, action: np.ndarray) -> PhaseSchedule:
        if self._mode_control and self._mode_user_index is not None:
            mode_signal = action[self._mode_user_index]
            phase = 'flight' if mode_signal > 0 else 'walk'
            phase_step = (self._current_phase.step_in_phase + 1
                          if phase == self._current_phase.phase else 0)
        else:
            if step < self._walk_steps:
                phase = 'walk'
                phase_step = step
            else:
                phase = 'flight'
                phase_step = step - self._walk_steps
        return PhaseSchedule(phase, phase_step)

    def before_step(self, physics, action, random_state):
        # Protect from NaN actions.
        action[np.isnan(action)] = 0.0

        step = int(np.round(physics.data.time / self.control_timestep))
        self._current_phase = self._get_phase(step, action)

        # Remove user control from the action vector before passing to walker.
        freq_act = action[self._freq_user_index] if self._freq_user_index is not None else 0.0

        if self._mode_control and self._mode_user_index is not None:
            action[self._mode_user_index] = 0.0
        if self._freq_user_index is not None:
            action[self._freq_user_index] = 0.0

        if self._current_phase.phase == 'flight':
            # Apply WPG based wing control.
            base_freq = self._wbpg.base_beat_freq
            rel_range = self._wbpg.rel_freq_range
            ctrl = self._wbpg.step(
                ctrl_freq=base_freq * (1 + rel_range * freq_act))
            length = physics.bind(self._wing_joints).qpos
            action[self._wing_action_inds] += (ctrl - length)

            # Keep legs and adhesion idle while in flight.
            if self._leg_action_inds:
                action[list(self._leg_action_inds)] = 0.0
            if self._adhesion_action_inds:
                action[list(self._adhesion_action_inds)] = 0.0

            ref_index = min(self._current_phase.step_in_phase,
                            self._flight_steps - 1)
            ghost_qpos = self._flight_ref_qpos[ref_index]
            ghost_qpos = ghost_qpos + np.hstack((self._ghost_offset, 4 * [0]))
            ghost_qvel = self._flight_ref_qvel[ref_index]
            self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
            self._ghost.set_velocity(physics, ghost_qvel[:3], ghost_qvel[3:])
        else:
            # Walking segment: update ghost from mocap data.
            ref_index = min(self._current_phase.step_in_phase,
                            self._walk_steps - 1)
            ghost_qpos = (self._walk_ref_qpos[ref_index]
                          + self._ghost_offset_with_quat)
            ghost_qvel = self._walk_ref_qvel[ref_index]
            self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
            self._ghost.set_velocity(physics, ghost_qvel[:3], ghost_qvel[3:])

        super().before_step(physics, action, random_state)

    # ------------------------------------------------------------------
    # Reward and termination logic

    def get_reward_factors(self, physics):
        phase = self._current_phase.phase

        # CoM displacement reward shared across phases.
        ghost_xpos, ghost_quat = self._ghost.get_pose(physics)
        ghost_com = root2com(np.concatenate((ghost_xpos, ghost_quat)))
        model_com = physics.named.data.subtree_com['walker/']
        displacement = np.linalg.norm(ghost_com - model_com)
        displacement = rewards.tolerance(
            displacement,
            bounds=(0, 0),
            sigmoid='linear',
            margin=0.4,
            value_at_margin=0.0,
        )

        if phase == 'flight':
            quat = self.observables['walker/ref_root_quat'](physics)[0]
            quat_dist = quat_dist_short_arc(np.array([1., 0, 0, 0]), quat)
            quat_dist = rewards.tolerance(
                quat_dist,
                bounds=(0, 0),
                sigmoid='linear',
                margin=np.pi,
                value_at_margin=0.0,
            )

            qpos_diff = physics.bind(self._leg_joints).qpos - self._leg_springrefs
            retract_legs = rewards.tolerance(
                qpos_diff,
                bounds=(0, 0),
                sigmoid='linear',
                margin=4.,
                value_at_margin=0.0,
            )
            return (displacement, quat_dist, retract_legs)

        # Walking reward path.
        step = min(self._current_phase.step_in_phase,
                   self._walk_steps - 1)
        max_dataset_index = max(0, self._walk_snippet['qpos'].shape[0]
                                - self._future_steps - 2)
        dataset_step = min(
            step // max(self._walk_stride, 1),
            max_dataset_index,
        )
        walker_ft = get_walker_features(physics, self._mocap_joints,
                                        self._mocap_sites)
        reference_ft = get_reference_features(
            self._walk_snippet, dataset_step)
        reward_factors = reward_factors_deep_mimic(
            walker_features=walker_ft,
            reference_features=reference_ft,
            weights=(20, 1, 1, 1),
        )

        qpos_diff = physics.bind(self._wing_joints).qpos
        retract_wings = rewards.tolerance(
            qpos_diff,
            bounds=(0, 0),
            sigmoid='linear',
            margin=3.,
            value_at_margin=0.0,
        )
        return tuple(np.hstack((reward_factors, retract_wings)))

    def check_termination(self, physics):
        step = round(physics.time() / self.control_timestep)
        if step >= self._combined_steps:
            self._reached_traj_end = True
            return True

        com_dist = np.linalg.norm(
            self.observables['walker/ref_displacement'](physics)[0])
        if com_dist > self._terminal_com_dist:
            return True

        if self._current_phase.phase == 'flight':
            if (physics.named.data.xpos['walker/thorax', 2]
                    < _TERMINAL_HEIGHT):
                return True
        else:
            linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
            angvel = np.linalg.norm(self._walker.observables.gyro(physics))
            if linvel > _TERMINAL_LINVEL or angvel > _TERMINAL_ANGVEL:
                return True

        return super().check_termination(physics)

    def get_discount(self, physics):
        if self._should_terminate and not self._reached_traj_end:
            return 0.0
        return super().get_discount(physics)
