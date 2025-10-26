# Unifying Flight and Walking Imitation Pipelines

This note summarizes the practical differences between the existing
`flight_imitation` and `walk_imitation` tasks and documents the changes required
for sharing a single policy across both behaviours. The goal is to roll out
pretrained specialists in order to generate hybrid trajectories whose action and
observation spaces are mutually compatible.

## Baseline Differences

| Aspect | Flight (`FlightImitationWBPG`) | Walking (`WalkImitation`) |
| --- | --- | --- |
| Active limbs | Wings only (legs and adhesion removed) | Legs, adhesion, abdomen (wings retracted) |
| User action channels | 1 (wing-beat frequency scaling) | 0 |
| Control/physics timestep | 2e-4 / 5e-5 s | 2e-3 / 2e-4 s |
| Reference data | CoM pose/velocity only (`com_qpos`, `com_qvel`) | Full mocap snippet with joint & site tracks |
| Reward structure | CoM displacement, body orientation, leg retraction | DeepMimic joint/site tracking, wing retraction |
| Default observables | Hover-centric vestibular bundle | Locomotion-centric proprioception |

The action specification incompatibility stems from removing unused actuators
(including adhesion) whenever the corresponding limb is disabled. Likewise, the
reference buffers stored in `FruitFlyTask._ref_qpos/_ref_qvel` only contain the
signals needed by each specialist.

## Unifying the Action Space

Two small additions allow the original tasks to retain a shared action vector:

* `flight_imitation` exposes `unified_action_space`. When enabled the flight
  walker keeps its leg and adhesion actuators, but the task zeros those entries
  on every step so physics still behaves like a legless flyer.
* `walk_imitation` mirrors the behaviour for wings. The walker keeps the wing
  actuators active and the task clamps their controls to zero unless the caller
  overrides them.

Under the hood both tasks rely on the new `inactive_action_classes` argument,
which simply aggregates the relevant indices from
`FruitFly._action_indices` before applying the action. This keeps policy and
rollout code unchanged while guaranteeing a consistent action ordering. When the
flag is `False` the original behaviour is preserved.

## Mixed Locomotion Environment

The `MixedLocomotionImitation` task extends `FruitFlyTask` with a superset of
wing and leg control channels while reusing the reward logic of the two
specialists:

* Walking and flight reference segments are sampled independently from the HDF5
  loaders, upsampled to a common control timestep (2e-4 s) and concatenated in a
  single `_ref_qpos/_ref_qvel` buffer.
* A wing-beat pattern generator continues to provide high-frequency wing
  references during the flight phase.
* The task exposes one user action for WBPG frequency scaling and optionally a
  second “mode switch” control. When disabled, the environment follows a fixed
  walk→flight schedule and expects the policy to command both limbs
  simultaneously.
* Rewards switch seamlessly between the flight (CoM + orientation + leg
  retraction) and walking (DeepMimic + wing retraction) formulations depending on
  the active phase.

`fly_envs.mixed_locomotion_imitation` wires together the necessary trajectory
loaders, arena and generators so the environment can be instantiated alongside
existing helpers.

## XML Considerations

No MuJoCo XML edits are required for the unified setup. When `unified_action_space`
(or the mixed task) is active the walker still references the original
`fruitfly.xml`; actuators that would have been removed simply receive zero
commands. If tighter leg retraction or altered adhesive gains are desired during
flight segments, the usual `<default>` overrides can be applied in a custom XML
file and passed through the existing `walker_xml_path` hook in
`FruitFlyTask`.
