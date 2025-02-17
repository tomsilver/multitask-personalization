"""A pybullet based environment."""

from __future__ import annotations

import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import (
    Pose,
    get_half_extents_from_aabb,
    get_pose,
    multiply_poses,
    rotate_pose,
    set_pose,
)
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, iter_between_joint_positions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
from tomsutils.llm import LargeLanguageModel
from tomsutils.spaces import EnumSpace
from tomsutils.utils import (
    render_textbox_on_image,
    sample_seed_from_rng,
)

from multitask_personalization.envs.pybullet.pybullet_human import (
    create_human_from_spec,
)
from multitask_personalization.envs.pybullet.pybullet_missions import (
    CleanSurfacesMission,
    HandOverBookMission,
    StoreHumanHeldObjectMission,
    StoreRobotHeldObjectMission,
    WaitMission,
)
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    HiddenSceneSpec,
    PyBulletSceneSpec,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    GripperAction,
    PyBulletAction,
    PyBulletMission,
    PyBulletState,
)


class PyBulletEnv(gym.Env[PyBulletState, PyBulletAction]):
    """A pybullet based environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_spec: PyBulletSceneSpec,
        llm: LargeLanguageModel,
        hidden_spec: HiddenSceneSpec | None = None,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        # Keep a separate rng for LLM seed generation so that things don't change
        # if we make modifications to the rest of the environment that affect when
        # self._rng is used. Important because of LLM prompt caching.
        self._book_llm_rng = np.random.default_rng(seed)
        self._mission_rng = np.random.default_rng(seed)
        self._seed = seed
        self.scene_spec = scene_spec
        self._hidden_spec = hidden_spec
        self.render_mode = "rgb_array"
        self._llm = llm
        # Prevent accidentally admonishing the robot while it's trying to retract
        # after just cleaning a bad surface.
        self._steps_since_last_cleaning_admonishment: int | None = None

        # Create action space.
        self.action_space = gym.spaces.OneOf(
            (
                gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
                EnumSpace([GripperAction.OPEN, GripperAction.CLOSE]),
                gym.spaces.Text(100000),
                EnumSpace([None]),
            )
        )

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(
                camera_target=self.scene_spec.camera_target,
                camera_distance=self.scene_spec.camera_distance,
                camera_pitch=self.scene_spec.camera_pitch,
                camera_yaw=self.scene_spec.camera_yaw,
            )
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create floor.
        self.floor_id = p.loadURDF(
            str(self.scene_spec.floor_urdf),
            self.scene_spec.floor_position,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )

        # Create walls.
        self.wall_ids = [
            create_pybullet_block(
                (1.0, 1.0, 1.0, 1.0),
                self.scene_spec.wall_half_extents,
                self.physics_client_id,
            )
            for _ in self.scene_spec.wall_poses
        ]
        wall_texture_id = p.loadTexture(
            str(self.scene_spec.wall_texture), self.physics_client_id
        )
        for wall_id, pose in zip(
            self.wall_ids, self.scene_spec.wall_poses, strict=True
        ):
            p.changeVisualShape(
                wall_id,
                -1,
                textureUniqueId=wall_texture_id,
                physicsClientId=self.physics_client_id,
            )
            set_pose(wall_id, pose, self.physics_client_id)

        # Create robot.
        self.robot = self._create_robot(self.scene_spec, self.physics_client_id)

        # Create robot stand.
        self.robot_stand_id = create_pybullet_cylinder(
            self.scene_spec.robot_stand_rgba,
            radius=self.scene_spec.robot_stand_radius,
            length=self.scene_spec.robot_stand_length,
            physics_client_id=self.physics_client_id,
        )

        # Create bed.
        self.bed_id = p.loadURDF(
            str(self.scene_spec.bed_urdf),
            self.scene_spec.bed_pose.position,
            self.scene_spec.bed_pose.orientation,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )

        # Create human.
        self.human = create_human_from_spec(
            self.scene_spec.human_spec, self.physics_client_id
        )

        # Create a sim human on which we will do motion planning, IK, etc.
        self._sim_human_physics_client_id = p.connect(p.DIRECT)
        self._sim_human = create_human_from_spec(
            self.scene_spec.human_spec, self._sim_human_physics_client_id
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.scene_spec.table_rgba,
            half_extents=self.scene_spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        surface_texture_id = p.loadTexture(
            str(self.scene_spec.surface_texture), self.physics_client_id
        )
        p.changeVisualShape(
            self.table_id,
            -1,
            textureUniqueId=surface_texture_id,
            physicsClientId=self.physics_client_id,
        )

        # Create cup.
        self.cup_id = create_pybullet_cylinder(
            self.scene_spec.object_rgba,
            self.scene_spec.object_radius,
            self.scene_spec.object_length,
            physics_client_id=self.physics_client_id,
        )

        # Create duster.
        self.duster_id, self.duster_head_link_id, self.duster_pole_link_id = (
            _create_duster(
                self.scene_spec.duster_head_forward_length,
                self.scene_spec.duster_head_long_length,
                self.scene_spec.duster_head_up_down_length,
                self.scene_spec.duster_head_rgba,
                self.scene_spec.duster_pole_radius,
                self.scene_spec.duster_pole_height,
                self.scene_spec.duster_pole_rgba,
                self.scene_spec.duster_pole_offset,
                physics_client_id=self.physics_client_id,
            )
        )

        # Create shelf.
        self.shelf_id, self.shelf_link_ids = _create_shelf(
            self.scene_spec.shelf_rgba,
            shelf_width=self.scene_spec.shelf_width,
            shelf_depth=self.scene_spec.shelf_depth,
            shelf_height=self.scene_spec.shelf_height,
            spacing=self.scene_spec.shelf_spacing,
            support_width=self.scene_spec.shelf_support_width,
            num_layers=self.scene_spec.shelf_num_layers,
            shelf_texture_id=surface_texture_id,
            physics_client_id=self.physics_client_id,
        )

        # Create books.
        self.book_ids: list[int] = []
        self.book_descriptions: list[str] = []  # created in reset()
        for book_half_extents in self.scene_spec.book_half_extents:
            book_id = create_pybullet_block(
                (1.0, 1.0, 1.0, 1.0),
                half_extents=book_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self.book_ids.append(book_id)

        # Track whether the object is held, and if so, with what grasp.
        self.current_grasp_transform: Pose | None = None
        self.current_held_object_id: int | None = None

        # Track what the human is holding and whether a handover is happening.
        self.current_human_grasp_transform: Pose | None = None
        self.current_human_held_object_id: int | None = None
        self._human_action_queue: list[tuple[str, JointPositions]] = []

        # Create and track dust patches.
        self._dust_patches = {
            (surface, link_id): self._create_dust_patch_array(surface, link_id)
            for surface in self.get_surface_names()
            for link_id in self.get_surface_link_ids(
                self.get_object_id_from_name(surface)
            )
        }

        # For analysis purposes only. Should not be used by approaches.
        self._user_satisfaction = 0.0

        # Track the thing that the human is saying right now.
        self.current_human_text: str | None = None

        # Track the current mission for the robot.
        self._current_mission: PyBulletMission | None = None

        # Reset all states.
        self._reset_from_scene_spec()

        # Save the transform between the base and stand.
        world_to_base = self.robot.get_base_pose()
        world_to_stand = get_pose(self.robot_stand_id, self.physics_client_id)
        self.robot_base_to_stand = multiply_poses(
            world_to_base.invert(), world_to_stand
        )

        # Save the default half extents.
        obj_links_to_save = [
            (self.duster_id, self.duster_head_link_id),
        ] + [(book_id, -1) for book_id in self.book_ids]
        self._default_half_extents = {
            (o, l): get_half_extents_from_aabb(o, self.physics_client_id, link_id=l)
            for o, l in obj_links_to_save
        }

        # Create another robot for mission simulation.
        self._mission_sim_robot = self._create_robot(
            self.scene_spec, p.connect(p.DIRECT)
        )

        # Uncomment for debug / development.
        # if use_gui:
        #     while True:
        #         self._step_simulator((1, GripperAction.OPEN))
        #         p.stepSimulation(self.physics_client_id)

    def get_state(self) -> PyBulletState:
        """Get the underlying state from the simulator."""
        robot_base = self.robot.get_base_pose()
        robot_joints = self.robot.get_joint_positions()
        human_base = self.human.get_base_pose()
        human_joints = self.human.get_joint_positions()
        cup_pose = get_pose(self.cup_id, self.physics_client_id)
        duster_pose = get_pose(self.duster_id, self.physics_client_id)
        book_poses = [
            get_pose(book_id, self.physics_client_id) for book_id in self.book_ids
        ]
        held_object = (
            None
            if self.current_held_object_id is None
            else self.get_name_from_object_id(self.current_held_object_id)
        )
        human_held_object = (
            None
            if self.current_human_held_object_id is None
            else self.get_name_from_object_id(self.current_human_held_object_id)
        )
        # Convert PyBullet dust patch object colors into numpy array of levels.
        np_dust_patches = {
            k: np.empty(v.shape, dtype=np.float_) for k, v in self._dust_patches.items()
        }
        for surf, pybullet_id_arr in self._dust_patches.items():
            for r in range(pybullet_id_arr.shape[0]):
                for c in range(pybullet_id_arr.shape[1]):
                    value = self._get_dust_level(pybullet_id_arr[r, c])
                    np_dust_patches[surf][r, c] = value
        return PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            cup_pose,
            duster_pose,
            book_poses,
            self.book_descriptions,
            self.current_grasp_transform,
            np_dust_patches,
            held_object,
            self.current_human_text,
            human_held_object,
            self.current_human_grasp_transform,
        )

    def set_state(self, state: PyBulletState) -> None:
        """Sync the simulator with the given state."""
        self.set_robot_base(state.robot_base)
        self.robot.set_joints(state.robot_joints)
        assert self.human.get_base_pose().allclose(state.human_base)
        self.human.set_joints(state.human_joints)
        set_pose(self.cup_id, state.cup_pose, self.physics_client_id)
        set_pose(self.duster_id, state.duster_pose, self.physics_client_id)
        for book_id, book_pose in zip(self.book_ids, state.book_poses, strict=True):
            set_pose(book_id, book_pose, self.physics_client_id)
        self.book_descriptions = state.book_descriptions
        self.current_grasp_transform = state.grasp_transform
        self.current_human_text = state.human_text
        self.current_held_object_id = (
            None
            if state.held_object is None
            else self.get_object_id_from_name(state.held_object)
        )
        if self.current_held_object_id:
            self._close_robot_fingers()
        else:
            self._open_robot_fingers()
        self.current_human_held_object_id = (
            None
            if state.human_held_object is None
            else self.get_object_id_from_name(state.human_held_object)
        )
        self.current_human_grasp_transform = state.human_grasp_transform
        # Set PyBullet dust patch object colors from numpy array of levels.
        for surf, np_dust_patch_array in state.surface_dust_patches.items():
            for r in range(np_dust_patch_array.shape[0]):
                for c in range(np_dust_patch_array.shape[1]):
                    pybullet_id = self._dust_patches[surf][r, c]
                    value = np_dust_patch_array[r, c]
                    self._set_dust_level(pybullet_id, value)

    def _reset_from_scene_spec(self) -> None:

        # Reset robot.
        self.robot.set_base(self.scene_spec.robot_base_pose)
        self.robot.set_joints(self.scene_spec.initial_joints)
        self.robot.open_fingers()

        # Reset human.
        self.human.set_joints(self.scene_spec.human_spec.init_joints)

        # Reset robot stand.
        set_pose(
            self.robot_stand_id,
            self.scene_spec.robot_stand_pose,
            self.physics_client_id,
        )

        # Reset table.
        set_pose(self.table_id, self.scene_spec.table_pose, self.physics_client_id)

        # Reset cup.
        set_pose(self.cup_id, self.scene_spec.object_pose, self.physics_client_id)

        # Reset duster.
        set_pose(self.duster_id, self.scene_spec.duster_pose, self.physics_client_id)

        # Reset shelf.
        set_pose(self.shelf_id, self.scene_spec.shelf_pose, self.physics_client_id)

        # Reset books.
        for book_pose, book_id in zip(
            self.scene_spec.book_poses,
            self.book_ids,
            strict=True,
        ):
            set_pose(book_id, book_pose, self.physics_client_id)

        # Reset held object statuses.
        self.current_grasp_transform = None
        self.current_held_object_id = None
        self.current_human_grasp_transform = None
        self.current_human_held_object_id = None
        self._human_action_queue = []

        # Reset dust patches.
        for surface, link_id in self._dust_patches:
            self._reset_dust_patch_array(surface, link_id)

        # Reset human text.
        self.current_human_text = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[PyBulletState, dict[str, Any]]:
        # Add more randomization in future PR.
        super().reset(seed=seed, options=options)
        self._reset_from_scene_spec()

        # Reset user satisfaction.
        self._user_satisfaction = 0.0

        # Randomize book descriptions.
        self.book_descriptions = self._generate_book_descriptions(
            num_books=len(self.book_ids),
            seed=sample_seed_from_rng(self._book_llm_rng),
        )

        # Update the book covers.
        for book_description, book_id in zip(
            self.book_descriptions, self.book_ids, strict=True
        ):
            texture_id = self._get_texture_from_book_description(book_description)
            if texture_id is not None:
                p.changeVisualShape(
                    book_id,
                    -1,
                    textureUniqueId=texture_id,
                    physicsClientId=self.physics_client_id,
                )

        # Randomize robot mission.
        if options is not None and "initial_mission" in options:
            self._current_mission = options["initial_mission"]
            assert self._current_mission is not None
        else:
            self._current_mission = self._generate_mission()

        # Tell the robot its mission.
        self.current_human_text = self._current_mission.get_mission_command()

        if self.current_human_text:
            logging.info(f"Human says: {self.current_human_text}")

        return self.get_state(), self._get_info()

    def _step_simulator(self, action: PyBulletAction) -> None:
        # Reset current human text.
        self.current_human_text = None
        # Handle dust: increase for any dust not touched, zero out dust that is
        # touched by the duster.
        wiper_overlap_obj_ids = set()
        if self.current_held_object_id == self.duster_id:
            wiper_aabb_min, wiper_aabb_max = self._get_duster_surface_aabb()
            wiper_overlap_obj_links = p.getOverlappingObjects(
                wiper_aabb_min, wiper_aabb_max, physicsClientId=self.physics_client_id
            )
            wiper_overlap_obj_ids = {o for o, _ in wiper_overlap_obj_links}
        dust_delta = self.scene_spec.surface_dust_delta
        max_dust = self.scene_spec.surface_max_dust
        report_surface_should_not_be_cleaned = False
        for surf, pybullet_id_arr in self._dust_patches.items():
            for patch_id in pybullet_id_arr.flat:
                level = self._get_dust_level(patch_id)
                new_level = np.clip(level + dust_delta, 0, max_dust)
                # If the robot is holding the duster, check for contact between
                # the duster and any patches and remove dust accordingly.
                if patch_id in wiper_overlap_obj_ids:
                    new_level = 0.0
                    # Check if this surface should not be cleaned.
                    assert self._hidden_spec is not None
                    if (
                        not np.isclose(level, 0.0)
                        and surf not in self._hidden_spec.surfaces_robot_can_clean
                    ):
                        report_surface_should_not_be_cleaned = True
                self._set_dust_level(patch_id, new_level)
        if self._steps_since_last_cleaning_admonishment is not None:
            self._steps_since_last_cleaning_admonishment += 1
        if report_surface_should_not_be_cleaned:
            # Give the robot a chance to get away from the surface or else it
            # might be admonished again just for retracting.
            if (
                self._steps_since_last_cleaning_admonishment is None
                or self._steps_since_last_cleaning_admonishment
                > self.scene_spec.cleaning_admonishment_min_time_interval
            ):
                self.current_human_text = "Don't clean there -- I can do it myself."
                self._steps_since_last_cleaning_admonishment = 0
        # Continue moving the human if there is a plan to do so.
        if self._human_action_queue:
            human_action_name, human_joint_action = self._human_action_queue.pop(0)
            if human_action_name != "wait":
                self.human.set_joints(human_joint_action)
            if human_action_name == "handover":
                # If there is no plan left, execute the transfer.
                if not self._human_action_queue:
                    # Use a fixed transform here instead, to guarantee that
                    # reverse handover will work.
                    self.current_human_held_object_id = self.current_held_object_id
                    self.current_human_grasp_transform = (
                        self.scene_spec.human_spec.grasp_transform
                    )
                    self.current_held_object_id = None
                    self.current_grasp_transform = None
                    # Make a plan for the human back to resting position.
                    target_human_joints = self.scene_spec.human_spec.reading_joints
                    human_retract_plan = self._get_human_arm_plan(
                        target_human_joints, "reset"
                    )
                    # This is really hacky but we need the human to wait a bit
                    # before the robot has moved back.
                    current_human_joints = self.human.get_joint_positions()
                    human_wait_plan = [
                        ("wait", current_human_joints)
                    ] * self.scene_spec.handover_human_num_wait_steps
                    self._human_action_queue = human_wait_plan + human_retract_plan
        # Opening or closing the gripper.
        if np.isclose(action[0], 1):
            if action[1] == GripperAction.CLOSE:
                world_to_robot = self.robot.get_end_effector_pose()
                end_effector_position = world_to_robot.position
                # Check for objects near the end effector.
                grasp_threshold = 1e-2
                # Despite documentation, this actually returns a list of body
                # AND link ID tuples.
                grasped_object_link_ids = set(
                    p.getOverlappingObjects(
                        [
                            end_effector_position[0] - grasp_threshold,
                            end_effector_position[1] - grasp_threshold,
                            end_effector_position[2] - grasp_threshold,
                        ],
                        [
                            end_effector_position[0] + grasp_threshold,
                            end_effector_position[1] + grasp_threshold,
                            end_effector_position[2] + grasp_threshold,
                        ],
                        physicsClientId=self.physics_client_id,
                    )
                )
                graspable_object_ids = set(self.book_ids) | {
                    self.cup_id,
                    self.duster_id,
                }
                grasped_object_ids = {
                    o for o, _ in grasped_object_link_ids if o in graspable_object_ids
                }
                assert len(grasped_object_ids) <= 1
                if grasped_object_ids:
                    object_id = next(iter(grasped_object_ids))
                    world_to_object = get_pose(object_id, self.physics_client_id)
                    self.current_grasp_transform = multiply_poses(
                        world_to_robot.invert(), world_to_object
                    )
                    self.current_held_object_id = object_id
                    self._close_robot_fingers()
                    if object_id == self.current_human_held_object_id:
                        # Reverse hand over. Move human back to rest position
                        # after waiting a bit for robot to finish.
                        self.current_human_held_object_id = None
                        self.current_human_grasp_transform = None
                        human_retract_plan = self._get_human_arm_plan(
                            self.scene_spec.human_spec.reading_joints,
                            "reverse-handover",
                        )
                        current_human_joints = self.human.get_joint_positions()
                        human_wait_plan = [
                            ("wait", current_human_joints)
                        ] * self.scene_spec.handover_human_num_wait_steps
                        self._human_action_queue = human_wait_plan + human_retract_plan
            elif action[1] == GripperAction.OPEN:
                self.current_grasp_transform = None
                self.current_held_object_id = None
                self._open_robot_fingers()
            return
        # Robot indicating hand over.
        if np.isclose(action[0], 2) and action[1] == "Here you go!":
            # If the action is invalid, do nothing.
            if self.current_held_object_id is None:
                return
            # If the handover position is unreachable, do nothing.
            handover_pose = rotate_pose(self.robot.get_end_effector_pose(), roll=np.pi)

            assert self._hidden_spec is not None
            if not self._hidden_spec.rom_model.check_position_reachable(
                np.array(handover_pose.position)
            ):
                return
            # Otherwise, initiate the handover.
            self._sim_human.set_joints(self.human.get_joint_positions())
            target_human_joints = inverse_kinematics(
                self._sim_human, handover_pose, best_effort=True, validate=False
            )
            # Make a plan for the human to grab the object.
            self._human_action_queue = self._get_human_arm_plan(
                target_human_joints, "handover"
            )
            return
        # Robot indicated done.
        if np.isclose(action[0], 2) and action[1] == "Done":
            return
        # Not used.
        if np.isclose(action[0], 2):
            return
        # Robot is waiting.
        if np.isclose(action[0], 3):
            return
        # Moving the robot.
        assert np.isclose(action[0], 0)
        joint_action: JointPositions = list(action[1])  # type: ignore
        base_position_delta = joint_action[:3]
        joint_angle_delta = joint_action[3:]
        # Update the robot base.
        world_to_base = self.robot.get_base_pose()
        dx, dy, dyaw = base_position_delta
        x, y, z = world_to_base.position
        roll, pitch, yaw = world_to_base.rpy
        next_base = Pose.from_rpy((x + dx, y + dy, z), (roll, pitch, yaw + dyaw))
        self.set_robot_base(next_base)
        # Update the robot arm angles.
        current_joints = self.robot.get_joint_positions()
        # Only update the arm, assuming the first 7 entries are the arm.
        arm_joints = current_joints[:7]
        new_arm_joints = np.add(arm_joints, joint_angle_delta)  # type: ignore
        new_joints = list(current_joints)
        new_joints[:7] = new_arm_joints
        clipped_joints = np.clip(
            new_joints, self.robot.joint_lower_limits, self.robot.joint_upper_limits
        )
        self.robot.set_joints(clipped_joints.tolist())

        # Apply the grasp transform if it exists.
        if self.current_grasp_transform:
            world_to_robot = self.robot.get_end_effector_pose()
            world_to_object = multiply_poses(
                world_to_robot, self.current_grasp_transform
            )
            assert self.current_held_object_id is not None
            set_pose(
                self.current_held_object_id, world_to_object, self.physics_client_id
            )
        # Do the same for the human.
        if self.current_human_grasp_transform:
            world_to_human = self.human.get_end_effector_pose()
            world_to_object = multiply_poses(
                world_to_human, self.current_human_grasp_transform
            )
            assert self.current_human_held_object_id is not None
            set_pose(
                self.current_human_held_object_id,
                world_to_object,
                self.physics_client_id,
            )

        return

    def step(
        self, action: PyBulletAction
    ) -> tuple[PyBulletState, float, bool, bool, dict[str, Any]]:
        # Advance the simulator.
        state = self.get_state()
        self._step_simulator(action)

        # Advance the mission.
        assert self._current_mission is not None
        mission_text, mission_satisfaction = self._current_mission.step(state, action)
        if mission_text is not None:
            if self.current_human_text is None:
                self.current_human_text = mission_text
            else:
                self.current_human_text += "\n" + mission_text
        self._user_satisfaction = mission_satisfaction
        # NOTE: the done bit is only used during evaluation. Do not assume
        # that the environment will be reset after done=True.
        done = mission_satisfaction != 0

        # Start a new mission if the current one is complete.
        if self._current_mission.check_complete(state, action):
            next_mission = self._generate_mission()
            self._reset_mission(next_mission)

        if self.current_human_text:
            logging.info(f"Human says: {self.current_human_text}")

        if self._user_satisfaction != 0:
            logging.info(f"User satisfaction: {self._user_satisfaction}")

        # Return the next state and default gym API stuff.
        return self.get_state(), 0.0, done, False, self._get_info()

    def _get_info(self) -> dict[str, Any]:
        assert self._current_mission is not None
        env_video_should_pause = self.current_human_text is not None
        return {
            "mission": self._current_mission.get_id(),
            "scene_spec": self.scene_spec,
            "user_satisfaction": self._user_satisfaction,
            "env_video_should_pause": env_video_should_pause,
        }

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        img = capture_image(
            self.physics_client_id,
            camera_target=self.scene_spec.camera_target,
            camera_distance=self.scene_spec.camera_distance,
            camera_pitch=self.scene_spec.camera_pitch,
            camera_yaw=self.scene_spec.camera_yaw,
            image_width=self.scene_spec.image_width,
            image_height=self.scene_spec.image_height,
        )
        # In non-render mode, PyBullet does not render background correctly.
        # We want the background to be black instead of white. Here, make the
        # assumption that all perfectly white pixels belong to the background
        # and manually swap in black.
        background_mask = (img == [255, 255, 255]).all(axis=2)
        img[background_mask] = 0
        # If the human has just said something, render it in the image.
        if self.current_human_text is not None:
            img = render_textbox_on_image(
                img,
                self.current_human_text,
                textbox_color=(125, 0, 125, 125),
                max_chars_per_line=100,
            )
        return img  # type: ignore

    def save_state(self, filepath: Path) -> None:
        """Save the current state to disk."""
        admonish_steps = self._steps_since_last_cleaning_admonishment
        state_dict = {
            "state": self.get_state(),
            "mission": self._current_mission,
            "user_satisfaction": self._user_satisfaction,
            "rng": self._rng,
            "mission_rng": self._mission_rng,
            "book_llm_rng": self._book_llm_rng,
            "steps_since_last_cleaning_admonishment": admonish_steps,
            "human_handover_joint_queue": self._human_action_queue,
        }
        with open(filepath, "wb") as f:
            pkl.dump(state_dict, f)
        logging.info(f"Saved state to {filepath}")

    def load_state(self, filepath: Path) -> None:
        """Reset the current environment state from a saved state."""
        with open(filepath, "rb") as f:
            state_dict = pkl.load(f)
        self.set_state(state_dict["state"])
        self._current_mission = state_dict["mission"]
        self._user_satisfaction = state_dict["user_satisfaction"]
        self._rng = state_dict["rng"]
        self._mission_rng = state_dict["mission_rng"]
        self._book_llm_rng = state_dict["book_llm_rng"]
        self._steps_since_last_cleaning_admonishment = state_dict[
            "steps_since_last_cleaning_admonishment"
        ]
        self._human_action_queue = state_dict["human_handover_joint_queue"]
        logging.info(f"Loaded state from {filepath}")

    def _object_name_to_id(self) -> dict[str, int]:
        # If book descriptions have not been generated yet, use placeholder.
        if not self.book_descriptions:
            book_descriptions = ["NOT SET"] * len(self.book_ids)
        else:
            book_descriptions = self.book_descriptions
        book_name_to_id = dict(zip(book_descriptions, self.book_ids, strict=True))
        return {
            "cup": self.cup_id,
            "table": self.table_id,
            "shelf": self.shelf_id,
            "duster": self.duster_id,
            "bed": self.bed_id,
            **book_name_to_id,
        }

    def get_object_id_from_name(self, object_name: str) -> int:
        """Get the PyBullet object ID given a name."""
        return self._object_name_to_id()[object_name]

    def get_name_from_object_id(self, object_id: int) -> str:
        """Inverse of get_object_id_from_name()."""
        obj_name_to_id = self._object_name_to_id()
        obj_id_to_name = {v: k for k, v in obj_name_to_id.items()}
        return obj_id_to_name[object_id]

    def get_surface_names(self) -> set[str]:
        """Get all possible surfaces in the environment."""
        return {"table", "shelf"}

    def get_surface_ids(self) -> set[int]:
        """Get all possible surfaces in the environment."""
        surface_names = self.get_surface_names()
        return {self.get_object_id_from_name(n) for n in surface_names}

    def get_surface_that_object_is_on(
        self, object_id: int, distance_threshold: float = 1e-3
    ) -> int:
        """Get the PyBullet ID of the surface that the object is on."""
        surfaces = self.get_surface_ids()
        assert object_id not in surfaces
        object_pose = get_pose(object_id, self.physics_client_id)
        for surface_id in surfaces:
            surface_pose = get_pose(surface_id, self.physics_client_id)
            # Check if object pose is above surface pose.
            # NOTE: this assumes that the local frame of the objects are
            # roughly at the center.
            if object_pose.position[2] < surface_pose.position[2]:
                continue
            # Check for contact.
            if check_body_collisions(
                object_id,
                surface_id,
                self.physics_client_id,
                distance_threshold=distance_threshold,
            ):
                return surface_id
        raise ValueError(f"Object {object_id} not on any surface.")

    def get_collision_ids(self) -> set[int]:
        """Get all collision IDs for the environment."""
        return set(self.book_ids) | {
            self.table_id,
            self.human.robot_id,
            self.shelf_id,
            self.duster_id,
            self.cup_id,
        }

    def get_surface_link_ids(self, object_id: int) -> set[int]:
        """Get all link IDs for surfaces for a given object."""
        if object_id == self.shelf_id:
            return self.shelf_link_ids
        return {-1}

    def get_aabb_dimensions(self, object_id: int) -> tuple[float, float, float]:
        """Get the 3D bounding box dimensions of an object."""
        (min_x, min_y, min_z), (max_x, max_y, max_z) = p.getAABB(
            object_id, -1, self.physics_client_id
        )
        return (max_x - min_x, max_y - min_y, max_z - min_z)

    def set_robot_base(self, base_pose: Pose) -> None:
        """Update the robot and stand pose."""
        self.robot.set_base(base_pose)
        next_stand_pose = multiply_poses(base_pose, self.robot_base_to_stand)
        set_pose(self.robot_stand_id, next_stand_pose, self.physics_client_id)

    def _create_robot(
        self, scene_spec: PyBulletSceneSpec, physics_client_id: int
    ) -> FingeredSingleArmPyBulletRobot:
        robot = create_pybullet_robot(
            scene_spec.robot_name,
            physics_client_id,
            base_pose=scene_spec.robot_base_pose,
            control_mode="reset",
            home_joint_positions=scene_spec.initial_joints,
            fixed_base=False,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        return robot

    def _reset_mission(self, next_mission: PyBulletMission) -> None:
        self._current_mission = next_mission
        # Tell the robot its new mission.
        mission_description = self._current_mission.get_mission_command()
        if self.current_human_text is None:
            self.current_human_text = mission_description
        else:
            self.current_human_text += "\n" + mission_description
        # Move the human arm to prepare for reverse handover.
        if self._current_mission.get_id() == "store human held object":
            self._human_action_queue = self._get_human_arm_plan(
                self.scene_spec.human_spec.reverse_handover_joints,
                "prepare-reverse-handover",
            )

    def _generate_mission(self) -> PyBulletMission:
        state = self.get_state()
        possible_missions = self._create_possible_missions()
        self._rng.shuffle(possible_missions)  # type: ignore
        for mission in possible_missions:
            if mission.check_initiable(state):
                return mission
        raise NotImplementedError

    def _create_possible_missions(self) -> list[PyBulletMission]:
        seed = sample_seed_from_rng(self._mission_rng)
        assert self._hidden_spec is not None
        # NOTE: don't use the real robot / real environment inside the missions
        # in case they want to do things like use robot FK.
        possible_missions: list[PyBulletMission] = [
            StoreRobotHeldObjectMission(self._mission_sim_robot),
        ]

        assert self._hidden_spec.missions in ["all", "handover-only", "clean-only"]

        if self._hidden_spec.missions in ["all", "handover-only"]:
            handover_mission = HandOverBookMission(
                self.book_descriptions,
                self._mission_sim_robot,
                self._hidden_spec.rom_model,
                self._hidden_spec.book_preferences,
                self._llm,
                seed=seed,
            )
            reverse_handover_mission = StoreHumanHeldObjectMission(
                self._mission_sim_robot,
                self.scene_spec.human_spec.reverse_handover_joints,
            )
            possible_missions.extend([handover_mission, reverse_handover_mission])

        if self._hidden_spec.missions in ["all", "clean-only"]:
            clean_mission = CleanSurfacesMission()
            possible_missions.append(clean_mission)

        return possible_missions

    def _generate_book_descriptions(self, num_books: int, seed: int) -> list[str]:
        if self.scene_spec.use_standard_books:
            assert num_books == 3
            return [
                "Title: Moby Dick. Author: Herman Melville.",
                "Title: The Hitchhiker's Guide to the Galaxy. Author: Douglas Adams.",
                "Title: The Lord of the Rings. Author: J. R. R. Tolkien.",
            ]
        assert self._hidden_spec is not None
        user_preferences = self._hidden_spec.book_preferences
        # pylint: disable=line-too-long
        prompt = f"""Generate a list of {num_books} real English-language book titles and authors. Be creative.

Generate one book that the user would love and other books that the user would hate, based on the following user preferences: "{user_preferences}"
        
Return the list in the following format:

1. [The user would love] Title: <title>. Author: <author>.
2. [The user would hate] Title: <title>. Author: <author>.
3. [The user would hate] Title: <title>. Author: <author>.
etc.

Return that list and nothing else. Do not explain anything."""
        for _ in range(100):  # retry until parsing works
            response = self._llm.sample_completions(
                prompt,
                imgs=None,
                temperature=1.0,
                seed=seed,
            )[0]
            book_descriptions: list[str] = []
            for i, line in enumerate(response.split("\n")):
                prefixes = (
                    f"{i+1}. [The user would love] ",
                    f"{i+1}. [The user would hate] ",
                )
                for prefix in prefixes:
                    if line.startswith(prefix):
                        book_description = line[len(prefix) :]
                        book_descriptions.append(book_description)
                        break
            if len(book_descriptions) == num_books:  # success
                return book_descriptions
        raise RuntimeError("LLM book description generation failed")

    def _get_texture_from_book_description(self, book_description: str) -> int | None:
        book_dir = Path(__file__).parent / "assets" / "books"
        if book_description == "Title: Moby Dick. Author: Herman Melville.":
            filepath = book_dir / "moby_dick" / "combined.jpg"
        elif (
            book_description
            == "Title: The Hitchhiker's Guide to the Galaxy. Author: Douglas Adams."
        ):
            filepath = book_dir / "hitchhikers" / "combined.jpg"
        elif (
            book_description
            == "Title: The Lord of the Rings. Author: J. R. R. Tolkien."
        ):
            filepath = book_dir / "lor" / "combined.jpg"
        else:
            return None
        return p.loadTexture(str(filepath), self.physics_client_id)

    def _create_dust_patch_array(self, surface_name: str, link_id: int) -> NDArray:
        """Create an array of PyBullet IDs."""
        half_extents = self._get_dust_patch_dimensions(surface_name, link_id)
        color = self.scene_spec.dust_color + (0.0,)
        s = self.scene_spec.surface_dust_patch_size
        patch_arr = np.empty((s, s), dtype=int)
        for r, c, pose in self._get_dust_patch_poses(surface_name, link_id):
            # NOTE: if a collision shape is not created, the AABB of the patch
            # will be wrong, which messes up all forms of collision checking or
            # overlap checking that might be used to detect dust wiping. So we
            # create a collision shape but then disable collisions using groups.
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.physics_client_id,
            )
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=color,
                physicsClientId=self.physics_client_id,
            )
            patch_id = p.createMultiBody(
                baseMass=-1,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=pose.position,
                baseOrientation=pose.orientation,
                physicsClientId=self.physics_client_id,
            )
            p.setCollisionFilterGroupMask(
                bodyUniqueId=patch_id,
                linkIndexA=-1,
                collisionFilterGroup=0,
                collisionFilterMask=0,
                physicsClientId=self.physics_client_id,
            )
            patch_arr[r, c] = patch_id

        return patch_arr

    def _reset_dust_patch_array(self, surface_name: str, link_id: int) -> None:
        patch_arr = self._dust_patches[(surface_name, link_id)]
        for r, c, pose in self._get_dust_patch_poses(surface_name, link_id):
            patch_id = patch_arr[r, c]
            set_pose(patch_id, pose, self.physics_client_id)
            self._set_dust_level(patch_id, 0.0)

    def _get_dust_patch_poses(
        self, surface_name: str, link_id: int
    ) -> Iterable[tuple[int, int, Pose]]:
        surface_id = self.get_object_id_from_name(surface_name)
        aabb_min, aabb_max = p.getAABB(
            surface_id, linkIndex=link_id, physicsClientId=self.physics_client_id
        )
        x_min, y_min, _ = aabb_min
        x_max, y_max, z_min = aabb_max
        s = self.scene_spec.surface_dust_patch_size
        half_extents = self._get_dust_patch_dimensions(surface_name, link_id)
        for r, x in enumerate(np.linspace(x_min, x_max, num=s, endpoint=False)):
            for c, y in enumerate(np.linspace(y_min, y_max, num=s, endpoint=False)):
                position = (
                    x + half_extents[0],
                    y + half_extents[1],
                    z_min + half_extents[2],
                )
                yield (r, c, Pose(position))

    def _get_dust_patch_dimensions(
        self, surface_name: str, link_id: int
    ) -> tuple[float, float, float]:
        surface_id = self.get_object_id_from_name(surface_name)
        aabb_min, aabb_max = p.getAABB(
            surface_id, linkIndex=link_id, physicsClientId=self.physics_client_id
        )
        x_min, y_min, _ = aabb_min
        x_max, y_max, z_min = aabb_max
        z_max = z_min + self.scene_spec.surface_dust_height
        s = self.scene_spec.surface_dust_patch_size
        half_extents = (
            (x_max - x_min) / (2 * s),
            (y_max - y_min) / (2 * s),
            (z_max - z_min) / 2,
        )
        return half_extents

    def _get_duster_surface_aabb(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        head_pose = get_link_pose(
            self.duster_id, self.duster_head_link_id, self.physics_client_id
        )
        padding = 1e-2
        min_tf = Pose(
            (
                -self.scene_spec.duster_head_long_length + padding,
                self.scene_spec.duster_head_forward_length,
                -self.scene_spec.duster_head_up_down_length + padding,
            )
        )
        max_tf = Pose(
            (
                self.scene_spec.duster_head_long_length - padding,
                self.scene_spec.duster_head_forward_length + padding,
                padding,
            )
        )
        corner1 = multiply_poses(head_pose, min_tf).position
        corner2 = multiply_poses(head_pose, max_tf).position
        aabb_min = (
            min(corner1[0], corner2[0]),
            min(corner1[1], corner2[1]),
            min(corner1[2], corner2[2]),
        )
        aabb_max = (
            max(corner1[0], corner2[0]),
            max(corner1[1], corner2[1]),
            max(corner1[2], corner2[2]),
        )

        # Uncomment to debug.
        # from pybullet_helpers.gui import visualize_aabb
        # visualize_aabb((aabb_min, aabb_max), self.physics_client_id)

        return (aabb_min, aabb_max)

    def _set_dust_level(self, patch_id: int, level: float) -> None:
        # Transparency alone doesn't seem to render correctly, so we also change
        # the color. But we still include alpha as the store of the dust value.
        clean_color_arr = np.array(self.scene_spec.table_rgba[:3])
        dirty_color_arr = np.array(self.scene_spec.dust_color)
        color_arr = level * dirty_color_arr + (1 - level) * clean_color_arr
        color = (color_arr[0], color_arr[1], color_arr[2], level)
        p.changeVisualShape(
            patch_id, -1, rgbaColor=color, physicsClientId=self.physics_client_id
        )

    def _get_dust_level(self, patch_id: int) -> float:
        while True:
            try:
                color = p.getVisualShapeData(
                    patch_id, physicsClientId=self.physics_client_id
                )[0][7]
                break
            except p.error:
                # PyBullet rarely throws pybullet.error: "Error receiving visual
                # shape info", probably some bug in pybullet.
                pass
        assert len(color) == 4
        return color[-1]

    def _close_robot_fingers(self) -> None:
        # Very specific finger change logic, just for videos.
        assert isinstance(self.robot, KinovaGen3RobotiqGripperPyBulletRobot)
        assert self.current_held_object_id is not None
        if self.current_held_object_id == self.duster_id:
            closed_finger_state = 0.6
        else:
            closed_finger_state = 0.3
        self.robot.set_finger_state(closed_finger_state)

    def _open_robot_fingers(self) -> None:
        return self.robot.open_fingers()

    def _get_human_arm_plan(
        self,
        target_human_joints: JointPositions,
        name: str,
    ) -> list[tuple[str, JointPositions]]:
        current_human_joints = self.human.get_joint_positions()
        joint_infos = [
            self.human.joint_info_from_name(j) for j in self.human.arm_joint_names
        ]

        joint_lst = iter_between_joint_positions(
            joint_infos,
            current_human_joints,
            target_human_joints,
            num_interp_per_unit=self.scene_spec.handover_num_waypoints,
            include_start=False,
        )

        return [(name, j) for j in joint_lst]

    def get_default_half_extents(
        self, object_id: int, link_id: int
    ) -> tuple[float, float, float]:
        """Get half extents for object and link id in default pose."""
        return self._default_half_extents[(object_id, link_id)]


def _create_duster(
    duster_head_forward_length: float,
    duster_head_long_length,
    duster_head_up_down_length,
    duster_head_rgba: tuple[float, float, float, float],
    duster_pole_radius: float,
    duster_pole_height: float,
    duster_pole_rgba: tuple[float, float, float, float],
    duster_pole_offset: tuple[float, float, float],
    physics_client_id: int,
) -> tuple[int, int, int]:
    """Returns body id, link id of the head, and link id of the pole."""

    # Create duster head.
    # NOTE: orient the head so that the z axis points in the wipe direction,
    # that is, in the direction of pole -> head.
    head_base_orn = p.getQuaternionFromEuler((0, np.pi, np.pi / 2))
    half_extents = [
        duster_head_long_length,
        duster_head_forward_length,
        duster_head_up_down_length,
    ]
    head_col_shape_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=physics_client_id,
    )
    head_visual_shape_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=duster_head_rgba,
        physicsClientId=physics_client_id,
    )
    head_base_position = (0, 0, 0)

    # Create duster pole.
    pole_col_shape_id = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=duster_pole_radius,
        height=duster_pole_height,
        physicsClientId=physics_client_id,
    )
    pole_visual_shape_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=duster_pole_radius,
        length=duster_pole_height,
        rgbaColor=duster_pole_rgba,
        physicsClientId=physics_client_id,
    )
    pole_base_position = duster_pole_offset
    pole_base_orn = (0, 0, 0, 1)

    collision_shape_ids = [head_col_shape_id, pole_col_shape_id]
    visual_shape_ids = [head_visual_shape_id, pole_visual_shape_id]
    base_positions = [head_base_position, pole_base_position]
    base_orientations = [head_base_orn, pole_base_orn]

    # Combine into multibody.
    duster_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=(0, 0, 0),  # changed externally
        linkMasses=[0] * 2,
        linkCollisionShapeIndices=collision_shape_ids,
        linkVisualShapeIndices=visual_shape_ids,
        linkPositions=base_positions,
        linkOrientations=base_orientations,
        linkInertialFramePositions=[[0, 0, 0]] * 2,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
        linkParentIndices=[0] * 2,
        linkJointTypes=[p.JOINT_FIXED] * 2,
        linkJointAxis=[[0, 0, 0]] * 2,
        physicsClientId=physics_client_id,
    )

    return duster_id, 0, 1


def _create_shelf(
    color: tuple[float, float, float, float],
    shelf_width: float,
    shelf_depth: float,
    shelf_height: float,
    spacing: float,
    support_width: float,
    num_layers: int,
    shelf_texture_id: int,
    physics_client_id: int,
) -> tuple[int, set[int]]:
    """Returns the shelf ID and the link IDs of the individual shelves."""

    collision_shape_ids = []
    visual_shape_ids = []
    base_positions = []
    base_orientations = []
    link_masses = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axes = []

    # Add each shelf layer to the lists.
    for i in range(num_layers):
        layer_z = i * (spacing + shelf_height)

        col_shape_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2],
            physicsClientId=physics_client_id,
        )
        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2],
            rgbaColor=color,
            physicsClientId=physics_client_id,
        )

        collision_shape_ids.append(col_shape_id)
        visual_shape_ids.append(visual_shape_id)
        base_positions.append([0, 0, layer_z])
        base_orientations.append([0, 0, 0, 1])
        link_masses.append(0)
        link_parent_indices.append(0)
        link_joint_types.append(p.JOINT_FIXED)
        link_joint_axes.append([0, 0, 0])

    shelf_link_ids = set(range(num_layers))

    # Add vertical side supports to the lists.
    support_height = (num_layers - 1) * spacing + (num_layers) * shelf_height
    support_half_height = support_height / 2

    for x_offset in [
        -shelf_width / 2 - support_width / 2,
        shelf_width / 2 + support_width / 2,
    ]:
        support_col_shape_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[support_width / 2, shelf_depth / 2, support_half_height],
            physicsClientId=physics_client_id,
        )
        support_visual_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[support_width / 2, shelf_depth / 2, support_half_height],
            rgbaColor=color,
            physicsClientId=physics_client_id,
        )

        collision_shape_ids.append(support_col_shape_id)
        visual_shape_ids.append(support_visual_shape_id)
        base_positions.append([x_offset, 0, support_half_height - shelf_height / 2])
        base_orientations.append([0, 0, 0, 1])
        link_masses.append(0)
        link_parent_indices.append(0)
        link_joint_types.append(p.JOINT_FIXED)
        link_joint_axes.append([0, 0, 0])

    # Create the multibody with all collision and visual shapes.
    shelf_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=(0, 0, 0),  # changed externally
        linkMasses=link_masses,
        linkCollisionShapeIndices=collision_shape_ids,
        linkVisualShapeIndices=visual_shape_ids,
        linkPositions=base_positions,
        linkOrientations=base_orientations,
        linkInertialFramePositions=[[0, 0, 0]] * len(collision_shape_ids),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(collision_shape_ids),
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
        physicsClientId=physics_client_id,
    )
    for link_id in range(p.getNumJoints(shelf_id, physicsClientId=physics_client_id)):
        p.changeVisualShape(
            shelf_id,
            link_id,
            textureUniqueId=shelf_texture_id,
            physicsClientId=physics_client_id,
        )

    return shelf_id, shelf_link_ids
