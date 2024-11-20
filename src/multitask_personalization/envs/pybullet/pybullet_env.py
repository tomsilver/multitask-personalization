"""A pybullet based environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import assistive_gym.envs
import gymnasium as gym
import numpy as np
import pybullet as p
from assistive_gym.envs.agents.furniture import Furniture
from gymnasium.core import RenderFrame
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
)
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
from tomsutils.llm import LargeLanguageModel
from tomsutils.spaces import EnumSpace
from tomsutils.utils import render_textbox_on_image

from multitask_personalization.envs.pybullet.pybullet_human_spec import (
    create_human_from_spec,
)
from multitask_personalization.envs.pybullet.pybullet_missions import (
    HandOverBookMission,
    StoreHeldObjectMission,
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

        # Create action space.
        self.action_space = gym.spaces.OneOf(
            (
                gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
                EnumSpace([GripperAction.OPEN, GripperAction.CLOSE]),
                EnumSpace([None]),
            )
        )

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=0)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        self.robot = self._create_robot(self.scene_spec, self.physics_client_id)

        # Create robot stand.
        self.robot_stand_id = create_pybullet_cylinder(
            self.scene_spec.robot_stand_rgba,
            radius=self.scene_spec.robot_stand_radius,
            length=self.scene_spec.robot_stand_length,
            physics_client_id=self.physics_client_id,
        )

        # Create human.
        self.human = create_human_from_spec(
            self.scene_spec.human_spec, self._rng, self.physics_client_id
        )

        # Create wheelchair.
        self.wheelchair = Furniture()
        directory = Path(assistive_gym.envs.__file__).parent / "assets"
        assert directory.exists()
        self.wheelchair.init(
            "wheelchair",
            directory,
            self.physics_client_id,
            self._rng,
            wheelchair_mounted=False,
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.scene_spec.table_rgba,
            half_extents=self.scene_spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Create cup.
        self.cup_id = create_pybullet_cylinder(
            self.scene_spec.object_rgba,
            self.scene_spec.object_radius,
            self.scene_spec.object_length,
            physics_client_id=self.physics_client_id,
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
            physics_client_id=self.physics_client_id,
        )

        # Create books.
        self.book_ids: list[int] = []
        self.book_descriptions: list[str] = []  # created in reset()
        for book_rgba, book_half_extents in zip(
            self.scene_spec.book_rgbas,
            self.scene_spec.book_half_extents,
            strict=True,
        ):
            book_id = create_pybullet_block(
                book_rgba,
                half_extents=book_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self.book_ids.append(book_id)

        # Create side table.
        self.side_table_id = create_pybullet_block(
            self.scene_spec.side_table_rgba,
            half_extents=self.scene_spec.side_table_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Create tray.
        self.tray_id = create_pybullet_block(
            self.scene_spec.tray_rgba,
            half_extents=self.scene_spec.tray_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Track whether the object is held, and if so, with what grasp.
        self.current_grasp_transform: Pose | None = None
        self.current_held_object_id: int | None = None

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

        # Uncomment for debug / development.
        # while True:
        #     p.stepSimulation(self.physics_client_id)

    def get_state(self) -> PyBulletState:
        """Get the underlying state from the simulator."""
        robot_base = self.robot.get_base_pose()
        robot_joints = self.robot.get_joint_positions()
        human_base = get_link_pose(self.human.body, -1, self.physics_client_id)
        human_joints = self.human.get_joint_angles(self.human.right_arm_joints)
        cup_pose = get_pose(self.cup_id, self.physics_client_id)
        book_poses = [
            get_pose(book_id, self.physics_client_id) for book_id in self.book_ids
        ]
        obj_to_obj_name = {
            None: None,
            self.cup_id: "cup",
        }
        for book_id, book_description in zip(
            self.book_ids, self.book_descriptions, strict=True
        ):
            obj_to_obj_name[book_id] = book_description

        held_object = obj_to_obj_name[self.current_held_object_id]
        return PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            cup_pose,
            book_poses,
            self.book_descriptions,
            self.current_grasp_transform,
            held_object,
            self.current_human_text,
        )

    def set_state(self, state: PyBulletState) -> None:
        """Sync the simulator with the given state."""
        set_pose(self.robot.robot_id, state.robot_base, self.physics_client_id)
        self.robot.set_joints(state.robot_joints)
        stand_pose = multiply_poses(state.robot_base, self.robot_base_to_stand)
        set_pose(self.robot_stand_id, stand_pose, self.physics_client_id)
        set_pose(self.human.body, state.human_base, self.physics_client_id)
        self.human.set_joint_angles(
            self.human.right_arm_joints,
            state.human_joints,
        )
        set_pose(self.cup_id, state.cup_pose, self.physics_client_id)
        for book_id, book_pose in zip(self.book_ids, state.book_poses, strict=True):
            set_pose(book_id, book_pose, self.physics_client_id)
        self.book_descriptions = state.book_descriptions
        self.current_grasp_transform = state.grasp_transform
        self.current_human_text = state.human_text
        obj_name_to_obj = {
            None: None,
            "cup": self.cup_id,
        }
        for book_id, book_description in zip(
            self.book_ids, self.book_descriptions, strict=True
        ):
            obj_name_to_obj[book_description] = book_id
        self.current_held_object_id = obj_name_to_obj[state.held_object]

    def _reset_from_scene_spec(self) -> None:

        # Reset robot.
        self.robot.set_base(self.scene_spec.robot_base_pose)
        self.robot.set_joints(self.scene_spec.initial_joints)
        self.robot.open_fingers()

        # Reset robot stand.
        set_pose(
            self.robot_stand_id,
            self.scene_spec.robot_stand_pose,
            self.physics_client_id,
        )

        # Reset wheelchair.
        set_pose(
            self.wheelchair.body,
            self.scene_spec.wheelchair_base_pose,
            self.physics_client_id,
        )

        # Reset table.
        set_pose(self.table_id, self.scene_spec.table_pose, self.physics_client_id)

        # Reset cup.
        set_pose(self.cup_id, self.scene_spec.object_pose, self.physics_client_id)

        # Reset shelf.
        set_pose(self.shelf_id, self.scene_spec.shelf_pose, self.physics_client_id)

        # Reset books.
        for book_pose, book_id in zip(
            self.scene_spec.book_poses,
            self.book_ids,
            strict=True,
        ):
            set_pose(book_id, book_pose, self.physics_client_id)

        # Reset side table.
        set_pose(
            self.side_table_id, self.scene_spec.side_table_pose, self.physics_client_id
        )

        # Reset tray.
        set_pose(self.tray_id, self.scene_spec.tray_pose, self.physics_client_id)

        # Reset held object statuses.
        self.current_grasp_transform = None
        self.current_held_object_id = None

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
            seed=int(self._book_llm_rng.integers(0, 2**31 - 1)),
        )

        # Randomize robot mission.
        self._current_mission = self._generate_mission()

        # Tell the robot its mission.
        self.current_human_text = self._current_mission.get_mission_command()

        return self.get_state(), self._get_info()

    def _step_simulator(self, action: PyBulletAction) -> None:
        # Opening or closing the gripper.
        if np.isclose(action[0], 1):
            if action[1] == GripperAction.CLOSE:
                world_to_robot = self.robot.get_end_effector_pose()
                end_effector_position = world_to_robot.position
                for object_id in [self.cup_id] + self.book_ids:
                    world_to_object = get_pose(object_id, self.physics_client_id)
                    object_position = world_to_object.position
                    dist = np.sum(
                        np.square(np.subtract(end_effector_position, object_position))
                    )
                    # Grasp successful.
                    if dist < 1e-3:
                        self.current_grasp_transform = multiply_poses(
                            world_to_robot.invert(), world_to_object
                        )
                        self.current_held_object_id = object_id
            elif action[1] == GripperAction.OPEN:
                self.current_grasp_transform = None
                self.current_held_object_id = None
            return
        # Robot indicating done.
        if np.isclose(action[0], 2):
            return
        # Moving the robot.
        joint_action = list(action[1])  # type: ignore
        base_position_delta = joint_action[:3]
        joint_angle_delta = joint_action[3:]
        # Update the robot base.
        world_to_base = self.robot.get_base_pose()
        dx, dy, dyaw = base_position_delta
        x, y, z = world_to_base.position
        roll, pitch, yaw = world_to_base.rpy
        next_base = Pose.from_rpy((x + dx, y + dy, z), (roll, pitch, yaw + dyaw))
        self.robot.set_base(next_base)
        next_stand_pose = multiply_poses(next_base, self.robot_base_to_stand)
        set_pose(self.robot_stand_id, next_stand_pose, self.physics_client_id)
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
        return

    def step(
        self, action: PyBulletAction
    ) -> tuple[PyBulletState, float, bool, bool, dict[str, Any]]:
        # Advance the simulator.
        state = self.get_state()
        self._step_simulator(action)

        # Advance the mission.
        assert self._current_mission is not None
        self.current_human_text, self._user_satisfaction = self._current_mission.step(
            state, action
        )

        # Start a new mission if the current one is complete.
        if self._current_mission.check_complete(state, action):
            self._current_mission = self._generate_mission()
            # Tell the robot its new mission.
            mission_description = self._current_mission.get_mission_command()
            if self.current_human_text is None:
                self.current_human_text = mission_description
            else:
                self.current_human_text += "\n" + mission_description

        if self.current_human_text:
            logging.info(f"Human says: {self.current_human_text}")

        # Return the next state and default gym API stuff.
        return self.get_state(), 0.0, False, False, self._get_info()

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
        target = get_link_pose(
            self.human.body,
            self.human.right_wrist,
            self.physics_client_id,
        ).position
        img = capture_image(
            self.physics_client_id,
            camera_target=target,
            camera_distance=self.scene_spec.camera_distance,
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

    def get_object_id_from_name(self, object_name: str) -> int:
        """Get the PyBullet object ID given a name."""
        if object_name in self.book_descriptions:
            idx = self.book_descriptions.index(object_name)
            return self.book_ids[idx]
        return {
            "cup": self.cup_id,
            "table": self.table_id,
            "tray": self.tray_id,
            "shelf": self.shelf_id,
        }[object_name]

    def get_surface_names(self) -> set[str]:
        """Get all possible surfaces in the environment."""
        return {"table", "tray", "shelf"}

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
            self.human.body,
            self.wheelchair.body,
            self.shelf_id,
            self.tray_id,
            self.side_table_id,
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

    def _generate_mission(self) -> PyBulletMission:
        state = self.get_state()
        seed = int(self._mission_rng.integers(0, 2**31 - 1))
        assert self._hidden_spec is not None
        # NOTE: don't use the real robot / real environment inside the missions
        # in case they want to do things like use robot FK.
        physics_client_id = p.connect(p.DIRECT)
        sim_robot = self._create_robot(self.scene_spec, physics_client_id)
        possible_missions: list[PyBulletMission] = [
            HandOverBookMission(
                self.book_descriptions,
                sim_robot,
                self._hidden_spec.rom_model,
                self._hidden_spec.book_preferences,
                self._llm,
                seed=seed,
            ),
            StoreHeldObjectMission(),
        ]
        for mission in possible_missions:
            if mission.check_initiable(state):
                return mission
        raise NotImplementedError

    def _generate_book_descriptions(self, num_books: int, seed: int) -> list[str]:
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


def _create_shelf(
    color: tuple[float, float, float, float],
    shelf_width: float,
    shelf_depth: float,
    shelf_height: float,
    spacing: float,
    support_width: float,
    num_layers: int,
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
    support_height = (num_layers - 1) * spacing + (num_layers - 1) * shelf_height
    support_half_height = support_height / 2

    for x_offset in [
        -shelf_width / 2 + support_width / 2,
        shelf_width / 2 - support_width / 2,
    ]:
        for y_offset in [
            -shelf_depth / 2 + support_width / 2,
            shelf_depth / 2 - support_width / 2,
        ]:
            support_col_shape_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[support_width / 2, support_width / 2, support_half_height],
                physicsClientId=physics_client_id,
            )
            support_visual_shape_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[support_width / 2, support_width / 2, support_half_height],
                rgbaColor=color,
                physicsClientId=physics_client_id,
            )

            collision_shape_ids.append(support_col_shape_id)
            visual_shape_ids.append(support_visual_shape_id)
            base_positions.append([x_offset, y_offset, support_half_height])
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
        linkLowerLimits=[1] * len(collision_shape_ids),
        linkUpperLimits=[-1] * len(collision_shape_ids),
        physicsClientId=physics_client_id,
    )

    return shelf_id, shelf_link_ids
