"""A pybullet based environment."""

from __future__ import annotations

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
from tomsutils.spaces import EnumSpace

from multitask_personalization.envs.pybullet.pybullet_structs import (
    GripperAction,
    PyBulletAction,
    PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    HiddenTaskSpec,
    PyBulletTaskSpec,
)
from multitask_personalization.envs.pybullet.pybullet_utils import (
    create_human_from_spec,
)


class PyBulletEnv(gym.Env[PyBulletState, PyBulletAction]):
    """A pybullet based environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_spec: PyBulletTaskSpec,
        hidden_spec: HiddenTaskSpec | None = None,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self.task_spec = task_spec
        self._hidden_spec = hidden_spec
        self.render_mode = "rgb_array"

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
        robot = create_pybullet_robot(
            self.task_spec.robot_name,
            self.physics_client_id,
            base_pose=self.task_spec.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self.task_spec.initial_joints,
            fixed_base=False,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.open_fingers()
        self.robot = robot

        # Create robot stand.
        self.robot_stand_id = create_pybullet_cylinder(
            self.task_spec.robot_stand_rgba,
            radius=self.task_spec.robot_stand_radius,
            length=self.task_spec.robot_stand_length,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self.robot_stand_id, self.task_spec.robot_stand_pose, self.physics_client_id
        )

        # Save the transform between the base and stand.
        world_to_base = self.robot.get_base_pose()
        world_to_stand = get_pose(self.robot_stand_id, self.physics_client_id)
        self.robot_base_to_stand = multiply_poses(
            world_to_base.invert(), world_to_stand
        )

        # Create human.
        self.human = create_human_from_spec(
            self.task_spec.human_spec, self._rng, self.physics_client_id
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
        set_pose(
            self.wheelchair.body,
            self.task_spec.wheelchair_base_pose,
            self.physics_client_id,
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.task_spec.table_rgba,
            half_extents=self.task_spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self.task_spec.table_pose, self.physics_client_id)

        # Create object.
        self.cup_id = create_pybullet_cylinder(
            self.task_spec.object_rgba,
            self.task_spec.object_radius,
            self.task_spec.object_length,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.cup_id, self.task_spec.object_pose, self.physics_client_id)

        # Create shelf.
        self.shelf_id = _create_shelf(
            self.task_spec.shelf_rgba,
            shelf_width=self.task_spec.shelf_width,
            shelf_depth=self.task_spec.shelf_depth,
            shelf_height=self.task_spec.shelf_height,
            spacing=self.task_spec.shelf_spacing,
            support_width=self.task_spec.shelf_support_width,
            num_layers=self.task_spec.shelf_num_layers,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.shelf_id, self.task_spec.shelf_pose, self.physics_client_id)

        # Create books.
        self.book_ids: list[int] = []
        for book_rgba, book_half_extents, book_pose in zip(
            self.task_spec.book_rgbas,
            self.task_spec.book_half_extents,
            self.task_spec.book_poses,
            strict=True,
        ):
            book_id = create_pybullet_block(
                book_rgba,
                half_extents=book_half_extents,
                physics_client_id=self.physics_client_id,
            )
            set_pose(book_id, book_pose, self.physics_client_id)
            self.book_ids.append(book_id)

        # Create side table.
        self.side_table_id = create_pybullet_block(
            self.task_spec.side_table_rgba,
            half_extents=self.task_spec.side_table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self.side_table_id, self.task_spec.side_table_pose, self.physics_client_id
        )

        # Create tray.
        self.tray_id = create_pybullet_block(
            self.task_spec.tray_rgba,
            half_extents=self.task_spec.tray_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.tray_id, self.task_spec.tray_pose, self.physics_client_id)

        # Track whether the object is held, and if so, with what grasp.
        self.current_grasp_transform: Pose | None = None
        self.current_held_object_id: int | None = None

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
        for i, book_id in enumerate(self.book_ids):
            obj_to_obj_name[book_id] = f"book{i}"

        held_object = obj_to_obj_name[self.current_held_object_id]
        return PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            cup_pose,
            book_poses,
            self.current_grasp_transform,
            held_object,
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
        self.current_grasp_transform = state.grasp_transform
        obj_name_to_obj = {
            None: None,
            "cup": self.cup_id,
        }
        for i, book_id in enumerate(self.book_ids):
            obj_name_to_obj[f"book{i}"] = book_id
        self.current_held_object_id = obj_name_to_obj[state.held_object]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[PyBulletState, dict[str, Any]]:
        # Implement this in future PR.
        super().reset(seed=seed, options=options)
        return self.get_state(), {}

    def step(
        self, action: PyBulletAction
    ) -> tuple[PyBulletState, float, bool, bool, dict[str, Any]]:
        """Advance the simulator given an action."""
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
            reward, done = self._get_reward_and_done(robot_indicated_done=False)
            return self.get_state(), reward, done, False, {}
        if np.isclose(action[0], 2):
            reward, done = self._get_reward_and_done(robot_indicated_done=True)
            return self.get_state(), reward, done, False, {}
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

        reward, done = self._get_reward_and_done(robot_indicated_done=False)
        return self.get_state(), reward, done, False, {}

    def _get_reward_and_done(
        self, robot_indicated_done: bool = False
    ) -> tuple[float, bool]:
        if self._hidden_spec is None:
            raise NotImplementedError("Should not call step() in sim")
        if self.task_spec.task_objective == "hand over book":
            # Robot needs to indicate done for the handover task.
            if not robot_indicated_done:
                return 0.0, False
            # Must be holding a book.
            if self.current_held_object_id not in self.book_ids:
                return 0.0, False
            book_idx = self.book_ids.index(self.current_held_object_id)
            book_name = f"book{book_idx}"
            # Should be holding a preferred book.
            if book_name not in self._hidden_spec.book_preferences:
                return 0.0, False
            # Holding a preferred book, so check if it's being held at a
            # position that is reachable by the person.
            end_effector_position = self.robot.get_end_effector_pose().position
            reachable = self._hidden_spec.rom_model.check_position_reachable(
                np.array(end_effector_position)
            )
            if not reachable:
                return 0.0, False
            # Success!
            return 1.0, True
        raise NotImplementedError

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        target = get_link_pose(
            self.human.body,
            self.human.right_wrist,
            self.physics_client_id,
        ).position
        return capture_image(  # type: ignore
            self.physics_client_id,
            camera_target=target,
            camera_distance=self.task_spec.camera_distance,
        )

    def get_object_id_from_name(self, object_name: str) -> int:
        """Get the PyBullet object ID given a name."""
        if object_name.startswith("book"):
            idx = int(object_name[len("book") :])
            return self.book_ids[idx]
        return {
            "cup": self.cup_id,
            "table": self.table_id,
            "tray": self.tray_id,
            "shelf": self.shelf_id,
        }[object_name]

    def get_surface_ids(self) -> set[int]:
        """Get all possible surfaces in the environment."""
        surface_names = ["table", "tray", "shelf"]
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

    def get_aabb_dimensions(self, object_id: int) -> tuple[float, float, float]:
        """Get the 3D bounding box dimensions of an object."""
        (min_x, min_y, min_z), (max_x, max_y, max_z) = p.getAABB(
            object_id, -1, self.physics_client_id
        )
        return (max_x - min_x, max_y - min_y, max_z - min_z)


def _create_shelf(
    color: tuple[float, float, float, float],
    shelf_width: float,
    shelf_depth: float,
    shelf_height: float,
    spacing: float,
    support_width: float,
    num_layers: int,
    physics_client_id: int,
) -> int:

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

    return shelf_id
