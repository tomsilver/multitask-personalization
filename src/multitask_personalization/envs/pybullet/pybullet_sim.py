"""A shared simulator used for both the MDP and intake process."""

from __future__ import annotations

from pathlib import Path

import assistive_gym.envs
import numpy as np
import pybullet as p
from assistive_gym.envs.agents.furniture import Furniture
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.human_creation import HumanCreation
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder

from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _PyBulletAction,
    _PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)


class PyBulletSimulator:
    """A shared simulator used for both MDP and intake."""

    def __init__(
        self,
        task_spec: PyBulletTaskSpec,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self.task_spec = task_spec

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
        robot.close_fingers()
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

        # Create human.
        human_creation = HumanCreation(
            self.physics_client_id, np_random=self._rng, cloth=False
        )
        self.human = Human([], controllable=False)
        self.human.init(
            human_creation,
            static_human_base=True,
            impairment="none",
            gender="male",
            config=None,
            id=self.physics_client_id,
            np_random=self._rng,
        )
        set_pose(
            self.human.body, self.task_spec.human_base_pose, self.physics_client_id
        )

        # Use some default joint positions from assistive gym first.
        joints_positions = [
            (self.human.j_right_elbow, -90),
            (self.human.j_left_elbow, -90),
            (self.human.j_right_hip_x, -90),
            (self.human.j_right_knee, 80),
            (self.human.j_left_hip_x, -90),
            (self.human.j_left_knee, 80),
            (self.human.j_head_x, 0.0),
            (self.human.j_head_y, 0.0),
            (self.human.j_head_z, 0.0),
        ]
        self.human.setup_joints(
            joints_positions, use_static_joints=True, reactive_force=None
        )
        # Now set arm joints using task spec.
        self.human.set_joint_angles(
            self.human.right_arm_joints, self.task_spec.human_joints
        )

        # Create wheelchair.
        furniture = Furniture()
        directory = Path(assistive_gym.envs.__file__).parent / "assets"
        assert directory.exists()
        furniture.init(
            "wheelchair",
            directory,
            self.physics_client_id,
            self._rng,
            wheelchair_mounted=False,
        )
        set_pose(
            furniture.body, self.task_spec.wheelchair_base_pose, self.physics_client_id
        )

        # Placeholder for full range of motion model.
        self.rom_sphere_center = get_link_pose(
            self.human.body, self.human.right_wrist, self.physics_client_id
        ).position
        self.rom_sphere_radius = 0.25
        # Visualize.
        # shape_id = p.createVisualShape(
        #     shapeType=p.GEOM_SPHERE,
        #     radius=self.rom_sphere_radius,
        #     rgbaColor=(1.0, 0.0, 0.0, 0.5),
        #     physicsClientId=self.physics_client_id,
        # )
        # collision_id = p.createCollisionShape(
        #     shapeType=p.GEOM_SPHERE,
        #     radius=1e-6,
        #     physicsClientId=self.physics_client_id,
        # )
        # self._rom_viz_id = p.createMultiBody(
        #     baseMass=-1,
        #     baseCollisionShapeIndex=collision_id,
        #     baseVisualShapeIndex=shape_id,
        #     basePosition=self.rom_sphere_center,
        #     baseOrientation=[0, 0, 0, 1],
        #     physicsClientId=self.physics_client_id,
        # )

        # Create table.
        self.table_id = create_pybullet_block(
            self.task_spec.table_rgba,
            half_extents=self.task_spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self.task_spec.table_pose, self.physics_client_id)

        # Create object.
        self.object_id = create_pybullet_cylinder(
            self.task_spec.object_rgba,
            self.task_spec.object_radius,
            self.task_spec.object_length,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.object_id, self.task_spec.object_pose, self.physics_client_id)

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

        # Create book.
        self.book_id = create_pybullet_block(
            self.task_spec.book_rgba,
            half_extents=self.task_spec.book_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.book_id, self.task_spec.book_pose, self.physics_client_id)

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

    def get_state(self) -> _PyBulletState:
        """Get the underlying state from the simulator."""
        robot_base = self.robot.get_base_pose()
        robot_joints = self.robot.get_joint_positions()
        human_base = get_link_pose(self.human.body, -1, self.physics_client_id)
        human_joints = self.human.get_joint_angles(self.human.right_arm_joints)
        object_pose = get_pose(self.object_id, self.physics_client_id)
        book_pose = get_pose(self.book_id, self.physics_client_id)
        held_object = {
            None: None,
            self.object_id: "cup",
            self.book_id: "book",
        }[self.current_held_object_id]
        return _PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            object_pose,
            book_pose,
            self.current_grasp_transform,
            held_object,
        )

    def set_state(self, state: _PyBulletState) -> None:
        """Sync the simulator with the given state."""
        set_pose(self.robot.robot_id, state.robot_base, self.physics_client_id)
        self.robot.set_joints(state.robot_joints)
        set_pose(self.human.body, state.human_base, self.physics_client_id)
        self.human.set_joint_angles(
            self.human.right_arm_joints,
            state.human_joints,
        )
        set_pose(self.object_id, state.object_pose, self.physics_client_id)
        set_pose(self.book_id, state.book_pose, self.physics_client_id)
        self.current_grasp_transform = state.grasp_transform
        self.current_held_object_id = {
            None: None,
            "cup": self.object_id,
            "book": self.book_id,
        }[state.held_object]

    def step(self, action: _PyBulletAction) -> None:
        """Advance the simulator given an action."""
        if np.isclose(action[0], 1):
            if action[1] == _GripperAction.CLOSE:
                world_to_robot = self.robot.get_end_effector_pose()
                end_effector_position = world_to_robot.position
                for object_id in [self.object_id, self.book_id]:
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
            elif action[1] == _GripperAction.OPEN:
                self.current_grasp_transform = None
                self.current_held_object_id = None
            return
        if np.isclose(action[0], 2):
            return  # will handle none case later, when the human moves
        joint_action = list(action[1])  # type: ignore
        base_position_delta = joint_action[:3]
        joint_angle_delta = joint_action[3:]
        del base_position_delta  # TODO
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
