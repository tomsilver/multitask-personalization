"""A shared simulator used for both the MDP and intake process."""

from __future__ import annotations

from pathlib import Path

import assistive_gym.envs
import numpy as np
import pybullet as p
from assistive_gym.envs.agents.furniture import Furniture
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.human_creation import HumanCreation
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder

from multitask_personalization.envs.pybullet.pybullet_scene_description import (
    PyBulletSceneDescription,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _PyBulletAction,
    _PyBulletState,
)


class PyBulletSimulator:
    """A shared simulator used for both MDP and intake."""

    def __init__(
        self,
        scene_description: PyBulletSceneDescription,
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:

        self._rng = np.random.default_rng(seed)
        self.scene_description = scene_description

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=0)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            self.scene_description.robot_name,
            self.physics_client_id,
            base_pose=self.scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self.scene_description.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create robot stand.
        self.robot_stand_id = create_pybullet_block(
            self.scene_description.robot_stand_rgba,
            half_extents=self.scene_description.robot_stand_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.robot_stand_id,
            self.scene_description.robot_stand_pose.position,
            self.scene_description.robot_stand_pose.orientation,
            physicsClientId=self.physics_client_id,
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
        p.resetBasePositionAndOrientation(
            self.human.body,
            self.scene_description.human_base_pose.position,
            self.scene_description.human_base_pose.orientation,
            physicsClientId=self.physics_client_id,
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
        # Now set arm joints using scene description.
        self.human.set_joint_angles(
            self.human.right_arm_joints, self.scene_description.human_joints
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
        p.resetBasePositionAndOrientation(
            furniture.body,
            self.scene_description.wheelchair_base_pose.position,
            self.scene_description.wheelchair_base_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Placeholder for full range of motion model.
        self.rom_sphere_center = get_link_pose(
            self.human.body, self.human.right_wrist, self.physics_client_id
        ).position
        self.rom_sphere_radius = 0.25
        # Visualize.
        shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.rom_sphere_radius,
            rgbaColor=(1.0, 0.0, 0.0, 0.5),
            physicsClientId=self.physics_client_id,
        )
        collision_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=1e-6,
            physicsClientId=self.physics_client_id,
        )
        self._rom_viz_id = p.createMultiBody(
            baseMass=-1,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=shape_id,
            basePosition=self.rom_sphere_center,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.physics_client_id,
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.scene_description.table_rgba,
            half_extents=self.scene_description.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.table_id,
            self.scene_description.table_pose.position,
            self.scene_description.table_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create object.
        self.object_id = create_pybullet_cylinder(
            self.scene_description.object_rgba,
            self.scene_description.object_radius,
            self.scene_description.object_length,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.object_id,
            self.scene_description.object_pose.position,
            self.scene_description.object_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Track whether the object is held, and if so, with what grasp.
        self.current_grasp_transform: Pose | None = None

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
        return _PyBulletState(
            robot_base,
            robot_joints,
            human_base,
            human_joints,
            object_pose,
            self.current_grasp_transform,
        )

    def set_state(self, state: _PyBulletState) -> None:
        """Sync the simulator with the given state."""
        p.resetBasePositionAndOrientation(
            self.robot.robot_id,
            state.robot_base.position,
            state.robot_base.orientation,
            physicsClientId=self.physics_client_id,
        )
        self.robot.set_joints(state.robot_joints)
        p.resetBasePositionAndOrientation(
            self.human.body,
            state.human_base.position,
            state.human_base.orientation,
            physicsClientId=self.physics_client_id,
        )
        self.human.set_joint_angles(
            self.human.right_arm_joints,
            state.human_joints,
        )
        p.resetBasePositionAndOrientation(
            self.object_id,
            state.object_pose.position,
            state.object_pose.orientation,
            physicsClientId=self.physics_client_id,
        )
        self.current_grasp_transform = state.grasp_transform

    def step(self, action: _PyBulletAction) -> None:
        """Advance the simulator given an action."""
        if np.isclose(action[0], 1):
            if action[1] == _GripperAction.CLOSE:
                world_to_robot = self.robot.get_end_effector_pose()
                end_effector_position = world_to_robot.position
                world_to_object = get_pose(self.object_id, self.physics_client_id)
                object_position = world_to_object.position
                dist = np.sum(
                    np.square(np.subtract(end_effector_position, object_position))
                )
                # Grasp successful.
                if dist < 1e-3:
                    self.current_grasp_transform = multiply_poses(
                        world_to_robot.invert(), world_to_object
                    )
            elif action[1] == _GripperAction.OPEN:
                self.current_grasp_transform = None
            return
        if np.isclose(action[0], 2):
            return  # will handle none case later, when the human moves
        joint_angle_delta = action[1]
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
            p.resetBasePositionAndOrientation(
                self.object_id,
                world_to_object.position,
                world_to_object.orientation,
                physicsClientId=self.physics_client_id,
            )
