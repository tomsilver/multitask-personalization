"""A domain-specific parameterized policy for pybullet."""

import abc
from typing import Any, Callable, Iterator, Sequence, TypeAlias

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
)
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.motion_planning import (
    run_base_motion_planning,
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.planning import TaskThenMotionPlanner
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from multitask_personalization.envs.pybullet.pybullet_sim import (
    PyBulletSimulator,
)
from multitask_personalization.envs.pybullet.pybullet_structs import (
    _GripperAction,
    _PyBulletAction,
    _PyBulletState,
)
from multitask_personalization.envs.pybullet.pybullet_task_spec import (
    PyBulletTaskSpec,
)
from multitask_personalization.methods.policies.parameterized_policy import (
    ParameterizedPolicy,
)
from multitask_personalization.utils import sample_spherical

##############################################################################
#                               Perception                                   #
##############################################################################

# Create generic types.
robot_type = Type("robot")
object_type = Type("obj")  # NOTE: pyperplan breaks with 'object' type name
TYPES = {robot_type, object_type}

# Create predicates.
IsMovable = Predicate("IsMovable", [object_type])
NotIsMovable = Predicate("NotIsMovable", [object_type])
On = Predicate("On", [object_type, object_type])
NothingOn = Predicate("NothingOn", [object_type])
Holding = Predicate("Holding", [robot_type, object_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
HandedOver = Predicate("HandedOver", [object_type])

PREDICATES = {
    IsMovable,
    NotIsMovable,
    On,
    NothingOn,
    Holding,
    GripperEmpty,
    HandedOver,
}


class PyBulletPerceiver(Perceiver[_PyBulletState]):
    """A perceiver for the pybullet env."""

    def __init__(self, sim: PyBulletSimulator) -> None:
        # Use the simulator for geometric computations.
        self._sim = sim

        # Create constant objects.
        self._robot = Object("robot", robot_type)
        self._book = Object("book", object_type)
        self._cup = Object("cup", object_type)
        self._tray = Object("tray", object_type)
        self._shelf = Object("shelf", object_type)
        self._table = Object("table", object_type)

        # Map from symbolic objects to PyBullet IDs in simulator.
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._book: self._sim.book_id,
            self._cup: self._sim.cup_id,
            self._tray: self._sim.tray_id,
            self._shelf: self._sim.shelf_id,
            self._table: self._sim.table_id,
        }

        # Store on relations for predicate interpretations.
        self._on_relations: set[tuple[Object, Object]] = set()

        # Create predicate interpreters.
        self._predicate_interpreters = [
            self._interpret_IsMovable,
            self._interpret_NotIsMovable,
            self._interpret_On,
            self._interpret_NothingOn,
            self._interpret_Holding,
            self._interpret_GripperEmpty,
            self._interpret_HandedOver,
        ]

    def reset(
        self,
        obs: _PyBulletState,
        info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        atoms = self._parse_observation(obs)
        objects = self._get_objects()
        goal = self._get_goal()
        return objects, atoms, goal

    def step(self, obs: _PyBulletState) -> set[GroundAtom]:
        atoms = self._parse_observation(obs)
        return atoms

    def _get_objects(self) -> set[Object]:
        return set(self._pybullet_ids)

    def _set_sim_from_obs(self, obs: _PyBulletState) -> None:
        self._sim.set_state(obs)

    def _get_goal(self) -> set[GroundAtom]:
        task_objective = self._sim.task_spec.task_objective
        if task_objective == "hand over cup":
            return {GroundAtom(HandedOver, [self._cup])}
        if task_objective == "hand over book":
            return {GroundAtom(HandedOver, [self._book])}
        if task_objective == "place book on tray":
            return {GroundAtom(On, [self._book, self._tray])}
        raise NotImplementedError

    def _parse_observation(self, obs: _PyBulletState) -> set[GroundAtom]:

        # Sync the simulator so that interpretation functions can use PyBullet
        # direction.
        self._set_sim_from_obs(obs)

        # Compute which things are on which other things.
        self._on_relations = self._get_on_relations_from_sim()

        # Create current atoms.
        atoms: set[GroundAtom] = set()
        for interpret_fn in self._predicate_interpreters:
            atoms.update(interpret_fn())

        return atoms

    def _get_on_relations_from_sim(self) -> set[tuple[Object, Object]]:
        on_relations = set()
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}
        for obj1 in candidates:
            obj1_pybullet_id = self._pybullet_ids[obj1]
            pose1 = get_pose(obj1_pybullet_id, self._sim.physics_client_id)
            for obj2 in candidates:
                if obj1 == obj2:
                    continue
                obj2_pybullet_id = self._pybullet_ids[obj2]
                # Check if obj1 pose is above obj2 pose.
                pose2 = get_pose(obj2_pybullet_id, self._sim.physics_client_id)
                if pose1.position[2] < pose2.position[2]:
                    continue
                # Check for contact.
                if check_body_collisions(
                    obj1_pybullet_id, obj2_pybullet_id, self._sim.physics_client_id
                ):
                    on_relations.add((obj1, obj2))
        return on_relations

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        movable_objs = {self._book, self._cup}
        return {GroundAtom(IsMovable, [o]) for o in movable_objs}

    def _interpret_NotIsMovable(self) -> set[GroundAtom]:
        objs = {o for o in self._get_objects() if o.is_instance(object_type)}
        movable_atoms = self._interpret_IsMovable()
        movable_objs = {a.objects[0] for a in movable_atoms}
        not_movable_objs = objs - movable_objs
        return {GroundAtom(NotIsMovable, [o]) for o in not_movable_objs}

    def _interpret_On(self) -> set[GroundAtom]:
        return {GroundAtom(On, r) for r in self._on_relations}

    def _interpret_NothingOn(self) -> set[GroundAtom]:
        objs = {o for o in self._get_objects() if o.is_instance(object_type)}
        for _, bot in self._on_relations:
            objs.discard(bot)
        return {GroundAtom(NothingOn, [o]) for o in objs}

    def _interpret_Holding(self) -> set[GroundAtom]:
        if self._sim.current_held_object_id is not None:
            pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
            held_obj = pybullet_id_to_obj[self._sim.current_held_object_id]
            return {GroundAtom(Holding, [self._robot, held_obj])}
        return set()

    def _interpret_GripperEmpty(self) -> set[GroundAtom]:
        if not self._sim.current_grasp_transform:
            return {GroundAtom(GripperEmpty, [self._robot])}
        return set()

    def _interpret_HandedOver(self) -> set[GroundAtom]:
        handed_over_objs: set[Object] = set()
        handover_padding = 1e-2
        for obj in [self._cup, self._book]:
            obj_pybullet_id = self._pybullet_ids[obj]
            pose = get_pose(obj_pybullet_id, self._sim.physics_client_id)
            dist = np.sqrt(
                np.sum(np.subtract(pose.position, self._sim.rom_sphere_center) ** 2)
            )
            if dist < self._sim.rom_sphere_radius + handover_padding:
                handed_over_objs.add(obj)
        return {GroundAtom(HandedOver, [o]) for o in handed_over_objs}


##############################################################################
#                                Operators                                   #
##############################################################################

Robot = Variable("?robot", robot_type)
Obj = Variable("?obj", object_type)
Surface = Variable("?surface", object_type)
PickOperator = LiftedOperator(
    "Pick",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NothingOn, [Obj]),
        LiftedAtom(On, [Obj, Surface]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
    },
)

PlaceOperator = LiftedOperator(
    "Place",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

HandOverOperator = LiftedOperator(
    "HandOver",
    [Robot, Obj],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
    },
    add_effects={
        LiftedAtom(HandedOver, [Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

OPERATORS = {PickOperator, PlaceOperator, HandOverOperator}

##############################################################################
#                                  Skills                                    #
##############################################################################


PyBulletSkillHyperparameters: TypeAlias = float  # just a radius for now, more to come


class PyBulletSkill(LiftedOperatorSkill[_PyBulletState, _PyBulletAction]):
    """Shared functionality between skills."""

    def __init__(
        self,
        sim: PyBulletSimulator,
        get_skill_hyperparameters: Callable[[], PyBulletSkillHyperparameters],
        max_motion_planning_time: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._sim = sim
        self._get_skill_hyperparameters = get_skill_hyperparameters
        self._task_spec = sim.task_spec
        self._max_motion_planning_time = max_motion_planning_time
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_plan: list[_PyBulletAction] = []

        # Create constant objects.
        self._robot = Object("robot", robot_type)
        self._book = Object("book", object_type)
        self._cup = Object("cup", object_type)
        self._tray = Object("tray", object_type)
        self._shelf = Object("shelf", object_type)
        self._table = Object("table", object_type)

        # Map from symbolic objects to PyBullet IDs in simulator.
        self._pybullet_ids = {
            self._book: self._sim.book_id,
            self._cup: self._sim.cup_id,
            self._tray: self._sim.tray_id,
            self._shelf: self._sim.shelf_id,
            self._table: self._sim.table_id,
        }

    def reset(self, ground_operator: GroundOperator) -> None:
        self._current_plan = []
        return super().reset(ground_operator)

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: _PyBulletState
    ) -> _PyBulletAction:
        if not self._current_plan:
            kinematic_state = self._obs_to_kinematic_state(obs)
            kinematic_plan = self._get_kinematic_plan_given_objects(
                objects, kinematic_state
            )
            self._current_plan = self._kinematic_plan_to_action_plan(kinematic_plan)
        return self._current_plan.pop(0)

    @abc.abstractmethod
    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState]:
        """Generate a plan given an initial kinematic state and objects."""

    def _kinematic_plan_to_action_plan(
        self, kinematic_plan: list[KinematicState]
    ) -> list[_PyBulletAction]:
        action_plan: list[_PyBulletAction] = []
        for s0, s1 in zip(kinematic_plan[:-1], kinematic_plan[1:], strict=True):
            actions = self._kinematic_transition_to_actions(s0, s1)
            action_plan.extend(actions)
        return action_plan

    def _kinematic_transition_to_actions(
        self, state: KinematicState, next_state: KinematicState
    ) -> list[_PyBulletAction]:
        assert state.robot_base_pose is not None
        assert next_state.robot_base_pose is not None
        base_delta = (
            next_state.robot_base_pose.position[0] - state.robot_base_pose.position[0],
            next_state.robot_base_pose.position[1] - state.robot_base_pose.position[1],
            next_state.robot_base_pose.rpy[2] - state.robot_base_pose.rpy[2],
        )
        joint_delta = np.subtract(next_state.robot_joints, state.robot_joints)
        delta = list(base_delta) + list(joint_delta[:7])
        actions: list[_PyBulletAction] = [(0, delta)]
        if next_state.attachments and not state.attachments:
            actions.append((1, _GripperAction.CLOSE))
        elif state.attachments and not next_state.attachments:
            actions.append((1, _GripperAction.OPEN))
        return actions

    def _object_to_pybullet_id(self, obj: Object) -> int:
        return self._pybullet_ids[obj]

    def _obs_to_kinematic_state(self, obs: _PyBulletState) -> KinematicState:
        robot_joints = obs.robot_joints
        object_poses = {
            self._sim.cup_id: obs.object_pose,
            self._sim.book_id: obs.book_pose,
            self._sim.table_id: self._task_spec.table_pose,
            self._sim.shelf_id: self._task_spec.shelf_pose,
            self._sim.tray_id: self._task_spec.tray_pose,
        }
        attachments: dict[int, Pose] = {}
        if obs.held_object == "cup":
            assert obs.grasp_transform is not None
            attachments[self._sim.cup_id] = obs.grasp_transform
        if obs.held_object == "book":
            assert obs.grasp_transform is not None
            attachments[self._sim.book_id] = obs.grasp_transform
        return KinematicState(robot_joints, object_poses, attachments, obs.robot_base)


class PickSkill(PyBulletSkill):
    """Skill for picking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState]:

        _, obj, surface = objects
        obj_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)

        collision_ids = {
            self._sim.table_id,
            self._sim.human.body,
            self._sim.wheelchair.body,
            self._sim.shelf_id,
            self._sim.tray_id,
            self._sim.side_table_id,
        }

        def _grasp_generator() -> Iterator[Pose]:
            while True:
                angle_offset = self._rng.uniform(-np.pi, np.pi)
                relative_pose = get_poses_facing_line(
                    axis=(0.0, 0.0, 1.0),
                    point_on_line=(0.0, 0.0, 0),
                    radius=1e-3,
                    num_points=1,
                    angle_offset=angle_offset,
                )[0]
                yield relative_pose

        kinematic_plan = get_kinematic_plan_to_pick_object(
            state,
            self._sim.robot,
            obj_id,
            surface_id,
            collision_ids,
            grasp_generator=_grasp_generator(),
        )

        assert kinematic_plan is not None
        return kinematic_plan


class PlaceSkill(PyBulletSkill):
    """Skill for placing."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState]:
        _, obj, surface = objects

        obj_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {obj_id}

        base_to_platform = self._sim.robot_base_to_stand
        surface_extents = self._get_aabb_dimensions(surface_id)
        object_extents = self._get_aabb_dimensions(obj_id)
        placement_lb = (
            -surface_extents[0] / 2 + object_extents[0] / 2,
            -surface_extents[1] / 2 + object_extents[1] / 2,
            surface_extents[2] / 2 + object_extents[2] / 2,
        )
        placement_ub = (
            surface_extents[0] / 2 - object_extents[0] / 2,
            surface_extents[1] / 2 - object_extents[1] / 2,
            surface_extents[2] / 2 + object_extents[2] / 2,
        )

        def _placement_generator() -> Iterator[Pose]:
            # Sample on the surface of the table.
            while True:
                yield Pose(tuple(self._rng.uniform(placement_lb, placement_ub)))

        state.set_pybullet(self._sim.robot)
        current_base_pose = self._sim.robot.get_base_pose()
        surface_pose = get_pose(surface_id, self._sim.physics_client_id)

        # Use pre-defined staging base pose for now. Generalize this later.
        target_base_pose = Pose(
            (
                surface_pose.position[0] - surface_extents[0],
                surface_pose.position[1] - surface_extents[1],
                0.0,
            ),
            orientation=current_base_pose.orientation,
        )

        base_motion_plan = run_base_motion_planning(
            self._sim.robot,
            current_base_pose,
            target_base_pose,
            position_lower_bounds=self._task_spec.world_lower_bounds[:2],
            position_upper_bounds=self._task_spec.world_upper_bounds[:2],
            collision_bodies=collision_ids,
            seed=self._seed,
            physics_client_id=self._sim.physics_client_id,
            platform=self._sim.robot_stand_id,
            held_object=obj_id,
            base_link_to_held_obj=state.attachments[obj_id],
        )

        assert base_motion_plan is not None

        kinematic_plan: list[KinematicState] = []
        for base_pose in base_motion_plan:
            kinematic_plan.append(state.copy_with(robot_base_pose=base_pose))

        # Also update the platform in sim.
        state = kinematic_plan[-1]
        state.set_pybullet(self._sim.robot)
        assert state.robot_base_pose is not None
        platform_pose = multiply_poses(state.robot_base_pose, base_to_platform)
        set_pose(self._sim.robot_stand_id, platform_pose, self._sim.physics_client_id)

        # Prepare to place.
        placement_kinematic_plan = get_kinematic_plan_to_place_object(
            state,
            self._sim.robot,
            obj_id,
            surface_id,
            collision_ids,
            _placement_generator(),
            max_motion_planning_time=self._max_motion_planning_time,
        )
        assert placement_kinematic_plan is not None
        kinematic_plan.extend(placement_kinematic_plan)

        return kinematic_plan

    def _get_aabb_dimensions(self, obj_id: int) -> tuple[float, float, float]:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = p.getAABB(
            obj_id, -1, self._sim.physics_client_id
        )
        return (max_x - min_x, max_y - min_y, max_z - min_z)


class HandoverSkill(PyBulletSkill):
    """Skill for handover."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return HandOverOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState]:
        _, obj = objects

        obj_id = self._object_to_pybullet_id(obj)
        collision_ids = set(state.object_poses) - {obj_id}

        # Sample a reachable handover pose.
        handover_pose: Pose | None = None
        skill_hyperparameters = self._get_skill_hyperparameters()
        while True:
            candidate = self._sample_handover_pose(skill_hyperparameters)
            try:
                inverse_kinematics(self._sim.robot, candidate)
                handover_pose = candidate
                break
            except InverseKinematicsError:
                continue
        assert handover_pose is not None

        # Motion plan to hand over.
        state.set_pybullet(self._sim.robot)
        robot_joint_plan = run_smooth_motion_planning_to_pose(
            handover_pose,
            self._sim.robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed,
            max_time=self._max_motion_planning_time,
            held_object=obj_id,
            base_link_to_held_obj=state.attachments[obj_id],
        )
        assert robot_joint_plan is not None
        kinematic_plan: list[KinematicState] = []
        for robot_joints in robot_joint_plan:
            kinematic_plan.append(state.copy_with(robot_joints=robot_joints))

        return kinematic_plan

    def _get_aabb_dimensions(self, obj_id: int) -> tuple[float, float, float]:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = p.getAABB(
            obj_id, -1, self._sim.physics_client_id
        )
        return (max_x - min_x, max_y - min_y, max_z - min_z)

    def _sample_handover_pose(self, rom_radius: float) -> Pose:
        # Get the sphere center from the simulator.
        center = get_link_pose(
            self._sim.human.body,
            self._sim.human.right_wrist,
            self._sim.physics_client_id,
        ).position
        position = sample_spherical(center, rom_radius, self._rng)
        orientation = (
            0.8522037863731384,
            0.4745013415813446,
            -0.01094298530369997,
            0.22017613053321838,
        )
        pose = Pose(position, orientation)
        return pose


SKILLS = {PickSkill, PlaceSkill, HandoverSkill}

##############################################################################
#                                 Planner                                    #
##############################################################################


class PyBulletParameterizedPolicy(
    ParameterizedPolicy[_PyBulletState, _PyBulletAction, PyBulletSkillHyperparameters]
):
    """A domain-specific hyper-parameterized policy for pybullet."""

    def __init__(
        self,
        task_spec: PyBulletTaskSpec,
        seed: int = 0,
        max_motion_planning_time: float = 1.0,
        planner_id: str = "pyperplan",
    ) -> None:
        super().__init__()
        # Create a shared simulator for planning and perception.
        self._sim = PyBulletSimulator(task_spec, use_gui=False)
        # Create perceiver.
        self._perceiver = PyBulletPerceiver(self._sim)
        # Give hyperparameter access to skills.
        self._skills: set[PyBulletSkill] = {
            skill_cls(
                self._sim,  # type: ignore
                self._get_skill_hyperparameters,
                max_motion_planning_time=max_motion_planning_time,
                seed=seed,
            )
            for skill_cls in SKILLS
        }
        # Create planner.
        self._planner = TaskThenMotionPlanner(
            TYPES,
            PREDICATES,
            self._perceiver,
            OPERATORS,
            self._skills,  # type: ignore
            planner_id=planner_id,
        )
        self._planner_called = False

    def reset(self, task_id: str, parameters: PyBulletSkillHyperparameters) -> None:
        super().reset(task_id, parameters)
        self._planner_called = False

    def step(self, state: _PyBulletState) -> _PyBulletAction:
        if not self._planner_called:
            self._planner.reset(state, {})
            self._planner_called = True
        return self._planner.step(state)

    def _get_skill_hyperparameters(self) -> PyBulletSkillHyperparameters:
        assert self._current_parameters is not None
        return self._current_parameters
