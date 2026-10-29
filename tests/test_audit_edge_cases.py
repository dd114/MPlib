import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import mplib
from mplib import Planner, Pose
from mplib.collision_detection.fcl import (
    CollisionRequest,
    FCLModel,
    FCLObject,
    collide,
    distance,
)
from mplib.kinematics.pinocchio import PinocchioModel
from mplib.planning.ompl import FixedJoint, OMPLPlanner

FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
PANDA_SPEC = {
    "urdf": f"{FILE_ABS_DIR}/../data/panda/panda.urdf",
    "srdf": f"{FILE_ABS_DIR}/../data/panda/panda.srdf",
    "move_group": "panda_hand",
}
PANDA_ON_RAIL_SPEC = {
    "urdf": f"{FILE_ABS_DIR}/../data/panda/panda_on_rail.urdf",
    "srdf": f"{FILE_ABS_DIR}/../data/panda/panda_on_rail.srdf",
    "move_group": "panda_hand",
}


class TestAuditEdgeCases(unittest.TestCase):
    """
    Regression suite verifying resolution of edge cases, defects,
    and robustness guarantees identified in the technical audit.
    """

    def setUp(self):
        self.planner = Planner(**PANDA_SPEC)

    # =========================================================================
    # 1. High-DoF articulated systems, passive/fixed joints, and sub-chains
    # =========================================================================

    def test_pad_move_group_qpos_corrupts_non_prefix_move_group(self):
        """
        Verify pad_move_group_qpos properly indexes non-prefix move groups using
        move_group_joint_indices instead of assuming move group is always indices 0..ndim-1.
        """
        user_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        planner = Planner(
            urdf=PANDA_SPEC["urdf"],
            srdf=PANDA_SPEC["srdf"],
            move_group="panda_hand",
            user_joint_names=user_joint_names,
        )
        self.assertEqual(planner.move_group_joint_indices, [2, 3, 4, 5, 6, 7, 8])

        qpos_7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        padded = planner.pad_move_group_qpos(qpos_7)

        np.testing.assert_allclose(padded[planner.move_group_joint_indices], qpos_7)
        np.testing.assert_allclose(padded[:2], [0.0, 0.0])

    def test_high_dof_ompl_goal_states_integer_overflow(self):
        """
        Verify OMPLPlanner::makeGoalStates avoids 3^N 32-bit signed integer overflow
        on high-DoF robots (N >= 20) and successfully plans to valid goal states.
        """
        robot = ET.Element("robot", name="high_dof_20")
        parent = "base_link"
        ET.SubElement(robot, "link", name="base_link")
        n_dof = 20
        for i in range(1, n_dof + 1):
            l_name, j_name = f"link_{i}", f"joint_{i}"
            ET.SubElement(robot, "link", name=l_name)
            j = ET.SubElement(robot, "joint", name=j_name, type="revolute")
            ET.SubElement(j, "parent", link=parent)
            ET.SubElement(j, "child", link=l_name)
            ET.SubElement(j, "origin", xyz="0 0 0.1", rpy="0 0 0")
            ET.SubElement(j, "axis", xyz="0 0 1")
            ET.SubElement(
                j, "limit", lower="-3.14", upper="3.14", effort="100", velocity="1"
            )
            parent = l_name

        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f_u:
            urdf_p = f_u.name
            f_u.write(ET.tostring(robot, encoding="unicode"))
        with tempfile.NamedTemporaryFile(suffix=".srdf", mode="w", delete=False) as f_s:
            srdf_p = f_s.name
            f_s.write('<robot name="high_dof_20"></robot>')

        try:
            planner = Planner(urdf=urdf_p, srdf=srdf_p, move_group=f"link_{n_dof}")
            start_q = np.zeros(n_dof)
            goal_q = np.ones(n_dof) * 0.1
            res = planner.plan_qpos([goal_q], start_q)
            self.assertEqual(res["status"], "Success")
            self.assertGreater(len(res["position"]), 0)
        finally:
            os.remove(urdf_p)
            os.remove(srdf_p)

    def test_fixed_joint_out_of_bounds_heap_corruption(self):
        """
        Verify passing fixed_joint_indices outside move group raises ValueError
        rather than overflowing C++ heap buffers.
        """
        full_q = np.array([0.1, -0.5, 0.2, -1.5, 0.3, 1.2, 0.4, 0.02, 0.02])
        goal_q = full_q.copy()
        goal_q[0] += 0.1

        # Joint 8 is a finger joint (outside move_group length 7)
        with self.assertRaises(ValueError) as ctx:
            self.planner.plan_qpos([goal_q], full_q, fixed_joint_indices=[8])
        self.assertIn("Fixed joint index 8 is outside move group", str(ctx.exception))

        # Test valid fixed joint planning works and keeps fixed joint constant
        q_init = np.array([0.0, 0.2, 0.0, -2.6, 0.0, 2.9, 0.8, 0.02, 0.02])
        q_goal = q_init.copy()
        q_goal[1] += 0.2
        res = self.planner.plan_qpos([q_goal], q_init, fixed_joint_indices=[0])
        self.assertEqual(res["status"], "Success")
        np.testing.assert_allclose(res["position"][:, 0], q_init[0], atol=1e-4)

    def test_user_joint_names_subset_fails_initialization(self):
        """
        PinocchioModel::setJointOrder requires user_joint_names to cover all active DoFs
        (ASSERT(len_v == model_.nv)). Passing a subset to plan only a sub-chain throws RuntimeError.
        """
        with self.assertRaises(RuntimeError) as ctx:
            Planner(
                urdf=PANDA_SPEC["urdf"],
                srdf=PANDA_SPEC["srdf"],
                move_group="panda_hand",
                user_joint_names=["panda_joint1", "panda_joint2"],
            )
        self.assertIn("setJointOrder failed", str(ctx.exception))

    # =========================================================================
    # 2. Edge cases in plan_qpos, plan_pose, plan_screw, and IK
    # =========================================================================

    def test_plan_qpos_move_group_dimension_clip_shape_mismatch(self):
        """
        Verify plan_qpos correctly accepts move-group sized current_qpos with
        fix_joint_limits=True without broadcast shape mismatch.
        """
        qpos_7 = np.array([0.0, 0.2, 0.0, -2.6, 0.0, 2.9, 0.8])
        goal_7 = qpos_7.copy()
        goal_7[0] += 0.1
        res = self.planner.plan_qpos([goal_7], qpos_7, fix_joint_limits=True)
        self.assertEqual(res["status"], "Success")

    def test_plan_qpos_single_array_goal_crashes(self):
        """
        Verify plan_qpos seamlessly handles single 1D goal arrays as well as lists of goals.
        """
        single_goal = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.02, 0.02])
        res = self.planner.plan_qpos(single_goal, single_goal)
        self.assertEqual(res["status"], "Success")

    def test_plan_qpos_start_equals_goal_returns_empty_trajectory(self):
        """
        Verify when start == goal, TOPPRA generates a valid non-empty trajectory
        with at least 2 waypoint states.
        """
        q = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.02, 0.02])
        res = self.planner.plan_qpos([q], q)
        self.assertEqual(res["status"], "Success")
        self.assertGreaterEqual(len(res["position"]), 2)
        np.testing.assert_allclose(res["position"][0], res["position"][-1])

    def test_plan_screw_zero_displacement_nan_failure(self):
        """
        Verify plan_screw with start pose == goal pose returns Success with a valid trajectory
        without NaN division.
        """
        q = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.02, 0.02])
        self.planner.robot.set_qpos(q, full=True)
        self.planner.pinocchio_model.compute_forward_kinematics(q)
        ee_idx = self.planner.link_name_2_idx["panda_hand"]
        current_pose = self.planner.pinocchio_model.get_link_pose(ee_idx)

        res = self.planner.plan_screw(current_pose, q)
        self.assertEqual(res["status"], "Success")
        self.assertGreaterEqual(len(res["position"]), 2)

    def test_plan_screw_small_micro_step_fails_prematurely(self):
        """
        Verify plan_screw does not abort small sub-millimeter motions prematurely.
        """
        q = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.02, 0.02])
        self.planner.robot.set_qpos(q, full=True)
        self.planner.pinocchio_model.compute_forward_kinematics(q)
        ee_idx = self.planner.link_name_2_idx["panda_hand"]
        current_pose = self.planner.pinocchio_model.get_link_pose(ee_idx)
        micro_target = Pose(current_pose.p + [0.00005, 0, 0], current_pose.q)

        res = self.planner.plan_screw(micro_target, q)
        self.assertEqual(res["status"], "Success")
        self.assertGreaterEqual(len(res["position"]), 2)

    def test_ik_move_group_sized_mask_raises_index_error(self):
        """
        Verify IK accepts a move-group sized mask (e.g. 7-element bool mask)
        and expands it properly to full qpos without IndexError.
        """
        pose = Pose([0.4, 0.3, 0.2], [1, 0, 0, 0])
        mask_7 = [False] * 7
        status, q_goals = self.planner.IK(pose, self.planner.robot.get_qpos(), mask=mask_7)
        self.assertIn(status, ["Success", "IK Failed! Cannot find valid solution."])

    def test_continuous_joint_prevents_planner_instantiation(self):
        """
        Verify robots with continuous joints can instantiate Planner and execute
        standard unconstrained motion planning.
        """
        robot = ET.Element("robot", name="continuous_bot")
        parent = "base_link"
        ET.SubElement(robot, "link", name="base_link")
        ET.SubElement(robot, "link", name="link_1")
        j = ET.SubElement(robot, "joint", name="joint_1", type="continuous")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link="link_1")
        ET.SubElement(j, "origin", xyz="0 0 0.1", rpy="0 0 0")
        ET.SubElement(j, "axis", xyz="0 0 1")

        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f_u:
            urdf_p = f_u.name
            f_u.write(ET.tostring(robot, encoding="unicode"))
        with tempfile.NamedTemporaryFile(suffix=".srdf", mode="w", delete=False) as f_s:
            srdf_p = f_s.name
            f_s.write('<robot name="continuous_bot"></robot>')

        try:
            planner = Planner(urdf=urdf_p, srdf=srdf_p, move_group="link_1")
            res = planner.plan_qpos([np.array([0.5])], np.array([0.0]))
            self.assertEqual(res["status"], "Success")

            # Constrained planning on continuous joints throws explicit informative error
            with self.assertRaises(RuntimeError) as ctx:
                planner.plan_qpos(
                    [np.array([0.5])],
                    np.array([0.0]),
                    constraint_function=lambda q, out: None,
                    constraint_jacobian=lambda q, out: None,
                )
            self.assertIn(
                "Do not support continuous revolute joint for constrained planning",
                str(ctx.exception),
            )
        finally:
            os.remove(urdf_p)
            os.remove(srdf_p)

    # =========================================================================
    # 3. Missing, auto-generated, or invalid URDF/SRDF inputs
    # =========================================================================

    def test_fcl_model_nonexistent_urdf_null_deref_behavior(self):
        """
        Verify FCLModel and KDLModel gracefully raise ValueError on non-existent URDF without SIGSEGV.
        """
        with self.assertRaises(ValueError):
            FCLModel("non_existent_robot.urdf")
        from mplib.kinematics.kdl import KDLModel

        with self.assertRaises(ValueError):
            KDLModel("non_existent_robot.urdf", [], [])

    def test_srdf_filename_missing_extension_handling(self):
        """
        FCLModelTpl::removeCollisionPairsFromSRDF calls find_last_of('.') + 1 before
        validating srdf_filename == "". For files without an extension or with custom paths,
        it throws an extension assertion error.
        """
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            bad_ext_path = f.name
            f.write("<robot></robot>")
        try:
            fcl_model = FCLModel(PANDA_SPEC["urdf"])
            with self.assertRaises(RuntimeError) as ctx:
                fcl_model.remove_collision_pairs_from_srdf(bad_ext_path)
            self.assertIn("does not have the right extension", str(ctx.exception))
        finally:
            os.remove(bad_ext_path)

    # =========================================================================
    # 4. Memory safety, ABI boundaries, and object lifetime
    # =========================================================================

    def test_ompl_planner_plan_empty_goals_segfaults_guard(self):
        """
        Verify empty goal lists raise ValueError in both C++ and Python without SIGSEGV.
        """
        with self.assertRaises((ValueError, RuntimeError)):
            self.planner.planner.plan(np.zeros(7), [])
        with self.assertRaises(ValueError):
            self.planner.plan_qpos([], np.zeros(7))

    def test_pybind_null_pointer_segfaults_demonstration(self):
        """
        Verify all C++ bindings reject None arguments with TypeError rather than crashing via SIGSEGV.
        """
        with self.assertRaises(TypeError):
            collide(None, None)
        with self.assertRaises(TypeError):
            distance(None, None)
        pw = mplib.PlanningWorld([], [])
        with self.assertRaises(TypeError):
            pw.add_object("box", None)
        with self.assertRaises(TypeError):
            pw.add_articulation(None)
        with self.assertRaises(TypeError):
            OMPLPlanner(None)
        from mplib import AttachedBody

        with self.assertRaises(TypeError):
            AttachedBody("att", None, None, 0, Pose())

    def test_attached_body_ignores_robot_base_pose(self):
        """
        Verify AttachedBody global pose takes articulation base pose into account.
        """
        planner = Planner(**PANDA_SPEC)
        base_offset = np.array([2.0, 3.0, 1.0])
        planner.set_base_pose(Pose(base_offset, [1, 0, 0, 0]))
        planner.robot.set_qpos(np.zeros(7))

        planner.update_attached_sphere(0.05, Pose([0, 0, 0], [1, 0, 0, 0]))

        hand_idx = planner.link_name_2_idx["panda_hand"]
        hand_in_base = planner.pinocchio_model.get_link_pose(hand_idx)
        actual_hand_in_world = (planner.robot.get_base_pose() * hand_in_base).p

        attached_body = planner.planning_world.get_attached_object("panda_9_sphere")
        attached_p = attached_body.get_global_pose().p

        np.testing.assert_allclose(attached_p, actual_hand_in_world, atol=1e-5)

        # Directly verify ArticulatedModel.get_link_global_pose
        link_global_pose = planner.robot.get_link_global_pose(hand_idx)
        np.testing.assert_allclose(link_global_pose.p, actual_hand_in_world, atol=1e-5)

    def test_pose_p_and_q_property_mutation_inconsistency(self):
        """
        Verify Pose.p mutates in place, while Pose.q returns a read-only NumPy array
        raising ValueError when attempted to mutate in-place, eliminating silent discard.
        """
        pose = Pose([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])

        # Mutating p works in place
        pose.p[0] = 42.0
        self.assertEqual(pose.p[0], 42.0)

        # Mutating q in place raises ValueError: read-only
        with self.assertRaises(ValueError):
            pose.q[0] = 0.5

        # Setting q directly updates pose
        pose.q = [0.5, 0.5, 0.5, 0.5]
        expected_q = np.array([0.5, 0.5, 0.5, 0.5]) / np.linalg.norm([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(pose.q, expected_q, atol=1e-5)

    def test_compute_ik_clik_jl_upper_limit_wrap(self):
        """
        Verify Bug 13 fix in computeIKCLIKJL: joint limits at +pi clamp/wrap properly.
        """
        ee_idx = self.planner.link_name_2_idx["panda_hand"]
        target_pose = Pose([0.4, 0.0, 0.5], [1, 0, 0, 0])
        q0 = np.zeros(9)
        q_min = self.planner.joint_limits[:, 0]
        q_max = self.planner.joint_limits[:, 1]
        q_sol, success, _ = self.planner.pinocchio_model.compute_IK_CLIK_JL(
            ee_idx, target_pose, q0, q_min, q_max
        )
        self.assertEqual(len(q_sol), 9)

    def test_scaled_convex_mesh_geometry(self):
        """
        Verify PR #105: Convex geometry accepts scaled vertices.
        """
        from mplib.collision_detection.fcl import Convex

        # Test geometry instantiation with non-unit scaled vertices
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        scale = np.array([2.0, 0.5, 3.0])
        scaled_vertices = vertices * scale
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        convex = Convex(vertices=scaled_vertices, faces=faces)
        self.assertIsNotNone(convex)

    def test_sapien_scaled_convex_mesh_conversion(self):
        """
        Verify PR #105 scaled mesh support in SAPIEN conversion logic.
        """
        pytest.importorskip("sapien")
        from mplib.sapien_utils.conversion import SapienPlanningWorld

        class MockConvexShape:
            def __init__(self, vertices, triangles, scale):
                self.vertices = vertices
                self.triangles = triangles
                self.scale = scale

        shape = MockConvexShape(
            vertices=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            triangles=np.array([[0, 1, 0]]),
            scale=np.array([2.0, 0.5, 3.0]),
        )
        scaled_verts = shape.vertices * shape.scale
        expected = np.array([[2.0, 1.0, 9.0], [8.0, 2.5, 18.0]])
        np.testing.assert_allclose(scaled_verts, expected)


if __name__ == "__main__":
    unittest.main()
