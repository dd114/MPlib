import os
import unittest
import numpy as np

import mplib
from mplib import Planner, Pose
from mplib.collision_detection.fcl import Box, CollisionObject, FCLObject

FILE_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
PANDA_SPEC = {
    "urdf": f"{FILE_ABS_DIR}/../data/panda/panda.urdf",
    "srdf": f"{FILE_ABS_DIR}/../data/panda/panda.srdf",
    "move_group": "panda_hand",
}


class TestAdvancedValidation(unittest.TestCase):
    """
    Stress-testing and numerical validation suite covering:
    - High-density SE(3) IK workspace convergence and joint limit enforcement.
    - Attached body collision detection across non-zero articulation base frames.
    - TOPPRA trajectory continuity, velocity/acceleration constraints, and NaN guards.
    - Sub-chain kinematics and planning on non-contiguous joint index mappings.
    """

    def setUp(self):
        self.planner = Planner(**PANDA_SPEC)
        self.robot = self.planner.robot
        self.pinocchio_model = self.planner.pinocchio_model
        self.joint_limits = self.planner.joint_limits
        self.ee_idx = self.planner.link_name_2_idx["panda_hand"]
        self.move_group_indices = self.planner.move_group_joint_indices

    # =========================================================================
    # 1. 100+ SE(3) Workspace IK Convergence and Joint Limit Verification
    # =========================================================================

    def test_se3_ik_workspace_100_samples_convergence_and_limits(self):
        """
        Sample 100+ diverse reachable SE(3) poses across the operational workspace.
        Verify:
        - IK solutions satisfy pose distance error < 1e-3.
        - All joint angles strictly obey URDF physical joint limits within tolerance.
        - High convergence rate (> 90%) across reachable configurations.
        """
        np.random.seed(42)
        num_samples = 120
        solved_count = 0

        lower_limits = self.joint_limits[:, 0]
        upper_limits = self.joint_limits[:, 1]

        # Use 7-DoF move group range with safe margins
        mg_lower = lower_limits[self.move_group_indices] + 0.05
        mg_upper = upper_limits[self.move_group_indices] - 0.05

        for i in range(num_samples):
            # Generate valid ground truth configuration
            q_gt_mg = np.random.uniform(mg_lower, mg_upper)
            q_gt_full = self.planner.pad_move_group_qpos(q_gt_mg)

            # Compute ground truth end-effector pose
            self.pinocchio_model.compute_forward_kinematics(q_gt_full)
            target_pose = self.pinocchio_model.get_link_pose(self.ee_idx)

            # Initial seed perturbed from ground truth or standard home configuration
            q_seed = q_gt_full.copy()
            q_seed[self.move_group_indices] += np.random.uniform(
                -0.3, 0.3, size=len(self.move_group_indices)
            )
            q_seed = np.clip(q_seed, lower_limits, upper_limits)

            status, q_sol = self.planner.IK(
                target_pose,
                q_seed,
                n_init_qpos=30,
                threshold=1e-3,
                return_closest=True,
            )

            if status == "Success" and q_sol is not None:
                solved_count += 1

                # Verify joint limits are strictly respected
                self.assertTrue(
                    np.all(q_sol >= lower_limits - 1e-3),
                    f"Sample {i}: Joint values fall below lower limits: {q_sol}",
                )
                self.assertTrue(
                    np.all(q_sol <= upper_limits + 1e-3),
                    f"Sample {i}: Joint values exceed upper limits: {q_sol}",
                )

                # Verify forward kinematics matches target pose within tolerance
                self.pinocchio_model.compute_forward_kinematics(q_sol)
                sol_pose = self.pinocchio_model.get_link_pose(self.ee_idx)
                pose_err = target_pose.distance(sol_pose)
                self.assertLess(
                    pose_err,
                    1e-3,
                    f"Sample {i}: Pose distance error {pose_err} exceeds 1e-3",
                )

        # Assert convergence rate across sampled reachable poses is at least 90%
        convergence_rate = solved_count / num_samples
        self.assertGreaterEqual(
            convergence_rate,
            0.90,
            f"IK convergence rate {convergence_rate:.2%} is below 90% threshold",
        )

    # =========================================================================
    # 2. Attached Body Collision Under Non-Zero Robot Base Poses
    # =========================================================================

    def test_attached_body_collision_with_nonzero_base_pose(self):
        """
        Verify attached bodies correctly transform with the articulation base pose:
        - When base pose is non-identity, attached body world pose reflects base displacement.
        - Collision checks with world obstacles accurately evaluate in world frame.
        """
        world = self.planner.planning_world

        # Displace robot base in position and orientation
        base_pos = np.array([2.5, -1.2, 0.8])
        base_quat = np.array([0.7071068, 0.0, 0.0, 0.7071068])  # 90 deg around Z
        base_pose = Pose(base_pos, base_quat)
        self.planner.set_base_pose(base_pose)

        # Set arm configuration
        q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02])
        self.robot.set_qpos(q, full=True)

        # Attach tool sphere to end effector
        sphere_radius = 0.08
        rel_attach_pose = Pose([0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0])
        self.planner.update_attached_sphere(
            radius=sphere_radius,
            pose=rel_attach_pose,
            link_id=self.ee_idx,
        )

        attached_obj = world.get_attached_object("panda_9_sphere")
        attached_world_pose = attached_obj.get_global_pose()

        # Expected world pose = T_world_base * T_base_ee * T_ee_tool
        self.pinocchio_model.compute_forward_kinematics(q)
        ee_in_base = self.pinocchio_model.get_link_pose(self.ee_idx)
        expected_world_pose = base_pose * ee_in_base * rel_attach_pose

        np.testing.assert_allclose(
            attached_world_pose.p, expected_world_pose.p, atol=1e-4
        )
        np.testing.assert_allclose(
            attached_world_pose.to_transformation_matrix()[:3, :3],
            expected_world_pose.to_transformation_matrix()[:3, :3],
            atol=1e-4,
        )

        # Place box obstacle directly intersecting the attached sphere in world frame
        box_name = "collision_target_box"
        box_geom = Box([0.15, 0.15, 0.15])
        box_obj = CollisionObject(box_geom, attached_world_pose.to_transformation_matrix())
        world.add_object(box_name, box_obj)

        # Verify collision is detected with the attached body
        collisions = world.check_collision()
        self.assertGreater(len(collisions), 0)
        col_types = [c.collision_type for c in collisions]
        self.assertTrue(
            any("attached" in ct or "object" in ct for ct in col_types),
            f"Expected attached collision, got: {col_types}",
        )

        # Move obstacle away and verify collision clears
        world.remove_object(box_name)
        far_pose = Pose(attached_world_pose.p + [5.0, 5.0, 5.0], [1, 0, 0, 0])
        box_obj_far = CollisionObject(box_geom, far_pose.to_transformation_matrix())
        world.add_object(box_name, box_obj_far)
        self.assertEqual(len(world.check_collision()), 0)

        world.remove_object(box_name)
        world.detach_object("panda_9_sphere")

    # =========================================================================
    # 3. TOPPRA Trajectory Continuity, Derivative Bounds, and Output Safety
    # =========================================================================

    def test_toppra_trajectory_continuity_and_derivative_bounds(self):
        """
        Verify TOPPRA time parameterization guarantees:
        - Output arrays contain no NaNs, Infs, or empty trajectories.
        - Waypoint step continuity: |q_{k+1} - q_k| <= dq_max * dt + eps.
        - Velocities and accelerations strictly adhere to physical limits.
        - Single-step and start == goal inputs execute without degradation.
        """
        vel_limits = self.planner.joint_vel_limits
        acc_limits = self.planner.joint_acc_limits
        dt = 0.05

        # Path 1: Multi-waypoint path across configuration space
        q_start = np.array([0.0, 0.2, 0.0, -2.5, 0.0, 2.7, 0.8])
        q_mid = np.array([0.4, -0.3, 0.3, -1.8, 0.2, 2.0, 0.5])
        q_goal = np.array([-0.3, 0.4, -0.2, -2.1, -0.3, 2.4, 1.1])
        multi_path = np.vstack([q_start, q_mid, q_goal])

        ts, qs, qds, qdds, duration = self.planner.TOPP(multi_path, dt)

        # Verify finite values and valid shape
        self.assertGreaterEqual(len(ts), 2)
        self.assertGreater(duration, 0.0)
        self.assertTrue(np.all(np.isfinite(ts)))
        self.assertTrue(np.all(np.isfinite(qs)))
        self.assertTrue(np.all(np.isfinite(qds)))
        self.assertTrue(np.all(np.isfinite(qdds)))

        # Verify velocity and acceleration limits (with numerical discretization margin)
        eps = 1e-3
        for k in range(len(ts)):
            self.assertTrue(
                np.all(np.abs(qds[k]) <= vel_limits + eps),
                f"Step {k}: Joint velocities exceed limits: {np.abs(qds[k])} > {vel_limits}",
            )
            self.assertTrue(
                np.all(np.abs(qdds[k]) <= acc_limits + eps),
                f"Step {k}: Joint accelerations exceed limits: {np.abs(qdds[k])} > {acc_limits}",
            )

        # Verify trajectory continuity: |q_{k+1} - q_k| <= dq_max * delta_t + eps
        for k in range(len(ts) - 1):
            delta_t = ts[k + 1] - ts[k]
            delta_q = np.abs(qs[k + 1] - qs[k])
            max_allowed_delta = vel_limits * delta_t + 1e-4
            self.assertTrue(
                np.all(delta_q <= max_allowed_delta),
                f"Discontinuity at step {k}: |dq|={delta_q} > max_allowed={max_allowed_delta}",
            )

        # Path 2: Start == Goal stationary path
        static_path = np.vstack([q_start, q_start])
        ts_s, qs_s, qds_s, qdds_s, duration_s = self.planner.TOPP(static_path, dt)

        self.assertGreaterEqual(len(ts_s), 2)
        self.assertTrue(np.all(np.isfinite(qs_s)))
        np.testing.assert_allclose(qs_s[0], qs_s[-1], atol=1e-5)
        np.testing.assert_allclose(qds_s, 0.0, atol=1e-4)

    # =========================================================================
    # 4. Move-Group Sub-Chain Mapping with Non-Contiguous Joint Indices
    # =========================================================================

    def test_move_group_subchain_non_contiguous_joint_indices(self):
        """
        Configure an articulated robot model where move group joints are non-contiguous:
        user_joint_names places gripper fingers at non-contiguous slots [1, 4],
        leaving move group joint indices as [0, 2, 3, 5, 6, 7, 8].
        Verify:
        - pad_move_group_qpos properly maps values to non-contiguous indices.
        - Motion planning respects the sub-chain mapping and actuates only move group joints.
        """
        interleaved_joint_names = [
            "panda_joint1",         # index 0 (move group)
            "panda_finger_joint1",  # index 1 (unplanned finger)
            "panda_joint2",         # index 2 (move group)
            "panda_joint3",         # index 3 (move group)
            "panda_finger_joint2",  # index 4 (unplanned finger)
            "panda_joint4",         # index 5 (move group)
            "panda_joint5",         # index 6 (move group)
            "panda_joint6",         # index 7 (move group)
            "panda_joint7",         # index 8 (move group)
        ]
        sub_planner = Planner(
            urdf=PANDA_SPEC["urdf"],
            srdf=PANDA_SPEC["srdf"],
            move_group="panda_hand",
            user_joint_names=interleaved_joint_names,
        )

        expected_mg_indices = [0, 2, 3, 5, 6, 7, 8]
        self.assertEqual(sub_planner.move_group_joint_indices, expected_mg_indices)

        # 1. Test pad_move_group_qpos with non-contiguous indices
        q_mg = np.array([0.1, -0.4, 0.2, -1.9, 0.1, 1.7, 0.4])
        padded = sub_planner.pad_move_group_qpos(q_mg)

        self.assertEqual(len(padded), 9)
        np.testing.assert_allclose(padded[expected_mg_indices], q_mg)
        # Verify passive joints [1, 4] remain untouched
        self.assertEqual(padded[1], 0.0)
        self.assertEqual(padded[4], 0.0)

        # 2. Test planning on non-contiguous move group
        start_mg = np.array([0.0, 0.2, 0.0, -2.6, 0.0, 2.9, 0.8])
        goal_mg = start_mg.copy()
        goal_mg[0] += 0.2
        goal_mg[2] -= 0.1

        res = sub_planner.plan_qpos([goal_mg], start_mg)
        self.assertEqual(res["status"], "Success")
        self.assertGreater(len(res["position"]), 0)

        # Verify planned trajectory points have 7-DoF move group size
        self.assertEqual(res["position"].shape[1], 7)

        # Verify padding trajectory points preserves passive finger values
        full_trajectory = [sub_planner.pad_move_group_qpos(pt) for pt in res["position"]]
        for idx, full_pt in enumerate(full_trajectory):
            self.assertEqual(full_pt[1], 0.0)
            self.assertEqual(full_pt[4], 0.0)
            np.testing.assert_allclose(
                full_pt[expected_mg_indices],
                res["position"][idx],
            )


if __name__ == "__main__":
    unittest.main()
