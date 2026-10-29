# MPlib: a Lightweight Motion Planning Library

<p align="center">
  <img src="https://raw.githubusercontent.com/haosulab/MPlib/main/docs/demo.gif">
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/mplib)](https://pypi.org/project/mplib/)
[![Downloads](https://static.pepy.tech/badge/mplib)](https://pepy.tech/project/mplib)
[![Build python wheels](https://img.shields.io/github/actions/workflow/status/haosulab/MPlib/build_and_publish.yml)](https://github.com/haosulab/MPlib/releases/tag/nightly)
[![Documentation](https://img.shields.io/readthedocs/motion-planning-lib)](https://motion-planning-lib.readthedocs.io/)
[![License](https://img.shields.io/github/license/haosulab/MPlib)](https://github.com/haosulab/MPlib?tab=MIT-1-ov-file#readme)

MPlib is a lightweight python package for motion planning,
which is decoupled from ROS and is easy to set up.  
With a few lines of python code, one can achieve most of the motion planning
functionalities in robot manipulation.

## Installation

### Standard Pip Installation

Pre-built binary wheels support Linux x86_64 environments:

```bash
pip install mplib
```

### Building from Source & Editable Installation

#### Prerequisites

Building MPlib from source requires a C++17 compiler (`g++` or `clang++`), CMake (>= 3.18), Ninja, and C++ dependency libraries:
- Eigen3 (>= 3.3)
- Pinocchio (>= 3.0)
- FCL / Coal
- OMPL (>= 1.5)
- Boost (>= 1.70: filesystem, serialization, system)
- Orocos-KDL
- URDFDOM
- Assimp

#### Option A: Managed Environment with Pixi (Recommended)

[Pixi](https://pixi.sh/) automatically installs required C++ libraries and Python toolchains (Python >= 3.10, < 3.13):

```bash
# Clone the repository
git clone https://github.com/haosulab/MPlib.git
cd MPlib

# Install development environment and execute test suite
pixi run pytest tests
```

#### Option B: Editable Installation via Pip (Python 3.10–3.12)

Within an existing Conda or system environment providing the prerequisite C++ libraries:

```bash
# Install runtime Python dependencies
pip install numpy toppra transforms3d

# Build C++ pybind11 extension in-place and install editable package
pip install -e . --no-build-isolation
```

## Quickstart

```python
import numpy as np
import mplib

# Initialize planner for Franka Emika Panda arm
planner = mplib.Planner(
    urdf="data/panda/panda.urdf",
    srdf="data/panda/panda.srdf",
    move_group="panda_hand",
)

# Set initial joint configuration (7 arm joints + 2 gripper fingers)
start_qpos = np.array([0.0, 0.2, 0.0, -2.6, 0.0, 2.9, 0.8, 0.02, 0.02])
planner.robot.set_qpos(start_qpos, full=True)

# Forward Kinematics: compute end-effector pose
planner.pinocchio_model.compute_forward_kinematics(start_qpos)
ee_idx = planner.link_name_2_idx["panda_hand"]
target_pose = planner.pinocchio_model.get_link_pose(ee_idx)

# Inverse Kinematics: solve for joint angles matching target pose
status, ik_qpos = planner.IK(target_pose, start_qpos, return_closest=True)
if status == "Success":
    print("IK Solution found:", ik_qpos[:7])

# Motion Planning: plan collision-free trajectory to target pose
target_pose.p[0] += 0.1  # Translate 10 cm in X
result = planner.plan_pose(target_pose, start_qpos)

if result["status"] == "Success":
    print(f"Planned trajectory duration: {result['duration']:.2f} s")
    print(f"Waypoints: {len(result['position'])}")
```

## Documentation

See the [tutorial and API reference](https://motion-planning-lib.readthedocs.io/latest/tutorials/getting_started.html) for detailed usage, scene collision objects, attached tools, and constrained motion planning.
