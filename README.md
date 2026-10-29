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

Pre-built pip packages support Ubuntu 20.04+ with Python 3.8+.

```bash
pip install mplib
```

### Developer Setup & Building from Source

MPlib supports reproducible development workflows using [Pixi](https://pixi.sh/):

```bash
# Clone the repository
git clone https://github.com/haosulab/MPlib.git && cd MPlib

# Install development environment and run tests
pixi run pytest tests
```

## Usage

See our [tutorial](https://motion-planning-lib.readthedocs.io/latest/tutorials/getting_started.html) for detailed usage and examples.

### Key Capabilities & Robustness Guarantees

- **Decoupled Kinematics & Planning**: Powered by Pinocchio 3.x, FCL/Coal collision detection, and OMPL.
- **Robust Coordinate Frames**: Attached bodies track robot link positions in the world frame regardless of articulation base displacement.
- **High-DoF & Continuous Joints**: Supports high-DoF robotic chains (>= 20 DoF) without goal state enumeration overflows, and unconstrained planning on continuous revolute joints.
- **Safety Guards**: Strict null and type guards across C++ and Pybind11 boundaries, preventing silent memory corruption or segmentation faults.
