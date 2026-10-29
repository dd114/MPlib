#include "mplib/planning/ompl/fixed_joint.h"

namespace mplib::planning::ompl {

template <typename S>
bool isFixedJoint(const FixedJointsTpl<S> &fixed_joints, size_t articulation_idx,
                  size_t joint_idx) {
  for (const auto &fixed_joint : fixed_joints)
    if (fixed_joint.articulation_idx == articulation_idx &&
        fixed_joint.joint_idx == joint_idx)
      return true;
  return false;
}

template <typename S>
VectorX<S> addFixedJoints(const FixedJointsTpl<S> &fixed_joints,
                          const VectorX<S> &state) {
  size_t total_rows = fixed_joints.size() + state.rows();
  for (const auto &fixed_joint : fixed_joints) {
    if (fixed_joint.joint_idx >= total_rows)
      throw std::invalid_argument(
          "Fixed joint index " + std::to_string(fixed_joint.joint_idx) +
          " is out of bounds for full state of size " + std::to_string(total_rows));
  }
  auto fixed_itr = fixed_joints.begin();
  VectorX<S> ret(total_rows);
  size_t j = 0;
  for (auto i = 0; i < ret.rows(); i++) {
    if (isFixedJoint<S>(fixed_joints, 0, i)) {
      ret[i] = fixed_itr->value;
      ++fixed_itr;
    } else {
      ret[i] = state[j++];
    }
  }
  return ret;
}

template <typename S>
VectorX<S> removeFixedJoints(const FixedJointsTpl<S> &fixed_joints,
                             const VectorX<S> &state) {
  for (const auto &fixed_joint : fixed_joints) {
    if (fixed_joint.joint_idx >= static_cast<size_t>(state.rows()))
      throw std::invalid_argument(
          "Fixed joint index " + std::to_string(fixed_joint.joint_idx) +
          " is out of bounds for state of size " + std::to_string(state.rows()));
  }
  if (fixed_joints.size() > static_cast<size_t>(state.rows()))
    throw std::invalid_argument("Cannot fix more joints than the state dimension");

  VectorX<S> ret(state.rows() - fixed_joints.size());
  size_t cnt = 0;
  for (auto i = 0; i < state.rows(); i++) {
    if (isFixedJoint<S>(fixed_joints, 0, i))
      continue;  // TODO[xinsong] only support one robot rn
    ret[cnt++] = state[i];
  }
  return ret;
}

// Explicit Template Instantiation Definition ==========================================
#define DEFINE_TEMPLATE_FIXED_JOINT(S)                                            \
  template bool isFixedJoint<S>(const FixedJointsTpl<S> &fixed_joints,            \
                                size_t articulation_idx, size_t joint_idx);       \
  template VectorX<S> addFixedJoints<S>(const FixedJointsTpl<S> &fixed_joints,    \
                                        const VectorX<S> &state);                 \
  template VectorX<S> removeFixedJoints<S>(const FixedJointsTpl<S> &fixed_joints, \
                                           const VectorX<S> &state)

DEFINE_TEMPLATE_FIXED_JOINT(float);
DEFINE_TEMPLATE_FIXED_JOINT(double);

}  // namespace mplib::planning::ompl
