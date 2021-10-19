///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for InvKin class
///
/// \details Perform inverse kinematics to output command positions, velocities and accelerations for the actuators
///          based on contact status and desired position, velocity and acceleration of the feet
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef INVKIN_H_INCLUDED
#define INVKIN_H_INCLUDED

#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "qrw/Params.hpp"
#include "qrw/Types.h"

class InvKin {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  InvKin();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] params Object that stores parameters
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params& params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~InvKin() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Solve the inverse kinematics problem by inverting the joint Jacobian
  ///
  /// \param[in] contacts Contact status of the four feet
  /// \param[in] pgoals Desired positions of the four feet in base frame
  /// \param[in] vgoals Desired velocities of the four feet in base frame
  /// \param[in] agoals Desired accelerations of the four feet in base frame
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void refreshAndCompute(Matrix14 const& contacts, Matrix43 const& pgoals, Matrix43 const& vgoals,
                         Matrix43 const& agoals);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Solve the inverse kinematics problem by inverting the feet Jacobian
  ///
  /// \param[in] q Estimated positions of the 12 actuators
  /// \param[in] dq Estimated velocities of the 12 actuators
  /// \param[in] contacts Contact status of the four feet
  /// \param[in] pgoals Desired positions of the four feet in base frame
  /// \param[in] vgoals Desired velocities of the four feet in base frame
  /// \param[in] agoals Desired accelerations of the four feet in base frame
  /// \param[in] x_cmd Desired position, orientation and velocity of the base
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void run_InvKin(VectorN const& q, VectorN const& dq, MatrixN const& contacts, MatrixN const& pgoals,
                  MatrixN const& vgoals, MatrixN const& agoals, MatrixN const& x_cmd);

  VectorN get_q_step() { return q_step_; }
  VectorN get_dq_cmd() { return dq_cmd_; }
  VectorN get_q_cmd() { return q_cmd_; }
  VectorN get_ddq_cmd() { return ddq_cmd_; }
  Eigen::Matrix<double, 12, 18> get_Jf() { return Jf_; }
  int get_foot_id(int i) { return foot_ids_[i]; }
  Matrix43 get_posf() { return posf_; }
  Matrix43 get_vf() { return vf_; }
  VectorN get_tasks_acc() { return acc; }
  VectorN get_tasks_vel() { return dx_r; }
  VectorN get_tasks_err() { return x_err; }

 private:
  Params* params_;  // Params object to store parameters

  // Matrices initialisation
  Matrix12 invJ;                       // Inverse of the feet Jacobian
  Eigen::Matrix<double, 1, 30> acc;    // Reshaped feet acceleration references to get command acc for actuators
  Eigen::Matrix<double, 1, 30> x_err;  // Reshaped feet position errors to get command position step for actuators
  Eigen::Matrix<double, 1, 30> dx_r;   // Reshaped feet velocity references to get command velocities for actuators

  Matrix43 pfeet_err;  // Feet position errors to get command position step for actuators
  Matrix43 vfeet_ref;  // Feet velocity references to get command velocities for actuators
  Matrix43 afeet;      // Feet acceleration references to get command accelerations for actuators

  int foot_ids_[4] = {0, 0, 0, 0};          // Feet frame IDs
  int foot_joints_ids_[4] = {3, 6, 9, 12};  // Feet joints IDs
  int base_id_ = 0;                         // Base ID

  Matrix43 posf_;                        // Current feet positions
  Matrix43 vf_;                          // Current feet linear velocities
  Matrix43 wf_;                          // Current feet angular velocities
  Matrix43 af_;                          // Current feet linear accelerations
  Matrix43 dJdq_;                        // Acceleration "drift"
  Eigen::Matrix<double, 12, 18> Jf_;     // Current feet Jacobian (only linear part)
  Eigen::Matrix<double, 6, 18> Jf_tmp_;  // Temporary storage variable to only retrieve the linear part of the Jacobian
                                         // and discard the angular part

  Vector3 posb_;                     // Position of the base
  Vector3 posb_ref_;                 // Reference position of the base
  Vector3 posb_err_;                 // Error in base position
  Matrix3 rotb_;                     // Orientation of the base
  Matrix3 rotb_ref_;                 // Reference orientation of the base
  Vector3 rotb_err_;                 // Error in base orientation
  Vector3 vb_;                       // Linear velocity of the base
  Vector3 vb_ref_;                   // Reference linear velocity of the base
  Vector3 wb_;                       // Angular velocity of the base
  Vector3 wb_ref_;                   // Reference angular velocity of the base
  Vector6 ab_;                       // Acceleration of the base
  Vector3 abasis;                    // Acceleration references for the base linear velocity task
  Vector3 awbasis;                   // Acceleration references for the base angular velocity task
  Matrix43 afeet_contacts_;          // Acceleration references for the feet contact task
  Eigen::Matrix<double, 6, 18> Jb_;  // Jacobian of the base (linear/angular)

  Eigen::Matrix<double, 30, 18> J_;     // Task Jacobian
  Eigen::Matrix<double, 18, 30> invJ_;  // Inverse of Task Jacobian

  Vector3 Kp_base_position;     // Proportional gains for base position task
  Vector3 Kd_base_position;     // Derivative gains for base position task
  Vector3 Kp_base_orientation;  // Proportional gains for base orientation task
  Vector3 Kd_base_orientation;  // Derivative gains for base orientation task
  Vector8 w_tasks;              // Weight vector for tasks weighting

  Vector18 ddq_cmd_;  // Actuator command accelerations
  Vector18 dq_cmd_;   // Actuator command velocities
  Vector19 q_cmd_;    // Actuator command positions
  Vector18 q_step_;   // Actuator command position steps

  pinocchio::Model model_;  // Pinocchio model for frame computations and inverse kinematics
  pinocchio::Data data_;    // Pinocchio datas for frame computations and inverse kinematics

  pinocchio::Model model_dJdq_;  // Pinocchio model for frame computations and inverse kinematics
  pinocchio::Data data_dJdq_;    // Pinocchio datas for frame computations and inverse kinematics
};

////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief Compute the pseudo inverse of a matrix using the Jacobi SVD formula
///
////////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Matrix_Type_>
Eigen::MatrixXd pseudoInverse(const _Matrix_Type_& a, double epsilon = std::numeric_limits<double>::epsilon()) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance =
      epsilon * static_cast<double>(std::max(a.cols(), a.rows())) * svd.singularValues().array().abs()(0);
  return svd.matrixV() *
         (svd.singularValues().array().abs() > tolerance)
             .select(svd.singularValues().array().inverse(), 0)
             .matrix()
             .asDiagonal() *
         svd.matrixU().adjoint();
}
#endif  // INVKIN_H_INCLUDED
