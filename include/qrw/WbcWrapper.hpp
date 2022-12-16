///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for WbcWrapper classes
///
/// \details WbcWrapper provides an interface for the user to solve the whole
/// body control problem
///          Internally it calls first the InvKin class to solve an inverse
///          kinematics problem then calls the QPWBC class to solve a box QP
///          problem based on result from the inverse kinematic and desired
///          ground forces
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef WBC_WRAPPER_H_INCLUDED
#define WBC_WRAPPER_H_INCLUDED

#include "qrw/InvKin.hpp"
#include "qrw/Params.hpp"
#include "qrw/QPWBC.hpp"

class WbcWrapper {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  WbcWrapper();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~WbcWrapper() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params &params);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run and solve one iteration of the whole body control (matrix
  /// update, invkin, QP)
  ///
  /// \param[in] q Estimated positions of the 12 actuators
  /// \param[in] dq Estimated velocities of the 12 actuators
  /// \param[in] f_cmd Reference contact forces received from the MPC
  /// \param[in] contacts Contact status of the four feet
  /// \param[in] pgoals Desired positions of the four feet in base frame
  /// \param[in] vgoals Desired velocities of the four feet in base frame
  /// \param[in] agoals Desired accelerations of the four feet in base frame
  /// \param[in] xgoals Desired position, orientation and velocities of the base
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void compute(VectorN const &q, VectorN const &dq, VectorN const &f_cmd,
               MatrixN const &contacts, MatrixN const &pgoals,
               MatrixN const &vgoals, MatrixN const &agoals,
               VectorN const &xgoals);

  VectorN get_bdes() { return bdes_; }
  VectorN get_qdes() { return qdes_; }
  VectorN get_vdes() { return vdes_; }
  VectorN get_tau_ff() { return tau_ff_; }
  VectorN get_ddq_cmd() { return ddq_cmd_; }
  VectorN get_dq_cmd() { return dq_cmd_; }
  VectorN get_q_cmd() { return q_cmd_; }
  VectorN get_f_with_delta() { return f_with_delta_; }
  VectorN get_ddq_with_delta() { return ddq_with_delta_; }
  VectorN get_nle() { return nle_; }
  MatrixN get_feet_pos() { return invkin_->get_posf().transpose(); }
  MatrixN get_feet_err() {
    return log_feet_pos_target - invkin_->get_posf().transpose();
  }
  MatrixN get_feet_vel() { return invkin_->get_vf().transpose(); }
  VectorN get_tasks_acc() { return invkin_->get_tasks_acc(); }
  VectorN get_tasks_vel() { return invkin_->get_tasks_vel(); }
  VectorN get_tasks_err() { return invkin_->get_tasks_err(); }
  MatrixN get_feet_pos_target() { return log_feet_pos_target; }
  MatrixN get_feet_vel_target() { return log_feet_vel_target; }
  MatrixN get_feet_acc_target() { return log_feet_acc_target; }

  VectorN get_Mddq() { return Mddq; };
  VectorN get_NLE() { return NLE; };
  VectorN get_JcTf() { return JcTf; };
  VectorN get_Mddq_out() { return Mddq_out; };
  VectorN get_JcTf_out() { return JcTf_out; };

 private:
  Params *params_;  // Object that stores parameters
  QPWBC *box_qp_;   // QP problem solver for the whole body control
  InvKin *invkin_;  // Inverse Kinematics solver for the whole body control

  pinocchio::Model model_;  // Pinocchio model for frame computations
  pinocchio::Data data_;    // Pinocchio datas for frame computations

  Eigen::Matrix<double, 18, 18> M_;  // Mass matrix
  Eigen::Matrix<double, 12, 6> Jc_;  // Jacobian matrix
  RowVector4 k_since_contact_;  // Number of time step during which feet have
                                // been in the current stance phase
  Vector7 bdes_;                // Desired base positions
  Vector12 qdes_;               // Desired actuator positions
  Vector12 vdes_;               // Desired actuator velocities
  Vector12 tau_ff_;             // Desired actuator torques (feedforward)

  Vector19 q_wbc_;    // Configuration vector for the whole body control
  Vector18 dq_wbc_;   // Velocity vector for the whole body control
  Vector18 ddq_cmd_;  // Command ccelerations computed by Inverse Kinematics
  Vector18 dq_cmd_;   // Command velocities computed by Inverse Kinematics
  Vector19 q_cmd_;    // Command positions computed by Inverse Kinematics
  Vector12 f_with_delta_;  // Contact forces with deltas found by QP solver
  Vector18
      ddq_with_delta_;  // Actuator accelerations with deltas found by QP solver

  Vector6 nle_;  // Non linear effects

  Matrix34 log_feet_pos_target;  // Store the target feet positions
  Matrix34 log_feet_vel_target;  // Store the target feet velocities
  Matrix34 log_feet_acc_target;  // Store the target feet accelerations

  // Log
  Vector6 Mddq;      // Generalized mass matrix * acceleration vector
  Vector6 NLE;       // Non linear effects
  Vector6 JcTf;      // Contact Jacobian * contact forces vector
  Vector6 Mddq_out;  // Generalized mass matrix * acceleration vector (after QP
                     // problem)
  Vector6
      JcTf_out;  // Contact Jacobian * contact forces vector (after QP problem)

  int k_log_;  // Counter for logging purpose
  int indexes_[4] = {
      10, 18, 26,
      34};  // Indexes of feet frames in this order: [FL, FR, HL, HR]
  bool enable_comp_forces_;  // Enable compensation forces for the QP problem
};

#endif  // WBC_WRAPPER_H_INCLUDED
