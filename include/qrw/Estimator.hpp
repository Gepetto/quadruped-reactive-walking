///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Estimator and ComplementaryFilter classes
///
/// \details These classes estimate the state of the robot based on sensor measurements
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ESTIMATOR_H_INCLUDED
#define ESTIMATOR_H_INCLUDED

#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"
#include "qrw/Types.h"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include <deque>

class ComplementaryFilter {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ComplementaryFilter();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~ComplementaryFilter() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize
  ///
  /// \param[in] dt Time step of the complementary filter
  /// \param[in] HP_x Initial value for the high pass filter
  /// \param[in] LP_x Initial value for the low pass filter
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(double dt, Vector3 HP_x, Vector3 LP_x);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the filtered output of the complementary filter
  ///
  /// \param[in] x Quantity handled by the filter
  /// \param[in] dx Derivative of the quantity
  /// \param[in] alpha Filtering coefficient between x and dx quantities
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector3 compute(Vector3 const& x, Vector3 const& dx, Vector3 const& alpha);

  Vector3 getX() { return x_; }           // Get the input quantity
  Vector3 getDX() { return dx_; }         // Get the derivative of the input quantity
  Vector3 getHpX() { return HP_x_; }      // Get the high-passed internal quantity
  Vector3 getLpX() { return LP_x_; }      // Get the low-passed internal quantity
  Vector3 getAlpha() { return alpha_; }   // Get the alpha coefficient of the filter
  Vector3 getFiltX() { return filt_x_; }  // Get the filtered output

 private:
  double dt_;
  Vector3 x_, dx_, HP_x_, LP_x_, alpha_, filt_x_;
};

class Estimator {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Estimator();

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
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Estimator() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Retrieve and update IMU data
  ///
  /// \param[in] baseLinearAcceleration Linear acceleration of the IMU (gravity compensated)
  /// \param[in] baseAngularVelocity Angular velocity of the IMU
  /// \param[in] baseOrientation Euler orientation of the IMU
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void get_data_IMU(Vector3 const& baseLinearAcceleration, Vector3 const& baseAngularVelocity,
                    Vector3 const& baseOrientation);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Retrieve and update position and velocity of the 12 actuators
  ///
  /// \param[in] q_mes Position of the 12 actuators
  /// \param[in] v_mes Velocity of the 12 actuators
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void get_data_joints(Vector12 const& q_mes, Vector12 const& v_mes);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimate base position and velocity using Forward Kinematics
  ///
  /// \param[in] feet_status Contact status of the four feet
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void get_data_FK(Eigen::Matrix<double, 1, 4> const& feet_status);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute barycenter of feet in contact
  ///
  /// \param[in] feet_status Contact status of the four feet
  /// \param[in] goals Target positions of the four feet
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void get_xyz_feet(Eigen::Matrix<double, 1, 4> const& feet_status, Matrix34 const& goals);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimate the velocity of the base with forward kinematics using a contact point that
  ///        is supposed immobile in world frame
  ///
  /// \param[in] contactFrameId Frame ID of the contact point
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector3 BaseVelocityFromKinAndIMU(int contactFrameId);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run one iteration of the estimator to get the position and velocity states of the robot
  ///
  /// \param[in] gait Gait matrix that stores current and future contact status of the feet
  /// \param[in] goals Target positions of the four feet
  /// \param[in] baseLinearAcceleration Linear acceleration of the IMU (gravity compensated)
  /// \param[in] baseAngularVelocity Angular velocity of the IMU
  /// \param[in] baseOrientation Quaternion orientation of the IMU
  /// \param[in] q_mes Position of the 12 actuators
  /// \param[in] v_mes Velocity of the 12 actuators
  /// \param[in] dummyPos Position of the robot in PyBullet simulator (only for simulation)
  /// \param[in] b_baseVel Velocity of the robot in PyBullet simulator (only for simulation)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void run_filter(MatrixN const& gait, MatrixN const& goals, VectorN const& baseLinearAcceleration,
                  VectorN const& baseAngularVelocity, VectorN const& baseOrientation, VectorN const& q_mes,
                  VectorN const& v_mes, VectorN const& dummyPos, VectorN const& b_baseVel);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Security check to verify that measured positions, velocities and required torques
  ///        are not above defined tresholds
  ///
  /// \param[in] tau_ff Feedforward torques outputted by the whole body control for the PD+
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int security_check(VectorN const& tau_ff);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update state vectors of the robot (q and v)
  ///        Update transformation matrices between world and horizontal frames
  ///
  /// \param[in] joystick_v_ref Reference velocity from the joystick
  /// \param[in] gait Gait object
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateState(VectorN const& joystick_v_ref, Gait& gait);

  VectorN getQFilt() { return q_filt_dyn_; }
  VectorN getVFilt() { return v_filt_dyn_; }
  VectorN getVSecu() { return v_secu_dyn_; }
  VectorN getVFiltBis() { return v_filt_bis_; }
  Vector3 getRPY() { return IMU_RPY_; }
  MatrixN getFeetStatus() { return feet_status_; }
  MatrixN getFeetGoals() { return feet_goals_; }
  Vector3 getFKLinVel() { return FK_lin_vel_; }
  Vector3 getFKXYZ() { return FK_xyz_; }
  Vector3 getXYZMeanFeet() { return xyz_mean_feet_; }
  Vector3 getFiltLinVel() { return b_filt_lin_vel_; }

  Vector3 getFilterVelX() { return filter_xyz_vel_.getX(); }
  Vector3 getFilterVelDX() { return filter_xyz_vel_.getDX(); }
  Vector3 getFilterVelAlpha() { return filter_xyz_vel_.getAlpha(); }
  Vector3 getFilterVelFiltX() { return filter_xyz_vel_.getFiltX(); }

  Vector3 getFilterPosX() { return filter_xyz_pos_.getX(); }
  Vector3 getFilterPosDX() { return filter_xyz_pos_.getDX(); }
  Vector3 getFilterPosAlpha() { return filter_xyz_pos_.getAlpha(); }
  Vector3 getFilterPosFiltX() { return filter_xyz_pos_.getFiltX(); }

  VectorN getQUpdated() { return q_up_; }
  VectorN getVRef() { return v_ref_; }
  VectorN getHV() { return h_v_; }
  VectorN getHVBis() { return h_v_bis_; }
  Matrix3 getoRh() { return oRh_; }
  Vector3 getoTh() { return oTh_; }
  double getYawEstim() { return yaw_estim_; }

 private:
  ComplementaryFilter filter_xyz_pos_;  // Complementary filter for base position
  ComplementaryFilter filter_xyz_vel_;  // Complementary filter for base velocity

  double dt_wbc;           // Time step of the estimator
  double alpha_v_;         // Low pass coefficient for the outputted filtered velocity
  double alpha_secu_;      // Low pass coefficient for the outputted filtered velocity for security check
  double offset_yaw_IMU_;  // Yaw orientation of the IMU at startup
  bool perfect_estimator;  // Enable perfect estimator (directly from the PyBullet simulation)
  int N_SIMULATION;        // Number of loops before the control ends
  int k_log_;              // Number of time the estimator has been called

  Vector3 IMU_lin_acc_;                     // Linear acceleration of the IMU (gravity compensated)
  Vector3 IMU_ang_vel_;                     // Angular velocity of the IMU
  Vector3 IMU_RPY_;                         // Roll Pitch Yaw orientation of the IMU
  pinocchio::SE3::Quaternion IMU_ang_pos_;  // Quaternion orientation of the IMU
  Vector12 actuators_pos_;                  // Measured positions of actuators
  Vector12 actuators_vel_;                  // Measured velocities of actuators

  Vector19 q_FK_;           // Configuration vector for Forward Kinematics
  Vector18 v_FK_;           // Velocity vector for Forward Kinematics
  MatrixN feet_status_;     // Contact status of the four feet
  MatrixN feet_goals_;      // Target positions of the four feet
  Vector3 FK_lin_vel_;      // Base linear velocity estimated by Forward Kinematics
  Vector3 FK_xyz_;          // Base position estimated by Forward Kinematics
  Vector3 b_filt_lin_vel_;  // Filtered estimated velocity at center base (base frame)

  Vector3 xyz_mean_feet_;  // Barycenter of feet in contact

  Eigen::Matrix<double, 1, 4> k_since_contact_;  // Number of loops during which each foot has been in contact
  int feet_indexes_[4] = {10, 18, 26, 34};       // Frame indexes of the four feet

  pinocchio::Model model_, model_for_xyz_;  // Pinocchio models for frame computations and forward kinematics
  pinocchio::Data data_, data_for_xyz_;     // Pinocchio datas for frame computations and forward kinematics

  Vector19 q_filt_;  // Filtered output configuration
  Vector18 v_filt_;  // Filtered output velocity
  Vector12 v_secu_;  // Filtered output velocity for security check

  VectorN q_filt_dyn_;  // Dynamic size version of q_filt_
  VectorN v_filt_dyn_;  // Dynamic size version of v_filt_
  VectorN v_secu_dyn_;  // Dynamic size version of v_secu_

  pinocchio::SE3 _1Mi_;  // Transform between the base frame and the IMU frame

  Vector12 q_security_;  // Position limits for the actuators above which safety controller is triggered

  // For updateState function
  VectorN q_up_;      // Configuration vector
  VectorN v_ref_;     // Reference velocity vector
  VectorN h_v_;       // Velocity vector in horizontal frame
  Matrix3 oRh_;       // Rotation between horizontal and world frame
  Vector3 oTh_;       // Translation between horizontal and world frame
  double yaw_estim_;  // Yaw angle in perfect world

  int N_queue_;
  VectorN v_filt_bis_;
  VectorN h_v_bis_;
  std::deque<double> vx_queue_, vy_queue_, vz_queue_, wR_queue_, wP_queue_, wY_queue_;
};
#endif  // ESTIMATOR_H_INCLUDED
