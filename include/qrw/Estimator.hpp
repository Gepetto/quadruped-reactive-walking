///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Estimator and ComplementaryFilter classes
///
/// \details These classes estimate the state of the robot based on sensor measurements
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ESTIMATOR_H_INCLUDED
#define ESTIMATOR_H_INCLUDED

#include <deque>

#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "qrw/ComplementaryFilter.hpp"
#include "qrw/Params.hpp"

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
  /// \brief Run one iteration of the estimator to get the position and velocity states of the robot
  ///
  /// \param[in] gait Gait matrix that stores current and future contact status of the feet
  /// \param[in] goals Target positions of the four feet
  /// \param[in] baseLinearAcceleration Linear acceleration of the IMU (gravity compensated)
  /// \param[in] baseAngularVelocity Angular velocity of the IMU
  /// \param[in] baseOrientation Quaternion orientation of the IMU
  /// \param[in] q_mes Position of the 12 actuators
  /// \param[in] v_mes Velocity of the 12 actuators
  /// \param[in] perfectPosition Position of the robot in world frame
  /// \param[in] b_perfectVelocity Velocity of the robot in base frame
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void run(MatrixN const& gait, MatrixN const& goals, VectorN const& baseLinearAcceleration,
           VectorN const& baseAngularVelocity, VectorN const& baseOrientation, VectorN const& q_mes,
           VectorN const& v_mes, VectorN const& perfectPosition, Vector3 const& b_perfectVelocity);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update state vectors of the robot (q and v)
  ///        Update transformation matrices between world and horizontal frames
  ///
  /// \param[in] joystick_v_ref Reference velocity from the joystick
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateReferenceState(VectorN const& vRef);

  VectorN getQEstimate() { return qEstimate_; }
  VectorN getVEstimate() { return vEstimate_; }
  VectorN getVSecurity() { return vSecurity_; }
  VectorN getFeetStatus() { return feetStatus_; }
  MatrixN getFeetTargets() { return feetTargets_; }
  Vector3 getBaseVelocityFK() { return baseVelocityFK_; }
  Vector3 getBasePositionFK() { return basePositionFK_; }
  Vector3 getFeetPositionBarycenter() { return feetPositionBarycenter_; }
  Vector3 getBBaseVelocity() { return b_baseVelocity_; }

  Vector3 getFilterVelX() { return velocityFilter_.getX(); }
  Vector3 getFilterVelDX() { return velocityFilter_.getDx(); }
  Vector3 getFilterVelAlpha() { return velocityFilter_.getAlpha(); }
  Vector3 getFilterVelFiltX() { return velocityFilter_.getFilteredX(); }

  Vector3 getFilterPosX() { return positionFilter_.getX(); }
  Vector3 getFilterPosDX() { return positionFilter_.getDx(); }
  Vector3 getFilterPosAlpha() { return positionFilter_.getAlpha(); }
  Vector3 getFilterPosFiltX() { return positionFilter_.getFilteredX(); }

  VectorN getQReference() { return qRef_; }
  VectorN getVReference() { return vRef_; }
  VectorN getBaseVelRef() { return baseVelRef_; }
  VectorN getBaseAccRef() { return baseAccRef_; }
  VectorN getHV() { return h_v_; }
  VectorN getVFiltered() { return vFiltered_; }
  VectorN getHVFiltered() { return h_vFiltered_; }
  Matrix3 getoRh() { return oRh_; }
  Matrix3 gethRb() { return hRb_; }
  Vector3 getoTh() { return oTh_; }

 private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Retrieve and update IMU data
  ///
  /// \param[in] baseLinearAcceleration Linear acceleration of the IMU (gravity compensated)
  /// \param[in] baseAngularVelocity Angular velocity of the IMU
  /// \param[in] baseOrientation Euler orientation of the IMU
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateIMUData(Vector3 const& baseLinearAcceleration, Vector3 const& baseAngularVelocity,
                     Vector3 const& baseOrientation, VectorN const& perfectPosition);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Retrieve and update position and velocity of the 12 actuators
  ///
  /// \param[in] q Position of the 12 actuators
  /// \param[in] v Velocity of the 12 actuators
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateJointData(Vector12 const& q, Vector12 const& v);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the feet relative data
  /// \details update feetStatus_, feetTargets_, feetStancePhaseDuration_ and phaseRemainingDuration_
  ///
  /// \param[in] gait Gait matrix that stores current and future contact status of the feet
  /// \param[in] feetTargets Target positions of the four feet
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updatFeetStatus(MatrixN const& gait, MatrixN const& feetTargets);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimate base position and velocity using Forward Kinematics
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateForwardKinematics();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute barycenter of feet in contact
  ///
  /// \param[in] feet_status Contact status of the four feet
  /// \param[in] goals Target positions of the four feet
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void computeFeetPositionBarycenter();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimate the velocity of the base with forward kinematics using a contact point that
  ///        is supposed immobile in world frame
  ///
  /// \param[in] contactFrameId Frame ID of the contact foot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector3 computeBaseVelocityFromFoot(int footId);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimate the position of the base with forward kinematics using a contact point that
  ///        is supposed immobile in world frame
  ///
  /// \param[in] footId Frame ID of the contact foot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector3 computeBasePositionFromFoot(int footId);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the alpha coefficient for the complementary filter
  /// \return alpha
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double computeAlphaVelocity();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimates the velocity vector
  /// \details The complementary filter combines data from the FK and the IMU acceleration data
  ///
  /// \param[in] b_perfectVelocity Perfect velocity of the base in the base frame
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void estimateVelocity(Vector3 const& b_perfectVelocity);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Estimates the configuration vector
  /// \details The complementary filter combines data from the FK and the estimated velocity
  ///
  /// \param[in] perfectPosition Perfect position of the base in the world frame
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void estimatePosition(Vector3 const& perfectPosition);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Filter the estimated velocity over a moving window
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void filterVelocity();

  bool perfectEstimator_;  // Enable perfect estimator (directly from the PyBullet simulation)
  bool solo3D_;            // Perfect estimator including yaw angle
  double dt_;              // Time step of the estimator
  bool initialized_;       // Is intiialized after the first update of the IMU
  Vector4 feetFrames_;     // Frame indexes of the four feet
  double footRadius_;      // radius of a foot
  Vector3 alphaPos_;       // Alpha coeefficient for the position complementary filter
  double alphaVelMax_;     // Maximum alpha value for the velocity complementary filter
  double alphaVelMin_;     // Minimum alpha value for the velocity complementary filter
  double alphaSecurity_;   // Low pass coefficient for the outputted filtered velocity for security check

  pinocchio::SE3 b_M_IMU_;              // Transform between the base frame and the IMU frame
  double IMUYawOffset_;                 // Yaw orientation of the IMU at startup
  Vector3 IMULinearAcceleration_;       // Linear acceleration of the IMU (gravity compensated)
  Vector3 IMUAngularVelocity_;          // Angular velocity of the IMU
  Vector3 IMURpy_;                      // Roll Pitch Yaw orientation of the IMU
  pinocchio::SE3::Quaternion IMUQuat_;  // Quaternion orientation of the IMU

  Vector12 qActuators_;  // Measured positions of actuators
  Vector12 vActuators_;  // Measured velocities of actuators

  int phaseRemainingDuration_;       // Number of iterations left for the current gait phase
  Vector4 feetStancePhaseDuration_;  // Number of loops during which each foot has been in contact
  Vector4 feetStatus_;               // Contact status of the four feet
  Matrix34 feetTargets_;             // Target positions of the four feet

  pinocchio::Model velocityModel_, positionModel_;  // Pinocchio models for frame computations and forward kinematics
  pinocchio::Data velocityData_, positionData_;     // Pinocchio datas for frame computations and forward kinematics
  Vector19 q_FK_;                                   // Configuration vector for Forward Kinematics
  Vector18 v_FK_;                                   // Velocity vector for Forward Kinematics
  Vector3 baseVelocityFK_;                          // Base linear velocity estimated by Forward Kinematics
  Vector3 basePositionFK_;                          // Base position estimated by Forward Kinematics
  Vector3 b_baseVelocity_;                          // Filtered estimated velocity at center base (base frame)
  Vector3 feetPositionBarycenter_;                  // Barycenter of feet in contact

  ComplementaryFilter positionFilter_;  // Complementary filter for base position
  ComplementaryFilter velocityFilter_;  // Complementary filter for base velocity
  Vector19 qEstimate_;                  // Filtered output configuration
  Vector18 vEstimate_;                  // Filtered output velocity
  Vector12 vSecurity_;                  // Filtered output velocity for security check

  int windowSize_;                                     // Number of samples in the averaging window
  Vector6 vFiltered_;                                  // Base velocity (in base frame) filtered by averaging window
  std::deque<double> vx_queue_, vy_queue_, vz_queue_;  // Queues that hold samples

  Vector18 qRef_;        // Configuration vector in ideal world frame
  Vector18 vRef_;        // Velocity vector in ideal world frame
  Vector6 baseVelRef_;   // Reference velocity vector
  Vector6 baseAccRef_;   // Reference acceleration vector
  Matrix3 oRh_;          // Rotation between horizontal and world frame
  Matrix3 hRb_;          // Rotation between base and horizontal frame
  Vector3 oTh_;          // Translation between horizontal and world frame
  Vector6 h_v_;          // Velocity vector in horizontal frame
  Vector6 h_vFiltered_;  // Base velocity (in horizontal frame) filtered by averaging window
};
#endif  // ESTIMATOR_H_INCLUDED
