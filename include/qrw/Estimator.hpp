///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Estimator class
///
/// \details This class estimates the state of the robot based on sensor measurements
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

class ComplementaryFilter
{
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
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(double dt);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the filtered output of the complementary filter
    ///
    /// \param[in] x (Vector3): quantity handled by the filter      
    /// \param[in] dx (Vector3): derivative of the quantity
    /// \param[in] alpha (Vector3): filtering coefficient between x and dx quantities
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Vector3 compute(Vector3 const& x, Vector3 const& dx, Vector3 const& alpha);

    Vector3 getX() { return x_; }           ///< Get the input quantity
    Vector3 getDX() { return dx_; }         ///< Get the derivative of the input quantity
    Vector3 getHpX() { return HP_x_; }      ///< Get the high-passed internal quantity
    Vector3 getLpX() { return LP_x_; }      ///< Get the low-passed internal quantity
    Vector3 getFiltX() { return filt_x_; }  ///< Get the filtered output

private:
    
    double dt_;
    Vector3 x_, dx_, HP_x_, LP_x_, filt_x_;
   
};


class Estimator
{
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
    /// \param[in] baseOrientation Quaternion orientation of the IMU
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void get_data_IMU(Vector3 baseLinearAcceleration, Vector3 baseAngularVelocity, Vector4 baseOrientation);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Retrieve and update position and velocity of the 12 actuators
    ///
    /// \param[in] q_mes position of the 12 actuators
    /// \param[in] v_mes velocity of the 12 actuators
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void get_data_joints(Vector12 q_mes, Vector12 v_mes);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Estimate base position and velocity using Forward Kinematics
    ///
    /// \param[in] feet_status contact status of the four feet
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void get_data_FK(Eigen::Matrix<double, 1, 4> feet_status);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute barycenter of feet in contact
    ///
    /// \param[in] feet_status contact status of the four feet
    /// \param[in] goals target positions of the four feet
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void get_xyz_feet(Eigen::Matrix<double, 1, 4> feet_status, Matrix34 goals);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Estimate the velocity of the base with forward kinematics using a contact point that 
    ///        is supposed immobile in world frame
    ///
    /// \param[in] contactFrameId frame ID of the contact point
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Vector3 BaseVelocityFromKinAndIMU(int contactFrameId);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Run one iteration of the estimator to get the position and velocity states of the robot
    ///
    /// \param[in] gait gait matrix that stores current and future contact status of the feet
    /// \param[in] goals target positions of the four feet
    /// \param[in] baseLinearAcceleration Linear acceleration of the IMU (gravity compensated)
    /// \param[in] baseAngularVelocity Angular velocity of the IMU
    /// \param[in] baseOrientation Quaternion orientation of the IMU
    /// \param[in] q_mes position of the 12 actuators
    /// \param[in] v_mes velocity of the 12 actuators
    /// \param[in] dummyPos position of the robot in PyBullet simulator (only for simulation)
    /// \param[in] b_baseVel velocity of the robot in PyBullet simulator (only for simulation)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void run_filter(MatrixN4 gait, Matrix34 goals, Vector3 baseLinearAcceleration,
                    Vector3 baseAngularVelocity, Vector4 baseOrientation, Vector12 q_mes, Vector12 v_mes,
                    Vector3 dummyPos, Vector3 b_baseVel);

private:

    ComplementaryFilter filter_xyz_pos_;  // Complementary filter for base position
    ComplementaryFilter filter_xyz_vel_;  // Complementary filter for base velocity

    double dt_wbc;  // Time step of the estimator
    double alpha_v_;  // Low pass coefficient for the outputted filtered velocity
    double alpha_secu_;  // Low pass coefficient for the outputted filtered velocity for security check
    double offset_yaw_IMU_;  // Yaw orientation of the IMU at startup
    bool perfect_estimator;  // Enable perfect estimator (directly from the PyBullet simulation)
    int N_SIMULATION;  // Number of loops before the control ends
    int k_log_;  // Number of time the estimator has been called

    Vector3 IMU_lin_acc_;  // Linear acceleration of the IMU (gravity compensated)
    Vector3 IMU_ang_vel_;  // Angular velocity of the IMU
    Vector3 IMU_RPY_;  // Roll Pitch Yaw orientation of the IMU
    pinocchio::SE3::Quaternion IMU_ang_pos_;  // Quaternion orientation of the IMU
    Vector12 actuators_pos_;  // Measured positions of actuators
    Vector12 actuators_vel_;  // Measured velocities of actuators

    Vector19 q_FK_;  // Configuration vector for Forward Kinematics
    Vector18 v_FK_;  // Velocity vector for Forward Kinematics
    Vector3 FK_lin_vel_;  // Base linear velocity estimated by Forward Kinematics
    Vector3 FK_xyz_;  // Base position estimated by Forward Kinematics

    Vector3 xyz_mean_feet_;  // Barycenter of feet in contact

    Eigen::Matrix<double, 1, 4> k_since_contact_;  // Number of loops during which each foot has been in contact
    int feet_indexes_[4] = {10, 18, 26, 34};  // Frame indexes of the four feet

    pinocchio::Model model_, model_for_xyz_;  // Pinocchio models for frame computations and forward kinematics
    pinocchio::Data data_, data_for_xyz_;  // Pinocchio datas for frame computations and forward kinematics

    Vector19 q_filt_;  // Filtered output configuration
    Vector18 v_filt_;  // Filtered output velocity
    Vector12 v_secu_;  // Filtered output velocity for security check

    pinocchio::SE3 _1Mi_;  // Transform between the base frame and the IMU frame

};
#endif  // ESTIMATOR_H_INCLUDED
