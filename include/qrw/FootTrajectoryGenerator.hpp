///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootTrajectoryGenerator class
///
/// \details This class generates a reference trajectory for the swing foot, in position, velocity
///           and acceleration
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TRAJGEN_H_INCLUDED
#define TRAJGEN_H_INCLUDED

#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"
#include "qrw/Types.h"

class FootTrajectoryGenerator
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    FootTrajectoryGenerator();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Initialize with given data
    ///
    /// \param[in] maxHeightIn Apex height of the swinging trajectory
    /// \param[in] lockTimeIn Target lock before the touchdown
    /// \param[in] target desired target location at the end of the swing phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(Params& params, Gait& gait);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~FootTrajectoryGenerator() {}  // Empty constructor


    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief updates the nex foot position, velocity and acceleration, and the foot goal position
    ///
    /// \param[in] j foot id
    /// \param[in] targetFootstep desired target location at the end of the swing phase
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void updateFootPosition(int const j, Vector3 const& targetFootstep);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
    ///        to the desired position on the ground (computed by the footstep planner)
    ///
    /// \param[in] k (int): number of time steps since the start of the simulation
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void update(int k, MatrixN const& targetFootstep);

    Eigen::MatrixXd getFootPositionBaseFrame(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &T);
    Eigen::MatrixXd getFootVelocityBaseFrame(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &v_ref, const Eigen::Matrix<double, 3, 1> &w_ref);
    Eigen::MatrixXd getFootAccelerationBaseFrame(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &w_ref);

    MatrixN getTargetPosition() { return targetFootstep_; }  ///< Get the foot goal position
    MatrixN getFootPosition() { return position_; }          ///< Get the next foot position
    MatrixN getFootVelocity() { return velocity_; }          ///< Get the next foot velocity
    MatrixN getFootAcceleration() { return acceleration_; }  ///< Get the next foot acceleration

private:
    Gait* gait_;        ///< Target lock before the touchdown
    double dt_wbc;      ///<
    int k_mpc;          ///<
    double maxHeight_;  ///< Apex height of the swinging trajectory
    double lockTime_;   ///< Target lock before the touchdown
    double vertTime_;   ///< Duration during which feet move only along Z when taking off and landing

    std::vector<int> feet;
    Vector4 t0s;
    Vector4 t_swing;

    Matrix34 targetFootstep_;  // Target for the X component

    Matrix64 Ax;  ///< Coefficients for the X component
    Matrix64 Ay;  ///< Coefficients for the Y component

    Matrix34 position_;      // position computed in updateFootPosition
    Matrix34 velocity_;      // velocity computed in updateFootPosition
    Matrix34 acceleration_;  // acceleration computed in updateFootPosition

    Matrix34 position_base_;      // position computed in updateFootPosition in base frame
    Matrix34 velocity_base_;      // velocity computed in updateFootPosition in base frame
    Matrix34 acceleration_base_;  // acceleration computed in updateFootPosition in base frame
};
#endif  // TRAJGEN_H_INCLUDED
