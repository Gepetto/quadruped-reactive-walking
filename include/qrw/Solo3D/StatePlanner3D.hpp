///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for StatePlanner class
///
/// \details Planner that outputs the reference trajectory of the base based on the reference
///          velocity given by the user and the current position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STATEPLANNER3D_H_INCLUDED
#define STATEPLANNER3D_H_INCLUDED

#include "qrw/Solo3D/Heightmap.hpp"
#include "qrw/Params.hpp"

class StatePlanner3D {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  StatePlanner3D();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
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
  ~StatePlanner3D() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Udate the average surface using the heightmap and compute the configurations
  ///
  /// \param[in] q current position vector of the flying base in horizontal frame (linear and angular stacked)
  /// \param[in] vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateSurface(VectorN const& q, Vector6 const& vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the reference trajectory of the CoM for each time step of the
  ///        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
  ///        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
  ///        linear velocity and angular velocity vertically stacked. The first column contains
  ///        the current state while the remaining N columns contains the desired future states.
  ///
  /// \param[in] q current position vector of the flying base in horizontal frame (linear and angular stacked)
  /// \param[in] v current velocity vector of the flying base in horizontal frame (linear and angular stacked)
  /// \param[in] vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void computeReferenceStates(VectorN const& q, Vector6 const& v, Vector6 const& vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute references configuration for MIP planner
  ///
  /// \param[in] q current position vector of the flying base in horizontal frame (linear and angular stacked)
  /// \param[in] vref desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void computeConfigurations(VectorN const& q, Vector6 const& vref);

  MatrixN getReferenceStates() { return referenceStates_; }
  MatrixN getConfigurations() { return configs_; }
  VectorN getFit() { return fit_; }

 private:
  int nStates_;                      // Number of timesteps in the prediction horizon
  double dt_;                        // Time step of the contact sequence (time step of the MPC)
  double referenceHeight_;           // Reference height for the base
  std::vector<double> maxVelocity_;  // Maximum velocity of the base

  Vector3 rpy_;   // Current roll, pitch and yaw angles of the base
  Matrix3 Rz_;    // Rotation matrix z-axis, yaw angle
  Vector3 DxDy_;  // Temporary vector for displacement on x and y axis

  MatrixN referenceStates_;  // Reference states (12 * (1 + N)) computed for the MPC

  VectorN dtVector_;     // Vector containing all time steps in the prediction horizon
  Heightmap heightmap_;  // Environment heightmap
  Vector3 rpyMap_;       // Rotation of the heightmap in RPY
  Vector3 fit_;          // Vector3 such as [a,b,c], such as ax + by -z + c = 0 locally fits the heightmap

  int nSteps_;           // number of steps to optimize with the MIP
  double stepDuration_;  // Duration of a step
  MatrixN configs_;      // Matrix of configurations for the MIP
  Vector7 config_;       // Temp vector to store a config
  Vector3 rpyConfig_;    // Current roll, pitch and yaw angles
};

#endif  // STATEPLANNER3D_H_INCLUDED
