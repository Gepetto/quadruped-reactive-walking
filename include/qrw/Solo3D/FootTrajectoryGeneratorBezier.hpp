///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootTrajectoryGenerator class
///
/// \details This class generates a reference trajectory for the swing foot, in position, velocity
///           and acceleration
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TRAJGEN_BEZIER_H_INCLUDED
#define TRAJGEN_BEZIER_H_INCLUDED

#include <vector>

#include "eiquadprog/eiquadprog-fast.hpp"
#include "ndcurves/bezier_curve.h"
#include "ndcurves/fwd.h"
#include "ndcurves/optimization/details.h"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"
#include "qrw/Solo3D/Surface.hpp"

using namespace ndcurves;
using namespace eiquadprog::solvers;

typedef std::vector<Surface> SurfaceVector;

class FootTrajectoryGeneratorBezier {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  FootTrajectoryGeneratorBezier();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] maxHeightIn Apex height of the swinging trajectory
  /// \param[in] lockTimeIn Target lock before the touchdown
  /// \param[in] target desired target location at the end of the swing phase
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params& params, Gait& gait, Surface initialSurface_in, double x_margin_max_in, double t_margin_in,
                  double z_margin_in, int N_samples_in, int N_samples_ineq_in, int degree_in);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~FootTrajectoryGeneratorBezier() {}  // Empty constructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief updates the nex foot position, velocity and acceleration, and the foot goal position
  ///
  /// \param[in] k (int): number of time steps since the start of the simulation
  /// \param[in] j (int): index of the foot
  /// \param[in] targetFootstep (Vector3): desired target location at the end of the swing phase
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateFootPosition(int const& k, int const& i_foot, Vector3 const& targetFootstep);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
  ///        to the desired position on the ground (computed by the footstep planner)
  ///
  /// \param[in] k (int): number of time steps since the start of the simulation
  /// \param[in] surfacesSelected (SurfaceVector): Vector of contact surfaces for each foot
  /// \param[in] targetFootstep (Matrix34): desired target location at the end of the swing phase
  /// \param[in] q (Vector18): State of the robot, (RPY formulation, size 18)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update(int k, MatrixN const& targetFootstep, SurfaceVector const& surfacesSelected, VectorN const& q);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the 3D position of the feet in world frame by forward kinematic, matrix position_FK_
  ///
  /// \param[in] q (Vector18): State of the robot, (RPY formulation, size 18)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_position_FK(VectorN const& q);

  void updatePolyCoeff_XY(int const& i_foot, Vector3 const& x_init, Vector3 const& v_init, Vector3 const& a_init,
                          Vector3 const& x_target, double const& t0, double const& t1);
  void updatePolyCoeff_Z(int const& i_foot, Vector3 const& x_init, Vector3 const& x_target, double const& t1,
                         double const& h);
  Vector3 evaluatePoly(int const& i_foot, int const& indice, double const& t);
  Vector3 evaluateBezier(int const& i_foot, int const& indice, double const& t);

  Eigen::MatrixXd getFootPositionBaseFrame(const Eigen::Matrix<double, 3, 3>& R, const Eigen::Matrix<double, 3, 1>& T);
  Eigen::MatrixXd getFootVelocityBaseFrame(const Eigen::Matrix<double, 3, 3>& R,
                                           const Eigen::Matrix<double, 3, 1>& v_ref,
                                           const Eigen::Matrix<double, 3, 1>& w_ref);
  Eigen::MatrixXd getFootAccelerationBaseFrame(const Eigen::Matrix<double, 3, 3>& R,
                                               const Eigen::Matrix<double, 3, 1>& w_ref,
                                               const Eigen::Matrix<double, 3, 1>& a_ref);

  MatrixN getTargetPosition() { return targetFootstep_; }  ///< Get the foot goal position
  MatrixN getFootPosition() { return position_; }          ///< Get the next foot position
  MatrixN getFootVelocity() { return velocity_; }          ///< Get the next foot velocity
  MatrixN getFootAcceleration() { return acceleration_; }  ///< Get the next foot acceleration
  MatrixN getFootJerk() { return jerk_; }                  // Get the next foot jerk
  Vector4 get_t0s() { return t0s; }
  Vector4 get_t_swing() { return t_swing; }

 private:
  Gait* gait_;        ///< Target lock before the touchdown
  double dt_wbc;      ///<
  int k_mpc;          ///<
  double maxHeight_;  ///< Apex height of the swinging trajectory
  double lockTime_;   ///< Target lock before the touchdown

  std::vector<int> feet;
  Vector4 t0s;
  Vector4 t0_bezier;
  Vector4 t_stop;
  Vector4 t_swing;

  Matrix34 targetFootstep_;  // Target for the X component

  Matrix64 Ax;  ///< Coefficients for the X component
  Matrix64 Ay;  ///< Coefficients for the Y component
  Matrix74 Az;  ///< Coefficients for the Z component

  Matrix34 position_;      // position computed in updateFootPosition
  Matrix34 position_FK_;   // position computed by Forward dynamics
  Matrix34 velocity_;      // velocity computed in updateFootPosition
  Matrix34 acceleration_;  // acceleration computed in updateFootPosition
  Matrix34 jerk_;          // Jerk computed in updateFootPosition

  Matrix34 position_base_;      // Position computed in updateFootPosition in base frame
  Matrix34 velocity_base_;      // Velocity computed in updateFootPosition in base frame
  Matrix34 acceleration_base_;  // Acceleration computed in updateFootPosition in base frame

  Vector2 intersectionPoint_;  // tmp point of intersection

  SurfaceVector newSurface_;
  SurfaceVector pastSurface_;

  typedef optimization::problem_definition<pointX_t, double> problem_definition_t;
  typedef optimization::problem_data<pointX_t, double> problem_data_t;
  static const bool safe = true;
  bool useBezier;

  // Number of points in the least square problem
  int N_samples;
  int N_samples_ineq;
  //  dimension of our problem (here 3 as our curve is 3D)
  int dim = 3;
  // Degree of the Bezier curve to match the polys
  int degree;
  // Size of the optimised vector in the least square (6 = nb of initial/final conditions)
  // = dim*(degree + 1 - 6)
  int res_size;

  // Vector of Problem Definition, containing Initial/Final conditions and flag for conditions.
  std::vector<problem_definition_t> pDefs;

  // Vector of Bezier curves, containing the curves optimised for each foot
  std::vector<bezier_t> fitBeziers;

  // QP matrix
  MatrixN P_;
  VectorN q_;

  MatrixN G_;
  VectorN h_;

  MatrixN C_;
  VectorN d_;

  VectorN x;

  std::vector<Vector3> ineq_;
  Vector4 ineq_vector_;
  Vector4 x_margin_;
  double x_margin_max_;  // margin around the obstacle
  double t_margin_;      // % of the curve after critical point
  double z_margin_;      // % of the height of the obstacle around the critical point

  // QP solver
  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;
  EiquadprogFast_status status;
  EiquadprogFast qp;

  // Pinocchio model for foot estimation
  pinocchio::Model model_;          // Pinocchio model for forward kinematics
  pinocchio::Data data_;            // Pinocchio datas for forward kinematics
  int foot_ids_[4] = {0, 0, 0, 0};  // Indexes of feet frames
  Matrix34 pos_feet_;               // Estimated feet positions based on measurements
  Vector19 q_FK_;                   // Estimated state of the base (height, roll, pitch, joints)

  // Methods to compute intersection point
  bool doIntersect_segment(Vector2 const& p1, Vector2 const& q1, Vector2 const& p2, Vector2 const& q2);
  bool onSegment(Vector2 const& p, Vector2 const& q, Vector2 const& r);
  int orientation(Vector2 const& p, Vector2 const& q, Vector2 const& r);
  void get_intersect_segment(Vector2 a1, Vector2 a2, Vector2 b1, Vector2 b2);
};
#endif  // TRAJGEN_H_INCLUDED
