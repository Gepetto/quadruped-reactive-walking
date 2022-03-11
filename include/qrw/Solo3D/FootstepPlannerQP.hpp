///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for FootstepPlanner class
///
/// \details Planner that outputs current and future locations of footsteps, the reference
///          trajectory of the base based on the reference velocity given by the user and the current
///          position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FOOTSTEPPLANNERQP_H_INCLUDED
#define FOOTSTEPPLANNERQP_H_INCLUDED

#include <vector>

#include "eiquadprog/eiquadprog-fast.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "qrw/Gait.hpp"
#include "qrw/Params.hpp"
#include "qrw/Solo3D/Surface.hpp"

// Order of feet/legs: FL, FR, HL, HR

using namespace eiquadprog::solvers;
typedef std::vector<Surface> SurfaceVector;
typedef std::vector<std::vector<Surface>> SurfaceVectorVector;

struct Pair {
  double F;  // First
  double S;  // Second
};

struct optimData {
  int phase;
  int foot;
  Surface surface;
  Vector3 constant_term;
  Matrix3 Rz_tmp;
};

class FootstepPlannerQP {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  FootstepPlannerQP();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Default constructor
  ///
  /// \param[in] params Object that stores parameters
  /// \param[in] gaitIn Gait object to hold the gait informations
  /// \param[in] initialSurface_in Surface of the floor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(Params& params, Gait& gaitIn, Surface initialSurface_in);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~FootstepPlannerQP() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh footsteps locations (computation and update of relevant matrices)
  ///
  ///  \param[in] refresh True if we move one step further in the gait
  ///  \param[in] k Number of remaining wbc time step for the current mpc time step (wbc frequency is higher so there
  ///  are inter-steps)
  ///  \param[in] q Current position vector of the flying base in horizontal frame (linear and angular stacked) +
  ///  actuators
  ///  \param[in] b_v Current velocity vector of the flying base in horizontal frame (linear and angular
  ///  stacked)
  ///  \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and angular
  ///  stacked)
  ///
  ///  Precondition : updateSurfaces should have been called to store the new surface planner result
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN updateFootsteps(bool refresh, int k, VectorN const& q, Vector6 const& b_v, Vector6 const& b_vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Updates the surfaces result from the surface planner
  ///
  ///////////////////////////////////////////////////////////////////////////////////////////////
  void updateSurfaces(SurfaceVectorVector const& potentialSurfaces, SurfaceVector const& surfaces,
                      bool const surfaceStatus, int const surfaceIteration);

  MatrixN getFootsteps();
  MatrixN getTargetFootsteps();
  MatrixN getRz();

 private:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the desired location of footsteps and update relevant matrices
  ///
  ///  \param[in] k Number of remaining wbc time step for the current mpc time step (wbc frequency is higher so there
  ///  are inter-steps)
  ///  \param[in] q Current position vector of the flying base in horizontal frame (linear and
  ///  angular stacked)
  ///  \param[in] b_v Current velocity vector of the flying base in horizontal frame (linear and
  ///  angular stacked)
  ///  \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and
  ///  angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN computeTargetFootstep(int k, Vector6 const& q, Vector6 const& b_v, Vector6 const& b_vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh feet position when entering a new contact phase
  ///
  ///  \param[in] q Current configuration vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateNewContact(Vector18 const& q);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first
  /// column)
  ///        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
  ///        For feet currently touching the ground the desired position is where they currently are.
  ///
  /// \param[in] k Number of remaining wbc time step for the current mpc time step (wbc frequency is higher so there
  ///  are inter-steps)
  /// \param[in] b_v Current velocity vector of sthe flying base in horizontal frame (linear and angular stacked)
  /// \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void computeFootsteps(int k, Vector6 const& b_v, Vector6 const& b_vref);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the target location on the ground of a given foot for an upcoming stance phase
  ///
  /// \param[in] i Considered phase (row of the gait matrix)
  /// \param[in] j Considered foot (col of the gait matrix)
  /// \param[in] b_v Current velocity vector of sthe flying base in horizontal frame (linear and angular stacked)
  /// \param[in] b_vref Desired velocity vector of the flying base in horizontal frame (linear and angular stacked)
  /// \param[in] footstep Vector x3 to update with the next foostep position of foot j, gait row i
  /// \param[in] feedback_term Boolean if the feedback term is taken into account
  ///
  /// \retval Vector of the next footstep position for foot j, gait index i
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void computeNextFootstep(int i, int j, Vector6 const& b_v, Vector6 const& b_vref, Vector3& footstep,
                           bool feedback_term);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update desired location of footsteps using information coming from the footsteps planner
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void updateTargetFootsteps();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Select a surface from a point in 2D inside potential surfaces
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Surface selectSurfaceFromPoint(Vector3 const& point, int phase, int moving_foot_index);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the remaining timing of the flying phase
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_remaining_time(int k);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the QP problem with the surface object
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int surfaceInequalities(int i_start, Surface const& surface, Vector3 const& next_ft, int id_foot, Matrix3 Rz_tmp);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Transform a std::vector of N 3x4 matrices into a single Nx12 matrix
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  MatrixN vectorToMatrix(std::vector<Matrix34> const& array);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute a distance from a point to a segment.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double minDistance(Pair const& A, Pair const& B, Pair const& E);

  Params* params_;  // Params object to store parameters
  Gait* gait_;      // Gait object to hold the gait informations

  double dt;      // Time step of the contact sequence (time step of the MPC)
  double dt_wbc;  // Time step of the whole body control
  double h_ref;   // Reference height for the trunk

  // Predefined quantities
  double g;  // Value of the gravity acceleartion
  double L;  // Value of the maximum allowed deviation due to leg length

  // Number of time steps in the prediction horizon
  int n_steps;  // T_mpc / time step of the MPC

  // Constant sized matrices
  Matrix34 footsteps_under_shoulders_;  // Positions of footsteps to be "under the shoulder"
  Matrix34 footsteps_offset_;           // Offset positions of footsteps
  Matrix34 currentFootstep_;            // Feet matrix
  Vector3 heuristic_fb_;                // Tmp vector3, heuristic term with feedback term
  Vector3 heuristic_;                   // Tmp vector3, heuristic term without feedback term
  Matrix34 targetFootstep_;             // In horizontal frame
  Matrix34 o_targetFootstep_;           // targetFootstep_ in world frame
  std::vector<Matrix34> footsteps_;     // Desired footsteps locations for each step of the horizon
  std::vector<Matrix34> b_footsteps_;   // Desired footsteps locations for each step of the horizon in base frame

  MatrixN Rz;      // Rotation matrix along z axis
  MatrixN Rz_tmp;  // Temporary rotation matrix along z axis
  VectorN dt_cum;  // Cumulated time vector
  VectorN yaws;    // Predicted yaw variation for each cumulated time in base frame
  VectorN dx;      // Predicted x displacement for each cumulated time
  VectorN dy;      // Predicted y displacement for each cumulated time

  Vector3 q_dxdy;  // Temporary storage variable for offset to the future position
  Vector3 q_tmp;   // Temporary storage variable for position of the base in the world
  Vector3 RPY_;    // Temporary storage variable for roll pitch yaw orientation

  pinocchio::Model model_;          // Pinocchio model for forward kinematics
  pinocchio::Data data_;            // Pinocchio datas for forward kinematics
  int foot_ids_[4] = {0, 0, 0, 0};  // Indexes of feet frames
  Matrix34 pos_feet_;               // Estimated feet positions based on measurements
  Vector19 q_FK_;

  int k_mpc;
  // QP problem :
  const int N = 14;     // Number of variables vx, vy, p1, p2, p3
  const int M = 6 * 4;  // Number of constraints inequalities

  std::vector<int> feet_;
  Vector4 t0s;
  Vector4 t_swing;

  VectorN weights_;
  Vector3 b_voptim;  // Velocity vector optimised in base frame
  Vector3 delta_x;   // Tmp Vector3 to store results from optimisation

  // Eiquadprog-Fast solves the problem :
  // min. 1/2 * x' C_ x + q_' x
  // s.t. C_ x + d_ = 0
  //      G_ x + h_ >= 0

  // Weight Matrix
  MatrixN P_;
  VectorN q_;

  MatrixN G_;
  VectorN h_;

  MatrixN C_;
  VectorN d_;

  // qp solver
  EiquadprogFast_status expected = EIQUADPROG_FAST_OPTIMAL;
  EiquadprogFast_status status;
  VectorN x;
  EiquadprogFast qp;

  bool useSL1M;
  bool surfaceStatus_;
  int surfaceIteration_;
  SurfaceVector surfaces_;
  SurfaceVectorVector potentialSurfaces_;
  Surface initialSurface_;

  std::vector<optimData> optimVector_;
  SurfaceVector selectedSurfaces_;
};

#endif  // FOOTSTEPPLANNERQP_H_INCLUDED
