#include "qrw/Estimator.hpp"

////////////////////////////////////
// Complementary filter functions
////////////////////////////////////

ComplementaryFilter::ComplementaryFilter()
    : x_(Vector3::Zero()),
      dx_(Vector3::Zero()),
      HP_x_(Vector3::Zero()),
      LP_x_(Vector3::Zero()),
      alpha_(Vector3::Zero()),
      filt_x_(Vector3::Zero()) {}

void ComplementaryFilter::initialize(double dt, Vector3 HP_x, Vector3 LP_x) {
  dt_ = dt;
  HP_x_ = HP_x;
  LP_x_ = LP_x;
}

Vector3 ComplementaryFilter::compute(Vector3 const& x, Vector3 const& dx, Vector3 const& alpha) {
  // For logging
  x_ = x;
  dx_ = dx;
  alpha_ = alpha;

  // Process high pass filter
  HP_x_ = alpha.cwiseProduct(HP_x_ + dx_ * dt_);

  // Process low pass filter
  LP_x_ = alpha.cwiseProduct(LP_x_) + (Vector3::Ones() - alpha).cwiseProduct(x_);

  // Add both to get the filtered output
  filt_x_ = HP_x_ + LP_x_;

  return filt_x_;
}

/////////////////////////
// Estimator functions
/////////////////////////

Estimator::Estimator()
    : dt_wbc(0.0),
      alpha_secu_(0.0),
      offset_yaw_IMU_(0.0),
      perfect_estimator(false),
      N_SIMULATION(0),
      k_log_(0),
      IMU_lin_acc_(Vector3::Zero()),
      IMU_ang_vel_(Vector3::Zero()),
      IMU_RPY_(Vector3::Zero()),
      oRb_(Matrix3::Identity()),
      IMU_ang_pos_(pinocchio::SE3::Quaternion(1.0, 0.0, 0.0, 0.0)),
      actuators_pos_(Vector12::Zero()),
      actuators_vel_(Vector12::Zero()),
      q_FK_(Vector19::Zero()),
      v_FK_(Vector18::Zero()),
      feet_status_(MatrixN::Zero(1, 4)),
      feet_goals_(MatrixN::Zero(3, 4)),
      FK_lin_vel_(Vector3::Zero()),
      FK_xyz_(Vector3::Zero()),
      b_filt_lin_vel_(Vector3::Zero()),
      xyz_mean_feet_(Vector3::Zero()),
      k_since_contact_(Eigen::Matrix<double, 1, 4>::Zero()),
      q_filt_(Vector19::Zero()),
      v_filt_(Vector18::Zero()),
      v_secu_(Vector12::Zero()),
      q_filt_dyn_(VectorN::Zero(19, 1)),
      v_filt_dyn_(VectorN::Zero(18, 1)),
      v_secu_dyn_(VectorN::Zero(12, 1)),
      q_up_(VectorN::Zero(18)),
      v_up_(VectorN::Zero(18)),
      v_ref_(VectorN::Zero(6)),
      h_v_(VectorN::Zero(6)),
      oRh_(Matrix3::Identity()),
      oTh_(Vector3::Zero()),
      yaw_estim_(0.0),
      N_queue_(0),
      v_filt_bis_(VectorN::Zero(6, 1)),
      h_v_windowed_(VectorN::Zero(6, 1)) {}

void Estimator::initialize(Params& params) {
  dt_wbc = params.dt_wbc;
  N_SIMULATION = params.N_SIMULATION;
  perfect_estimator = params.perfect_estimator;

  // Filtering estimated linear velocity
  N_queue_ = static_cast<int>(std::round(params.T_gait / dt_wbc));
  vx_queue_.resize(N_queue_, 0.0);  // List full of 0.0
  vy_queue_.resize(N_queue_, 0.0);  // List full of 0.0
  vz_queue_.resize(N_queue_, 0.0);  // List full of 0.0
  wR_queue_.resize(N_queue_, 0.0);  // List full of 0.0
  wP_queue_.resize(N_queue_, 0.0);  // List full of 0.0
  wY_queue_.resize(N_queue_, 0.0);  // List full of 0.0

  // Filtering velocities used for security checks
  double fc = 6.0;  // Cut frequency
  double y = 1 - std::cos(2 * M_PI * fc * dt_wbc);
  alpha_secu_ = -y + std::sqrt(y * y + 2 * y);

  FK_xyz_(2, 0) = params.h_ref;

  filter_xyz_vel_.initialize(dt_wbc, Vector3::Zero(), Vector3::Zero());
  filter_xyz_pos_.initialize(dt_wbc, Vector3::Zero(), FK_xyz_);

  _1Mi_ = pinocchio::SE3(pinocchio::SE3::Quaternion(1.0, 0.0, 0.0, 0.0), Vector3(0.1163, 0.0, 0.02));

  q_security_ = (Vector3(M_PI * 0.4, M_PI * 80 / 180, M_PI)).replicate<4, 1>();

  q_FK_(6, 0) = 1.0;        // Last term of the quaternion
  q_filt_(6, 0) = 1.0;      // Last term of the quaternion
  q_filt_dyn_(6, 0) = 1.0;  // Last term of the quaternion

  q_up_(2, 0) = params.h_ref;                       // Reference height
  q_up_.tail(12) = Vector12(params.q_init.data());  // Actuator initial positions

  // Path to the robot URDF (TODO: Automatic path)
  const std::string filename =
      std::string("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf");

  // Build model from urdf
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_for_xyz_, false);

  // Construct data from model
  data_ = pinocchio::Data(model_);
  data_for_xyz_ = pinocchio::Data(model_for_xyz_);

  // Update all the quantities of the model
  pinocchio::computeAllTerms(model_, data_, q_filt_, v_filt_);
  pinocchio::computeAllTerms(model_for_xyz_, data_for_xyz_, q_filt_, v_filt_);
}

void Estimator::get_data_IMU(Vector3 const& baseLinearAcceleration, Vector3 const& baseAngularVelocity,
                             Vector3 const& baseOrientation) {
  // Linear acceleration of the trunk (base frame)
  IMU_lin_acc_ = baseLinearAcceleration;

  // Angular velocity of the trunk (base frame)
  IMU_ang_vel_ = baseAngularVelocity;

  // Angular position of the trunk (local frame)
  IMU_RPY_ = baseOrientation;

  if (k_log_ <= 1) {
    offset_yaw_IMU_ = IMU_RPY_(2, 0);
  }
  IMU_RPY_(2, 0) -= offset_yaw_IMU_;  // Remove initial offset of IMU

  IMU_ang_pos_ =
      pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(IMU_RPY_(0, 0), IMU_RPY_(1, 0), IMU_RPY_(2, 0)));
  // Above could be commented since IMU_ang_pos yaw is not used anywhere and instead
  // replace by: IMU_ang_pos_ = baseOrientation_
}

void Estimator::get_data_joints(Vector12 const& q_mes, Vector12 const& v_mes) {
  actuators_pos_ = q_mes;
  actuators_vel_ = v_mes;
}

void Estimator::get_data_FK(Eigen::Matrix<double, 1, 4> const& feet_status) {
  // Update estimator FK model
  q_FK_.tail(12) = actuators_pos_;  // Position of actuators
  v_FK_.tail(12) = actuators_vel_;  // Velocity of actuators
  // Position and orientation of the base remain at 0
  // Linear and angular velocities of the base remain at 0

  // Update model used for the forward kinematics
  q_FK_.block(3, 0, 4, 1) << 0.0, 0.0, 0.0, 1.0;
  pinocchio::forwardKinematics(model_, data_, q_FK_, v_FK_);
  // pin.updateFramePlacements(self.model, self.data)

  // Update model used for the forward geometry
  q_FK_.block(3, 0, 4, 1) = IMU_ang_pos_.coeffs();
  pinocchio::forwardKinematics(model_for_xyz_, data_for_xyz_, q_FK_);

  // Get estimated velocity from updated model
  int cpt = 0;
  Vector3 vel_est = Vector3::Zero();
  Vector3 xyz_est = Vector3::Zero();
  for (int j = 0; j < 4; j++) {
    // Consider only feet in contact + Security margin after the contact switch
    if (feet_status(0, j) == 1.0 && k_since_contact_[j] >= 16) {
      // Estimated velocity of the base using the considered foot
      Vector3 vel_estimated_baseframe = BaseVelocityFromKinAndIMU(feet_indexes_[j]);

      // Estimated position of the base using the considered foot
      pinocchio::updateFramePlacement(model_for_xyz_, data_for_xyz_, feet_indexes_[j]);
      Vector3 xyz_estimated = -data_for_xyz_.oMf[feet_indexes_[j]].translation();

      // Logging
      // self.log_v_est[:, i, self.k_log] = vel_estimated_baseframe[0:3, 0]
      // self.log_h_est[i, self.k_log] = xyz_estimated[2]

      // Increment counter and add estimated quantities to the storage variables
      cpt++;
      vel_est += vel_estimated_baseframe;  // Linear velocity
      xyz_est += xyz_estimated;            // Position

      double r_foot = 0.0155;  // 31mm of diameter on meshlab
      if (j <= 1) {
        vel_est(0, 0) += r_foot * (actuators_vel_(1 + 3 * j, 0) - actuators_vel_(2 + 3 * j, 0));
      } else {
        vel_est(0, 0) += r_foot * (actuators_vel_(1 + 3 * j, 0) + actuators_vel_(2 + 3 * j, 0));
      }
    }
  }

  // If at least one foot is in contact, we do the average of feet results
  if (cpt > 0) {
    FK_lin_vel_ = vel_est / cpt;
    FK_xyz_ = xyz_est / cpt;
  }
}

void Estimator::get_xyz_feet(Eigen::Matrix<double, 1, 4> const& feet_status, Matrix34 const& goals) {
  int cpt = 0;
  Vector3 xyz_feet = Vector3::Zero();

  // Consider only feet in contact
  for (int j = 0; j < 4; j++) {
    if (feet_status(0, j) == 1.0) {
      cpt++;
      xyz_feet += goals.col(j);
    }
  }

  // If at least one foot is in contact, we do the average of feet results
  if (cpt > 0) {
    xyz_mean_feet_ = xyz_feet / cpt;
  }
}

Vector3 Estimator::BaseVelocityFromKinAndIMU(int contactFrameId) {
  Vector3 frameVelocity = pinocchio::getFrameVelocity(model_, data_, contactFrameId, pinocchio::LOCAL).linear();
  pinocchio::updateFramePlacement(model_, data_, contactFrameId);

  // Angular velocity of the base wrt the world in the base frame (Gyroscope)
  Vector3 _1w01 = IMU_ang_vel_;
  // Linear velocity of the foot wrt the base in the foot frame
  Vector3 _Fv1F = frameVelocity;
  // Level arm between the base and the foot
  Vector3 _1F = data_.oMf[contactFrameId].translation();
  // Orientation of the foot wrt the base
  Matrix3 _1RF = data_.oMf[contactFrameId].rotation();
  // Linear velocity of the base wrt world in the base frame
  Vector3 _1v01 = _1F.cross(_1w01) - _1RF * _Fv1F;

  // IMU and base frames have the same orientation
  // _iv0i = _1v01 + self.cross3(self._1Mi.translation.ravel(), _1w01.ravel())

  return _1v01;
}

void Estimator::run_filter(MatrixN const& gait, MatrixN const& goals, VectorN const& baseLinearAcceleration,
                           VectorN const& baseAngularVelocity, VectorN const& baseOrientation, VectorN const& q_mes,
                           VectorN const& v_mes, VectorN const& dummyPos, Vector3 const& b_baseVel) {
  feet_status_ = gait.block(0, 0, 1, 4);
  feet_goals_ = goals;

  int remaining_steps = 1;  // Remaining MPC steps for the current gait phase
  while ((gait.block(0, 0, 1, 4)).isApprox(gait.row(remaining_steps))) {
    remaining_steps++;
  }

  // Update IMU data
  get_data_IMU(baseLinearAcceleration, baseAngularVelocity, baseOrientation);

  // Angular position of the trunk
  Vector4 filt_ang_pos = IMU_ang_pos_.coeffs();

  // Angular velocity of the trunk
  Vector3 filt_ang_vel = IMU_ang_vel_;

  // Update joints data
  get_data_joints(q_mes, v_mes);

  // Update nb of iterations since contact
  k_since_contact_ += feet_status_;                                // Increment feet in stance phase
  k_since_contact_ = k_since_contact_.cwiseProduct(feet_status_);  // Reset feet in swing phase

  // Update forward kinematics data
  get_data_FK(feet_status_);

  // Update forward geometry data
  get_xyz_feet(feet_status_, goals);

  // Tune alpha depending on the state of the gait (close to contact switch or not)
  double a = std::ceil(k_since_contact_.maxCoeff() * 0.1) - 1;
  double b = static_cast<double>(remaining_steps);
  const double n = 1;  // Nb of steps of margin around contact switch

  const double v_max = 1.00;  // Maximum alpha value
  const double v_min = 0.97;  // Minimum alpha value
  double c = ((a + b) - 2 * n) * 0.5;
  double alpha = 0.0;
  if (a <= (n - 1) || b <= n) {  // If we are close from contact switch
    alpha = v_max;               // Only trust IMU data
  } else {
    alpha = v_min + (v_max - v_min) * std::abs(c - (a - n)) / c;
    // self.alpha = 0.997
  }

  // Use cascade of complementary filters

  // Rotation matrix to go from base frame to world frame
  Matrix3 oRb = IMU_ang_pos_.toRotationMatrix();

  // Get FK estimated velocity at IMU location (base frame)
  Vector3 cross_product = (_1Mi_.translation()).cross(IMU_ang_vel_);
  Vector3 i_FK_lin_vel = FK_lin_vel_ + cross_product;

  // Get FK estimated velocity at IMU location (world frame)
  Vector3 oi_FK_lin_vel = oRb * i_FK_lin_vel;

  // Integration of IMU acc at IMU location (world frame)
  Vector3 oi_filt_lin_vel = filter_xyz_vel_.compute(oi_FK_lin_vel, oRb * IMU_lin_acc_, alpha * Vector3::Ones());

  // Filtered estimated velocity at IMU location (base frame)
  Vector3 i_filt_lin_vel = oRb.transpose() * oi_filt_lin_vel;

  // Filtered estimated velocity at center base (base frame)
  b_filt_lin_vel_ = i_filt_lin_vel - cross_product;

  // Filtered estimated velocity at center base (world frame)
  Vector3 ob_filt_lin_vel = oRb * b_filt_lin_vel_;

  // Position of the center of the base from FGeometry and filtered velocity (world frame)
  Vector3 filt_lin_pos =
      filter_xyz_pos_.compute(FK_xyz_ + xyz_mean_feet_, ob_filt_lin_vel, Vector3(0.995, 0.995, 0.9));

  // Output filtered position vector (19 x 1)
  q_filt_.head(3) = filt_lin_pos;
  if (perfect_estimator) {                    // Base height directly from PyBullet
    q_filt_(2, 0) = dummyPos(2, 0) - 0.0155;  // Minus feet radius
  }
  q_filt_.block(3, 0, 4, 1) = filt_ang_pos;
  q_filt_.tail(12) = actuators_pos_;  // Actuators pos are already directly from PyBullet

  // Output filtered velocity vector (18 x 1)
  // Linear velocities directly from PyBullet if perfect estimator
  v_filt_.head(3) = perfect_estimator ? b_baseVel : b_filt_lin_vel_;
  v_filt_.block(3, 0, 3, 1) = filt_ang_vel;  // Angular velocities are already directly from PyBullet
  v_filt_.tail(12) = actuators_vel_;         // Actuators velocities are already directly from PyBullet

  vx_queue_.pop_back();
  vy_queue_.pop_back();
  vz_queue_.pop_back();
  vx_queue_.push_front(perfect_estimator ? b_baseVel(0) : b_filt_lin_vel_(0));
  vy_queue_.push_front(perfect_estimator ? b_baseVel(1) : b_filt_lin_vel_(1));
  vz_queue_.push_front(perfect_estimator ? b_baseVel(2) : b_filt_lin_vel_(2));
  v_filt_bis_(0) = std::accumulate(vx_queue_.begin(), vx_queue_.end(), 0.0) / N_queue_;
  v_filt_bis_(1) = std::accumulate(vy_queue_.begin(), vy_queue_.end(), 0.0) / N_queue_;
  v_filt_bis_(2) = std::accumulate(vz_queue_.begin(), vz_queue_.end(), 0.0) / N_queue_;
  /*
  wR_queue_.pop_back();
  wP_queue_.pop_back();
  wY_queue_.pop_back();
  wR_queue_.push_front(filt_ang_vel(0));
  wP_queue_.push_front(filt_ang_vel(1));
  wY_queue_.push_front(filt_ang_vel(2));
  v_filt_bis_(3) = std::accumulate(wR_queue_.begin(), wR_queue_.end(), 0.0) / N_queue_;
  v_filt_bis_(4) = std::accumulate(wP_queue_.begin(), wP_queue_.end(), 0.0) / N_queue_;
  v_filt_bis_(5) = std::accumulate(wY_queue_.begin(), wY_queue_.end(), 0.0) / N_queue_;*/
  v_filt_bis_.tail(3) = filt_ang_vel;  // No filtering for angular velocity
  //////

  // Update model used for the forward kinematics
  /*pin.forwardKinematics(self.model, self.data, q_up__filt, self.v_filt)
  pin.updateFramePlacements(self.model, self.data)

  z_min = 100
  for i in (np.where(feet_status == 1))[0]:  // Consider only feet in contact
      // Estimated position of the base using the considered foot
      framePlacement = pin.updateFramePlacement(self.model, self.data, self.indexes[i])
      z_min = np.min((framePlacement.translation[2], z_min))
  q_up__filt[2, 0] -= z_min*/

  //////

  // Output filtered actuators velocity for security checks
  v_secu_ = (1 - alpha_secu_) * actuators_vel_ + alpha_secu_ * v_secu_;

  // Copy data to dynamic sized matrices since Python converters for big sized fixed matrices do not exist
  // TODO: Find a way to cast a fixed size eigen matrix as dynamic size to remove the need for those variables
  q_filt_dyn_ = q_filt_;
  v_filt_dyn_ = v_filt_;
  v_secu_dyn_ = v_secu_;

  // Increment iteration counter
  k_log_++;
}

int Estimator::security_check(VectorN const& tau_ff) {
  if (((q_filt_.tail(12).cwiseAbs()).array() > q_security_.array()).any()) {  // Test position limits
    return 1;
  } else if (((v_secu_.cwiseAbs()).array() > 50.0).any()) {  // Test velocity limits
    return 2;
  } else if (((tau_ff.cwiseAbs()).array() > 8.0).any()) {  // Test feedforward torques limits
    return 3;
  }
  return 0;
}

void Estimator::updateState(VectorN const& joystick_v_ref, Gait& gait) {
  // TODO: Joystick velocity given in base frame and not in horizontal frame (case of non flat ground)

  // Update reference velocity vector
  v_ref_.head(3) = joystick_v_ref.head(3);
  v_ref_.tail(3) = joystick_v_ref.tail(3);

  // Update position and velocity state vectors
  if (!gait.getIsStatic()) {
    // Integration to get evolution of perfect x, y and yaw
    Matrix2 Ryaw;
    Ryaw << cos(yaw_estim_), -sin(yaw_estim_), sin(yaw_estim_), cos(yaw_estim_);

    v_up_.head(2) = Ryaw * v_ref_.head(2);
    q_up_.head(2) = q_up_.head(2) + v_up_.head(2) * dt_wbc;

    // Mix perfect x and y with height measurement
    q_up_[2] = q_filt_dyn_[2];

    // Mix perfect yaw with pitch and roll measurements
    v_up_[5] = v_ref_[5];
    yaw_estim_ += v_ref_[5] * dt_wbc;
    q_up_.block(3, 0, 3, 1) << IMU_RPY_[0], IMU_RPY_[1], yaw_estim_;

    // Transformation matrices between world and base frames
    oRb_ = pinocchio::rpy::rpyToMatrix(IMU_RPY_(0, 0), IMU_RPY_(1, 0), yaw_estim_);

    // Actuators measurements
    q_up_.tail(12) = q_filt_dyn_.tail(12);
    v_up_.tail(12) = v_filt_dyn_.tail(12);

    // Velocities are the one estimated by the estimator
    Matrix3 hRb = pinocchio::rpy::rpyToMatrix(IMU_RPY_[0], IMU_RPY_[1], 0.0);

    h_v_.head(3) = hRb * v_filt_.block(0, 0, 3, 1);
    h_v_.tail(3) = hRb * v_filt_.block(3, 0, 3, 1);
    h_v_windowed_.head(3) = hRb * v_filt_bis_.block(0, 0, 3, 1);
    h_v_windowed_.tail(3) = hRb * v_filt_bis_.block(3, 0, 3, 1);
  } else {
    // TODO: Adapt static mode to new version of the code
  }

  // Transformation matrices between world and horizontal frames
  oRh_ = Matrix3::Identity();
  oRh_.block(0, 0, 2, 2) << cos(yaw_estim_), -sin(yaw_estim_), sin(yaw_estim_), cos(yaw_estim_);
  oTh_ << q_up_[0], q_up_[1], 0.0;
}
