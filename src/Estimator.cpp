#include "qrw/Estimator.hpp"

////////////////////////////////////
// Complementary filter functions
////////////////////////////////////

ComplementaryFilter::ComplementaryFilter()
    : x_(Vector3::Zero())
    , dx_(Vector3::Zero())
    , HP_x_(Vector3::Zero())
    , LP_x_(Vector3::Zero())
    , filt_x_(Vector3::Zero())
{
}


void ComplementaryFilter::initialize(double dt, Vector3 HP_x, Vector3 LP_x)
{
    dt_ = dt;
    HP_x_ = HP_x;
    LP_x_ = LP_x;
}


Vector3 ComplementaryFilter::compute(Vector3 const& x, Vector3 const& dx, Vector3 const& alpha) 
{
    // For logging
    x_ = x;
    dx_ = dx;

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
    : dt_wbc(0.0)
    , alpha_v_(0.0)
    , alpha_secu_(0.0)
    , offset_yaw_IMU_(0.0)
    , perfect_estimator(false)
    , N_SIMULATION(0)
    , k_log_(0)
    , IMU_lin_acc_(Vector3::Zero())
    , IMU_ang_vel_(Vector3::Zero())
    , IMU_RPY_(Vector3::Zero())
    , IMU_ang_pos_(pinocchio::SE3::Quaternion(1.0, 0.0, 0.0, 0.0))
    , actuators_pos_(Vector12::Zero())
    , actuators_vel_(Vector12::Zero())
    , q_FK_(Vector19::Zero())
    , v_FK_(Vector18::Zero())
    , FK_lin_vel_(Vector3::Zero())
    , FK_xyz_(Vector3::Zero())
    , xyz_mean_feet_(Vector3::Zero())
    , k_since_contact_(Eigen::Matrix<double, 1, 4>::Zero())
    , q_filt_(Vector19::Zero())
    , v_filt_(Vector18::Zero())
    , v_secu_(Vector12::Zero())
    , q_filt_dyn_(MatrixN::Zero(19, 1))
    , v_filt_dyn_(MatrixN::Zero(18, 1))
    , v_secu_dyn_(MatrixN::Zero(12, 1))
{
}


void Estimator::initialize(Params& params)
{
    dt_wbc = params.dt_wbc;
    N_SIMULATION = params.N_SIMULATION;
    perfect_estimator = params.perfect_estimator;

    // Filtering estimated linear velocity
    double fc = 50.0;  // Cut frequency
    double y = 1 - std::cos(2 * M_PI * fc * dt_wbc);
    alpha_v_ = -y + std::sqrt(y * y + 2 * y);

    // Filtering velocities used for security checks
    fc = 6.0;  // Cut frequency
    y = 1 - std::cos(2 * M_PI * fc * dt_wbc);
    alpha_secu_ = -y + std::sqrt(y * y + 2 * y);

    FK_xyz_(2, 0) = params.h_ref;

    filter_xyz_vel_.initialize(dt_wbc, Vector3::Zero(), Vector3::Zero());
    filter_xyz_pos_.initialize(dt_wbc, Vector3::Zero(), FK_xyz_);

    _1Mi_ = pinocchio::SE3(pinocchio::SE3::Quaternion(1.0, 0.0, 0.0, 0.0), Vector3(0.1163, 0.0, 0.02));

    q_FK_(6, 0) = 1.0;  // Last term of the quaternion
    q_filt_(6, 0) = 1.0;  // Last term of the quaternion
    q_filt_dyn_(6, 0) = 1.0;  // Last term of the quaternion

    // Path to the robot URDF (TODO: Automatic path)
    const std::string filename = std::string("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf");

    // Build model from urdf
    pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);
    pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_for_xyz_, false);

    // Construct data from model
    data_ = pinocchio::Data(model_);
    data_for_xyz_ = pinocchio::Data(model_for_xyz_);

    // Update all the quantities of the model
    pinocchio::computeAllTerms(model_, data_ , q_filt_, v_filt_);
    pinocchio::computeAllTerms(model_for_xyz_, data_for_xyz_, q_filt_, v_filt_);
}


void Estimator::get_data_IMU(Vector3 baseLinearAcceleration, Vector3 baseAngularVelocity, Vector4 baseOrientation)
{
    // Linear acceleration of the trunk (base frame)
    IMU_lin_acc_ = baseLinearAcceleration;

    // Angular velocity of the trunk (base frame)
    IMU_ang_vel_ = baseAngularVelocity;

    // Angular position of the trunk (local frame)
    IMU_RPY_ = pinocchio::rpy::matrixToRpy(pinocchio::SE3::Quaternion(baseOrientation).toRotationMatrix());

    if(k_log_ <= 1) {
        offset_yaw_IMU_ = IMU_RPY_(2, 0);
    }
    IMU_RPY_(2, 0) -= offset_yaw_IMU_;  // Remove initial offset of IMU

    IMU_ang_pos_ = pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(IMU_RPY_(0, 0),
                                                                IMU_RPY_(1, 0),
                                                                IMU_RPY_(2, 0)));
    // Above could be commented since IMU_ang_pos yaw is not used anywhere and instead
    // replace by: IMU_ang_pos_ = baseOrientation_
}


void Estimator::get_data_joints(Vector12 q_mes, Vector12 v_mes) 
{
    actuators_pos_ = q_mes;
    actuators_vel_ = v_mes;
}

      
void Estimator::get_data_FK(Eigen::Matrix<double, 1, 4> feet_status)
{
    // Update estimator FK model
    q_FK_.tail(12) = actuators_pos_; // Position of actuators
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
        if(feet_status(0, j) == 1.0 && k_since_contact_[j] >= 16)
        {  
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
            xyz_est += xyz_estimated;  // Position

            double r_foot = 0.025; // 0.0155  // 31mm of diameter on meshlab
            if(j <= 1) {
                vel_est(0, 0) += r_foot * (actuators_vel_(1+3*j, 0) - actuators_vel_(2+3*j, 0));
            } 
            else {
                vel_est(0, 0) += r_foot * (actuators_vel_(1+3*j, 0) + actuators_vel_(2+3*j, 0));
            }
        }
    }

    // If at least one foot is in contact, we do the average of feet results
    if(cpt > 0) 
    {
        FK_lin_vel_ = vel_est / cpt;
        FK_xyz_ = xyz_est / cpt;
    }
}


void Estimator::get_xyz_feet(Eigen::Matrix<double, 1, 4> feet_status, Matrix34 goals)
{
    int cpt = 0;
    Vector3 xyz_feet = Vector3::Zero();

    // Consider only feet in contact
    for (int j = 0; j < 4; j++) {
        if(feet_status(0, j) == 1.0) {  
            cpt++;
            xyz_feet += goals.col(j);
        }
    }

    // If at least one foot is in contact, we do the average of feet results
    if(cpt > 0) {
        xyz_mean_feet_ = xyz_feet / cpt;
    }
}


Vector3 Estimator::BaseVelocityFromKinAndIMU(int contactFrameId)
{
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
                           VectorN const& v_mes, VectorN const& dummyPos, VectorN const& b_baseVel)
{
    int remaining_steps = 1;  // Remaining MPC steps for the current gait phase
    while((gait.block(0, 0, 1, 4)).isApprox(gait.row(remaining_steps))) {
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
    k_since_contact_ += gait.block(0, 0, 1, 4);  // Increment feet in stance phase
    k_since_contact_ = k_since_contact_.cwiseProduct(gait.block(0, 0, 1, 4));  // Reset feet in swing phase

    // Update forward kinematics data
    get_data_FK(gait.block(0, 0, 1, 4));

    // Update forward geometry data
    get_xyz_feet(gait.block(0, 0, 1, 4), goals);

    // Tune alpha depending on the state of the gait (close to contact switch or not)
    double a = std::ceil(k_since_contact_.maxCoeff() * 0.1) - 1;
    double b = static_cast<double>(remaining_steps);
    const double n = 1;  // Nb of steps of margin around contact switch

    const double v_max = 1.00;  // Maximum alpha value
    const double v_min = 0.97;  // Minimum alpha value
    double c = ((a + b) - 2 * n) * 0.5;
    double alpha = 0.0;
    if(a <= (n-1) || b <= n) {  // If we are close from contact switch
        alpha = v_max;  // Only trust IMU data
    }
    else {
        alpha = v_min + (v_max - v_min) * std::abs(c - (a - n)) / c;
        //self.alpha = 0.997
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
    Vector3 oi_filt_lin_vel = filter_xyz_vel_.compute(oi_FK_lin_vel, oRb * IMU_lin_acc_,
                                                      alpha * Vector3::Ones());

    // Filtered estimated velocity at IMU location (base frame)
    Vector3 i_filt_lin_vel = oRb.transpose() * oi_filt_lin_vel;

    // Filtered estimated velocity at center base (base frame)
    Vector3 b_filt_lin_vel = i_filt_lin_vel - cross_product;

    // Filtered estimated velocity at center base (world frame)
    Vector3 ob_filt_lin_vel = oRb * b_filt_lin_vel;

    // Position of the center of the base from FGeometry and filtered velocity (world frame)
    Vector3 filt_lin_pos = filter_xyz_pos_.compute(FK_xyz_ + xyz_mean_feet_, ob_filt_lin_vel, 
                                                   Vector3(0.995, 0.995, 0.9));

    // Velocity of the center of the base (base frame)
    Vector3 filt_lin_vel = b_filt_lin_vel;

    // Logging
    /*self.log_alpha[self.k_log] = self.alpha
    self.feet_status[:] = feet_status  // Save contact status sent to the estimator for logging
    self.feet_goals[:, :] = goals.copy()  // Save feet goals sent to the estimator for logging
    self.log_IMU_lin_acc[:, self.k_log] = self.IMU_lin_acc[:]
    self.log_HP_lin_vel[:, self.k_log] = self.HP_lin_vel[:]
    self.log_LP_lin_vel[:, self.k_log] = self.LP_lin_vel[:]
    self.log_FK_lin_vel[:, self.k_log] = self.FK_lin_vel[:]
    self.log_filt_lin_vel[:, self.k_log] = self.filt_lin_vel[:]
    self.log_o_filt_lin_vel[:, self.k_log] = self.o_filt_lin_vel[:, 0]*/

    // Output filtered position vector (19 x 1)
    q_filt_.head(3) = filt_lin_pos;
    if(perfect_estimator) {  // Base height directly from PyBullet
        q_filt_(2, 0) = dummyPos(2, 0) - 0.0155;  // Minus feet radius
    }
    q_filt_.block(3, 0, 4, 1) = filt_ang_pos;
    q_filt_.tail(12) = actuators_pos_;  // Actuators pos are already directly from PyBullet

    // Output filtered velocity vector (18 x 1)
    if(perfect_estimator) {  // Linear velocities directly from PyBullet
        v_filt_.head(3) = (1 - alpha_v_) * v_filt_.head(3) + alpha_v_ * b_baseVel;
    }
    else {
        v_filt_.head(3) = (1 - alpha_v_) * v_filt_.head(3) + alpha_v_ * filt_lin_vel;
    }
    v_filt_.block(3, 0, 3, 1) = filt_ang_vel;  // Angular velocities are already directly from PyBullet
    v_filt_.tail(12) = actuators_vel_;  // Actuators velocities are already directly from PyBullet

    //////

    // Update model used for the forward kinematics
    /*pin.forwardKinematics(self.model, self.data, self.q_filt, self.v_filt)
    pin.updateFramePlacements(self.model, self.data)

    z_min = 100
    for i in (np.where(feet_status == 1))[0]:  // Consider only feet in contact
        // Estimated position of the base using the considered foot
        framePlacement = pin.updateFramePlacement(self.model, self.data, self.indexes[i])
        z_min = np.min((framePlacement.translation[2], z_min))
    self.q_filt[2, 0] -= z_min*/

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
