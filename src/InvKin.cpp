#include "qrw/InvKin.hpp"

InvKin::InvKin()
    : invJ(Matrix12::Zero())
    , acc(Matrix112::Zero())
    , x_err(Matrix112::Zero())
    , dx_r(Matrix112::Zero())
    , pfeet_err(Matrix43::Zero())
    , vfeet_ref(Matrix43::Zero())
    , afeet(Matrix43::Zero())
    , posf_(Matrix43::Zero())
    , vf_(Matrix43::Zero())
    , wf_(Matrix43::Zero())
    , af_(Matrix43::Zero())
    , Jf_(Matrix12::Zero())
    , Jf_tmp_(Eigen::Matrix<double, 6, 12>::Zero())
    , ddq_cmd_(Vector12::Zero())
    , dq_cmd_(Vector12::Zero())
    , q_cmd_(Vector12::Zero())
    , q_step_(Vector12::Zero())
{}

void InvKin::initialize(Params& params) {

    // Params store parameters
    params_ = &params;

    // Path to the robot URDF (TODO: Automatic path)
    const std::string filename = std::string("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf");

    // Build model from urdf (base is not free flyer)
    pinocchio::urdf::buildModel(filename, model_, false);

    // Construct data from model
    data_ = pinocchio::Data(model_);

    // Update all the quantities of the model
    pinocchio::computeAllTerms(model_, data_ , VectorN::Zero(model_.nq), VectorN::Zero(model_.nv));

    // Get feet frame IDs
    foot_ids_[0] = static_cast<int>(model_.getFrameId("FL_FOOT")); // from long uint to int
    foot_ids_[1] = static_cast<int>(model_.getFrameId("FR_FOOT"));
    foot_ids_[2] = static_cast<int>(model_.getFrameId("HL_FOOT"));
    foot_ids_[3] = static_cast<int>(model_.getFrameId("HR_FOOT"));

}

void InvKin::refreshAndCompute(Matrix14 const& contacts, Matrix43 const& pgoals,
                               Matrix43 const& vgoals, Matrix43 const& agoals) {

    // Process feet
    for (int i = 0; i < 4; i++) {
        pfeet_err.row(i) = pgoals.row(i) - posf_.row(i);
        vfeet_ref.row(i) = vgoals.row(i);

        afeet.row(i) = + params_->Kp_flyingfeet * pfeet_err.row(i) - params_->Kd_flyingfeet * (vf_.row(i)-vgoals.row(i)) + agoals.row(i);
        if (contacts(0, i) == 1.0) {
            afeet.row(i).setZero(); // Set to 0.0 to disable position/velocity control of feet in contact
        }
        afeet.row(i) -= af_.row(i) + (wf_.row(i)).cross(vf_.row(i)); // Drift
    }

    // Store data and invert the Jacobian
    for (int i = 0; i < 4; i++) {
        acc.block(0, 3*i, 1, 3) = afeet.row(i);
        x_err.block(0, 3*i, 1, 3) = pfeet_err.row(i);
        dx_r.block(0, 3*i, 1, 3) = vfeet_ref.row(i);
        invJ.block(3*i, 3*i, 3, 3) = Jf_.block(3*i, 3*i, 3, 3).inverse();
    }

    // Once Jacobian has been inverted we can get command accelerations, velocities and positions
    ddq_cmd_ = invJ * acc.transpose();
    dq_cmd_ = invJ * dx_r.transpose();
    q_step_ = invJ * x_err.transpose(); // Not a position but a step in position

    /*
    std::cout << "J" << std::endl << Jf << std::endl;
    std::cout << "invJ" << std::endl << invJ << std::endl;
    std::cout << "acc" << std::endl << acc << std::endl;
    std::cout << "q_step" << std::endl << q_step << std::endl;
    std::cout << "dq_cmd" << std::endl << dq_cmd << std::endl;
    */
}

void InvKin::run_InvKin(VectorN const& q, VectorN const& dq, MatrixN const& contacts, MatrixN const& pgoals, MatrixN const& vgoals, MatrixN const& agoals)
{
    // Update model and data of the robot
    pinocchio::forwardKinematics(model_, data_, q, dq, VectorN::Zero(model_.nv));
    pinocchio::computeJointJacobians(model_, data_);
    pinocchio::updateFramePlacements(model_, data_);

    // Get data required by IK with Pinocchio
    for (int i = 0; i < 4; i++) {
        int idx = foot_ids_[i];
        posf_.row(i) = data_.oMf[idx].translation();
        pinocchio::Motion nu = pinocchio::getFrameVelocity(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED);
        vf_.row(i) = nu.linear();
        wf_.row(i) = nu.angular();
        af_.row(i) = pinocchio::getFrameAcceleration(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED).linear();
        Jf_tmp_.setZero(); // Fill with 0s because getFrameJacobian only acts on the coeffs it changes so the
        // other coeffs keep their previous value instead of being set to 0 
        pinocchio::getFrameJacobian(model_, data_, idx, pinocchio::LOCAL_WORLD_ALIGNED, Jf_tmp_);
        Jf_.block(3 * i, 0, 3, 12) = Jf_tmp_.block(0, 0, 3, 12);
    }

    // IK output for accelerations of actuators (stored in ddq_cmd_)
    // IK output for velocities of actuators (stored in dq_cmd_)
    refreshAndCompute(contacts, pgoals, vgoals, agoals);

    // IK output for positions of actuators
    q_cmd_ = q + q_step_;

}
