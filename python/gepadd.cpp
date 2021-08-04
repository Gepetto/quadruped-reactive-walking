#include "qrw/gepadd.hpp"
#include "qrw/InvKin.hpp"
#include "qrw/MPC.hpp"
#include "qrw/StatePlanner.hpp"
#include "qrw/Gait.hpp"
#include "qrw/FootstepPlanner.hpp"
#include "qrw/FootTrajectoryGenerator.hpp"
#include "qrw/QPWBC.hpp"
#include "qrw/Estimator.hpp"
#include "qrw/Joystick.hpp"
#include "qrw/Params.hpp"

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;

template <typename MPC>
struct MPCPythonVisitor : public bp::def_visitor<MPCPythonVisitor<MPC>>
{
    template <class PyClassMPC>
    void visit(PyClassMPC& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            .def(bp::init<Params&>(bp::args("params"),
                                               "Constructor with parameters."))

            // Run MPC from Python
            .def("run", &MPC::run, bp::args("num_iter", "xref_in", "fsteps_in"), "Run MPC from Python.\n")
            .def("get_latest_result", &MPC::get_latest_result,
                 "Get latest result (predicted trajectory  forces to apply).\n")
            .def("get_gait", &MPC::get_gait, "Get gait matrix.\n")
            .def("get_Sgait", &MPC::get_Sgait, "Get S_gait matrix.\n");
    }

    static void expose()
    {
        bp::class_<MPC>("MPC", bp::no_init).def(MPCPythonVisitor<MPC>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};

void exposeMPC() { MPCPythonVisitor<MPC>::expose(); }

/////////////////////////////////
/// Binding StatePlanner class
/////////////////////////////////
template <typename StatePlanner>
struct StatePlannerPythonVisitor : public bp::def_visitor<StatePlannerPythonVisitor<StatePlanner>>
{
    template <class PyClassStatePlanner>
    void visit(PyClassStatePlanner& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getReferenceStates", &StatePlanner::getReferenceStates, "Get xref matrix.\n")
            .def("getNSteps", &StatePlanner::getNSteps, "Get number of steps in prediction horizon.\n")

            .def("initialize", &StatePlanner::initialize, bp::args("params"),
                 "Initialize StatePlanner from Python.\n")

            // Run StatePlanner from Python
            .def("computeReferenceStates", &StatePlanner::computeReferenceStates, bp::args("q", "v", "b_vref", "z_average"),
                 "Run StatePlanner from Python.\n");
    }

    static void expose()
    {
        bp::class_<StatePlanner>("StatePlanner", bp::no_init).def(StatePlannerPythonVisitor<StatePlanner>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeStatePlanner() { StatePlannerPythonVisitor<StatePlanner>::expose(); }

/////////////////////////////////
/// Binding Gait class
/////////////////////////////////
template <typename Gait>
struct GaitPythonVisitor : public bp::def_visitor<GaitPythonVisitor<Gait>>
{
    template <class PyClassGait>
    void visit(PyClassGait& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getCurrentGait", &Gait::getCurrentGait, "Get currentGait_ matrix.\n")
            .def("isNewPhase", &Gait::isNewPhase, "Get newPhase_ boolean.\n")
            .def("getIsStatic", &Gait::getIsStatic, "Get is_static_ boolean.\n")

            .def("initialize", &Gait::initialize, bp::args("params"),
                 "Initialize Gait from Python.\n")

            // Update current gait matrix from Python
            .def("updateGait", &Gait::updateGait, bp::args("k", "k_mpc", "q", "joystickCode"),
                 "Update current gait matrix from Python.\n")

            // Set current gait matrix from Python
            .def("setGait", &Gait::setGait, bp::args("gaitMatrix"),
                 "Set current gait matrix from Python.\n");
    }

    static void expose()
    {
        bp::class_<Gait>("Gait", bp::no_init).def(GaitPythonVisitor<Gait>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeGait() { GaitPythonVisitor<Gait>::expose(); }

/////////////////////////////////
/// Binding FootstepPlanner class
/////////////////////////////////
template <typename FootstepPlanner>
struct FootstepPlannerPythonVisitor : public bp::def_visitor<FootstepPlannerPythonVisitor<FootstepPlanner>>
{
    template <class PyClassFootstepPlanner>
    void visit(PyClassFootstepPlanner& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getFootsteps", &FootstepPlanner::getFootsteps, "Get footsteps_ matrix.\n")
            .def("getRz", &FootstepPlanner::getRz, "Get rotation along z matrix.\n")

            .def("initialize", &FootstepPlanner::initialize, bp::args("params", "gaitIn"),
                 "Initialize FootstepPlanner from Python.\n")

            // Compute target location of footsteps from Python
            .def("updateFootsteps", &FootstepPlanner::updateFootsteps, bp::args("refresh", "k", "q", "b_v", "b_vref"),
                 "Update and compute location of footsteps from Python.\n");

    }

    static void expose()
    {
        bp::class_<FootstepPlanner>("FootstepPlanner", bp::no_init).def(FootstepPlannerPythonVisitor<FootstepPlanner>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeFootstepPlanner() { FootstepPlannerPythonVisitor<FootstepPlanner>::expose(); }

/////////////////////////////////
/// Binding FootTrajectoryGenerator class
/////////////////////////////////
template <typename FootTrajectoryGenerator>
struct FootTrajectoryGeneratorPythonVisitor : public bp::def_visitor<FootTrajectoryGeneratorPythonVisitor<FootTrajectoryGenerator>>
{
    template <class PyClassFootTrajectoryGenerator>
    void visit(PyClassFootTrajectoryGenerator& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("getFootPositionBaseFrame", &FootTrajectoryGenerator::getFootPositionBaseFrame,
                 bp::args("R", "T"), "Get position_ matrix in base frame.\n")
            .def("getFootVelocityBaseFrame", &FootTrajectoryGenerator::getFootVelocityBaseFrame,
                 bp::args("R", "v_ref", "w_ref"), "Get velocity_ matrix in base frame.\n")
            .def("getFootAccelerationBaseFrame", &FootTrajectoryGenerator::getFootAccelerationBaseFrame,
                 bp::args("R", "w_ref"), "Get acceleration_ matrix in base frame.\n")

            .def("getFootPosition", &FootTrajectoryGenerator::getFootPosition, "Get position_ matrix.\n")
            .def("getFootVelocity", &FootTrajectoryGenerator::getFootVelocity, "Get velocity_ matrix.\n")
            .def("getFootAcceleration", &FootTrajectoryGenerator::getFootAcceleration, "Get acceleration_ matrix.\n")

            .def("initialize", &FootTrajectoryGenerator::initialize, bp::args("params", "gaitIn"),
                 "Initialize FootTrajectoryGenerator from Python.\n")

            // Compute target location of footsteps from Python
            .def("update", &FootTrajectoryGenerator::update, bp::args("k", "targetFootstep"),
                 "Compute target location of footsteps from Python.\n");

    }

    static void expose()
    {
        bp::class_<FootTrajectoryGenerator>("FootTrajectoryGenerator", bp::no_init).def(FootTrajectoryGeneratorPythonVisitor<FootTrajectoryGenerator>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeFootTrajectoryGenerator() { FootTrajectoryGeneratorPythonVisitor<FootTrajectoryGenerator>::expose(); }

/////////////////////////////////
/// Binding InvKin class
/////////////////////////////////
template <typename InvKin>
struct InvKinPythonVisitor : public bp::def_visitor<InvKinPythonVisitor<InvKin>>
{
    template <class PyClassInvKin>
    void visit(PyClassInvKin& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            
            .def("initialize", &InvKin::initialize, bp::args("params"), "Initialize InvKin from Python.\n")

            .def("get_q_step", &InvKin::get_q_step, "Get position step of inverse kinematics.\n")
            .def("get_q_cmd", &InvKin::get_q_cmd, "Get position command.\n")
            .def("get_dq_cmd", &InvKin::get_dq_cmd, "Get velocity command.\n")
            .def("get_ddq_cmd", &InvKin::get_ddq_cmd, "Get acceleration command.\n")
            .def("get_foot_id", &InvKin::get_foot_id, bp::args("i"), "Get food frame id.\n")

            // Run InvKin from Python
            .def("run_InvKin", &InvKin::run_InvKin, bp::args("contacts", "pgoals", "vgoals", "agoals"),
                 "Run InvKin from Python.\n");
    }

    static void expose()
    {
        bp::class_<InvKin>("InvKin", bp::no_init).def(InvKinPythonVisitor<InvKin>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};

void exposeInvKin() { InvKinPythonVisitor<InvKin>::expose(); }

/////////////////////////////////
/// Binding QPWBC class
/////////////////////////////////
template <typename QPWBC>
struct QPWBCPythonVisitor : public bp::def_visitor<QPWBCPythonVisitor<QPWBC>>
{
    template <class PyClassQPWBC>
    void visit(PyClassQPWBC& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("initialize", &QPWBC::initialize, bp::args("params"), "Initialize QPWBC from Python.\n")

            .def("get_f_res", &QPWBC::get_f_res, "Get velocity goals matrix.\n")
            .def("get_ddq_res", &QPWBC::get_ddq_res, "Get acceleration goals matrix.\n")
            .def("get_H", &QPWBC::get_H, "Get H weight matrix.\n")

            // Run QPWBC from Python
            .def("run", &QPWBC::run, bp::args("M", "Jc", "f_cmd", "RNEA", "k_contacts"), "Run QPWBC from Python.\n");
    }

    static void expose()
    {
        bp::class_<QPWBC>("QPWBC", bp::no_init).def(QPWBCPythonVisitor<QPWBC>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};
void exposeQPWBC() { QPWBCPythonVisitor<QPWBC>::expose(); }

/////////////////////////////////
/// Binding WbcWrapper class
/////////////////////////////////
template <typename WbcWrapper>
struct WbcWrapperPythonVisitor : public bp::def_visitor<WbcWrapperPythonVisitor<WbcWrapper>>
{
    template <class PyClassWbcWrapper>
    void visit(PyClassWbcWrapper& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("initialize", &WbcWrapper::initialize, bp::args("params"), "Initialize WbcWrapper from Python.\n")

            .def("get_qdes", &WbcWrapper::get_qdes, "Get qdes_.\n")
            .def("get_vdes", &WbcWrapper::get_vdes, "Get vdes_.\n")
            .def("get_tau_ff", &WbcWrapper::get_tau_ff, "Get tau_ff_.\n")

            .def_readonly("qdes", &WbcWrapper::get_qdes)
            .def_readonly("vdes", &WbcWrapper::get_vdes)
            .def_readonly("tau_ff", &WbcWrapper::get_tau_ff)
            .def_readonly("f_with_delta", &WbcWrapper::get_f_with_delta)
            .def_readonly("feet_pos", &WbcWrapper::get_feet_pos)
            .def_readonly("feet_err", &WbcWrapper::get_feet_err)
            .def_readonly("feet_vel", &WbcWrapper::get_feet_vel)
            .def_readonly("feet_pos_target", &WbcWrapper::get_feet_pos_target)
            .def_readonly("feet_vel_target", &WbcWrapper::get_feet_vel_target)
            .def_readonly("feet_acc_target", &WbcWrapper::get_feet_acc_target)

            // Run WbcWrapper from Python
            .def("compute", &WbcWrapper::compute, bp::args("q", "dq", "f_cmd", "contacts", "pgoals", "vgoals", "agoals"), "Run WbcWrapper from Python.\n");
    }

    static void expose()
    {
        bp::class_<WbcWrapper>("WbcWrapper", bp::no_init).def(WbcWrapperPythonVisitor<WbcWrapper>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};
void exposeWbcWrapper() { WbcWrapperPythonVisitor<WbcWrapper>::expose(); }

/////////////////////////////////
/// Binding Estimator class
/////////////////////////////////
template <typename Estimator>
struct EstimatorPythonVisitor : public bp::def_visitor<EstimatorPythonVisitor<Estimator>>
{
    template <class PyClassEstimator>
    void visit(PyClassEstimator& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("initialize", &Estimator::initialize, bp::args("params"), "Initialize Estimator from Python.\n")
            .def("security_check", &Estimator::security_check, bp::args("tau_ff"), "Run security check.\n")
            .def("updateState", &Estimator::updateState, bp::args("joystick_v_ref", "gait"), "Update robot state.\n")

            .def("getQFilt", &Estimator::getQFilt, "Get filtered configuration.\n")
            .def("getVFilt", &Estimator::getVFilt, "Get filtered velocity.\n")
            .def("getVSecu", &Estimator::getVSecu, "Get filtered velocity for security check.\n")
            .def("getVFiltBis", &Estimator::getVFiltBis, "Get filtered velocity.\n")
            .def("getRPY", &Estimator::getRPY, "Get Roll Pitch Yaw.\n")
            .def("getFeetStatus", &Estimator::getFeetStatus, "")
            .def("getFeetGoals", &Estimator::getFeetGoals, "")
            .def("getFKLinVel", &Estimator::getFKLinVel, "")
            .def("getFKXYZ", &Estimator::getFKXYZ, "")
            .def("getXYZMeanFeet", &Estimator::getXYZMeanFeet, "")
            .def("getFiltLinVel", &Estimator::getFiltLinVel, "")
            .def("getFilterVelX", &Estimator::getFilterVelX, "")
            .def("getFilterVelDX", &Estimator::getFilterVelDX, "")
            .def("getFilterVelAlpha", &Estimator::getFilterVelAlpha, "")
            .def("getFilterVelFiltX", &Estimator::getFilterVelFiltX, "")
            .def("getFilterPosX", &Estimator::getFilterPosX, "")
            .def("getFilterPosDX", &Estimator::getFilterPosDX, "")
            .def("getFilterPosAlpha", &Estimator::getFilterPosAlpha, "")
            .def("getFilterPosFiltX", &Estimator::getFilterPosFiltX, "")
            .def("getQUpdated", &Estimator::getQUpdated, "")
            .def("getVRef", &Estimator::getVRef, "")
            .def("getHV", &Estimator::getHV, "")
            .def("getHVBis", &Estimator::getHVBis, "")
            .def("getoRh", &Estimator::getoRh, "")
            .def("getoTh", &Estimator::getoTh, "")
            .def("getYawEstim", &Estimator::getYawEstim, "")

            // Run Estimator from Python
            .def("run_filter", &Estimator::run_filter, bp::args("gait", "goals", "baseLinearAcceleration",
                                                                "baseAngularVelocity", "baseOrientation", "q_mes", "v_mes",
                                                                "dummyPos", "b_baseVel"), "Run Estimator from Python.\n");
    }

    static void expose()
    {
        bp::class_<Estimator>("Estimator", bp::no_init).def(EstimatorPythonVisitor<Estimator>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};
void exposeEstimator() { EstimatorPythonVisitor<Estimator>::expose(); }

/////////////////////////////////
/// Binding Joystick class
/////////////////////////////////
template <typename Joystick>
struct JoystickPythonVisitor : public bp::def_visitor<JoystickPythonVisitor<Joystick>>
{
    template <class PyClassJoystick>
    void visit(PyClassJoystick& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("handle_v_switch", &Joystick::handle_v_switch, bp::args("k", "k_switch", "v_switch"), "Run security check.\n");
    }

    static void expose()
    {
        bp::class_<Joystick>("Joystick", bp::no_init).def(JoystickPythonVisitor<Joystick>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};
void exposeJoystick() { JoystickPythonVisitor<Joystick>::expose(); }


/////////////////////////////////
/// Binding Params class
/////////////////////////////////
template <typename Params>
struct ParamsPythonVisitor : public bp::def_visitor<ParamsPythonVisitor<Params>>
{
    template <class PyClassParams>
    void visit(PyClassParams& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

            .def("initialize", &Params::initialize, bp::args("file_path"),
                 "Initialize Params from Python.\n")

            // Read Params from Python
            .def_readwrite("interface", &Params::interface)
            .def_readwrite("SIMULATION", &Params::SIMULATION)
            .def_readwrite("LOGGING", &Params::LOGGING)
            .def_readwrite("PLOTTING", &Params::PLOTTING)
            .def_readwrite("dt_wbc", &Params::dt_wbc)
            .def_readwrite("N_gait", &Params::N_gait)
            .def_readwrite("envID", &Params::envID)
            .def_readwrite("velID", &Params::velID)
            .def_readwrite("q_init", &Params::q_init)
            .def_readwrite("dt_mpc", &Params::dt_mpc)
            .def_readwrite("T_gait", &Params::T_gait)
            .def_readwrite("T_mpc", &Params::T_mpc)
            .def_readwrite("N_SIMULATION", &Params::N_SIMULATION)
            .def_readwrite("type_MPC", &Params::type_MPC)
            .def_readwrite("use_flat_plane", &Params::use_flat_plane)
            .def_readwrite("predefined_vel", &Params::predefined_vel)
            .def_readwrite("kf_enabled", &Params::kf_enabled)
            .def_readwrite("Kp_main", &Params::Kp_main)
            .def_readwrite("Kd_main", &Params::Kd_main)
            .def_readwrite("Kff_main", &Params::Kff_main)
            .def_readwrite("enable_pyb_GUI", &Params::enable_pyb_GUI)
            .def_readwrite("enable_multiprocessing", &Params::enable_multiprocessing)
            .def_readwrite("perfect_estimator", &Params::perfect_estimator)
            .def_readwrite("mass", &Params::mass)
            .def_readwrite("I_mat", &Params::I_mat)
            .def_readwrite("h_ref", &Params::h_ref)
            .def_readwrite("shoulders", &Params::shoulders)
            .def_readwrite("lock_time", &Params::lock_time)
            .def_readwrite("footsteps_init", &Params::footsteps_init)
            .def_readwrite("footsteps_under_shoulders", &Params::footsteps_under_shoulders);

    }

    static void expose()
    {
        bp::class_<Params>("Params", bp::no_init).def(ParamsPythonVisitor<Params>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposeParams() { ParamsPythonVisitor<Params>::expose(); }

/////////////////////////////////
/// Exposing classes
/////////////////////////////////
BOOST_PYTHON_MODULE(libquadruped_reactive_walking)
{
    boost::python::def("add", gepetto::example::add);
    boost::python::def("sub", gepetto::example::sub);

    eigenpy::enableEigenPy();

    exposeMPC();
    exposeStatePlanner();
    exposeGait();
    exposeFootstepPlanner();
    exposeFootTrajectoryGenerator();
    exposeInvKin();
    exposeQPWBC();
    exposeWbcWrapper();
    exposeEstimator();
    exposeJoystick();
    exposeParams();
}