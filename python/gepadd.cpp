#include "qrw/gepadd.hpp"
#include "qrw/InvKin.hpp"
#include "qrw/MPC.hpp"
#include "qrw/Planner.hpp"
#include "qrw/StatePlanner.hpp"
#include "qrw/Gait.hpp"
#include "qrw/QPWBC.hpp"

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
            .def(bp::init<double, int, double>(bp::args("dt_in", "n_steps_in", "T_gait_in"),
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
/// Binding Planner class
/////////////////////////////////
template <typename Planner>
struct PlannerPythonVisitor : public bp::def_visitor<PlannerPythonVisitor<Planner>>
{
    template <class PyClassPlanner>
    void visit(PyClassPlanner& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            .def(bp::init<double, double, double, double, int, double, const MatrixN&, const MatrixN&>(
                bp::args("dt_in", "dt_tsid_in", "T_gait_in", "T_mpc_in", "k_mpc_in", "h_ref_in",
                         "fsteps_in", "shoulders positions"),
                "Constructor with parameters."))

            .def("get_fsteps", &Planner::get_fsteps, "Get fsteps matrix.\n")
            .def("get_gait", &Planner::get_gait, "Get gait matrix.\n")
            .def("get_goals", &Planner::get_goals, "Get position goals matrix.\n")
            .def("get_vgoals", &Planner::get_vgoals, "Get velocity goals matrix.\n")
            .def("get_agoals", &Planner::get_agoals, "Get acceleration goals matrix.\n")

            // Run Planner from Python
            .def("run_planner", &Planner::run_planner, bp::args("k", "q", "v", "b_vref"),
                 "Run Planner from Python.\n")

            // Update gait matrix from Python
            .def("updateGait", &Planner::updateGait, bp::args("k", "k_mpc", "q", "joystickCode"),
                 "Update gait matrix from Python.\n")

            // Set gait matrix from Python
            .def("setGait", &Planner::setGait, bp::args("gaitMatrix"),
                 "Set gait matrix from Python.\n");
    }

    static void expose()
    {
        bp::class_<Planner>("Planner", bp::no_init).def(PlannerPythonVisitor<Planner>());

        ENABLE_SPECIFIC_MATRIX_TYPE(MatrixN);
    }
};
void exposePlanner() { PlannerPythonVisitor<Planner>::expose(); }

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

            .def("getXReference", &StatePlanner::getXReference, "Get xref matrix.\n")

            .def("initialize", &StatePlanner::initialize, bp::args("dt_in", "T_mpc_in", "h_ref_in"),
                 "Initialize StatePlanner from Python.\n")

            // Run StatePlanner from Python
            .def("computeRefStates", &StatePlanner::computeRefStates, bp::args("q", "v", "b_vref", "z_average"),
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

            .def("initialize", &Gait::initialize, bp::args("dt_in", "T_gait_in", "T_mpc_in"),
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
/// Binding InvKin class
/////////////////////////////////
template <typename InvKin>
struct InvKinPythonVisitor : public bp::def_visitor<InvKinPythonVisitor<InvKin>>
{
    template <class PyClassInvKin>
    void visit(PyClassInvKin& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))
            .def(bp::init<double>(bp::args("dt_in"), "Constructor with parameters."))

            .def("get_q_step", &InvKin::get_q_step, "Get velocity goals matrix.\n")
            .def("get_dq_cmd", &InvKin::get_dq_cmd, "Get acceleration goals matrix.\n")

            // Run InvKin from Python
            .def("refreshAndCompute", &InvKin::refreshAndCompute,
                 bp::args("x_cmd", "contacts", "goals", "vgoals", "agoals", "posf", "vf", "wf", "af", "Jf",
                          "posb", "rotb", "vb", "ab", "Jb"),
                 "Run InvKin from Python.\n");
    }

    static void expose()
    {
        bp::class_<InvKin>("InvKin", bp::no_init).def(InvKinPythonVisitor<InvKin>());

        ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    }
};

void exposeInvKin() { InvKinPythonVisitor<InvKin>::expose(); }

template <typename QPWBC>
struct QPWBCPythonVisitor : public bp::def_visitor<QPWBCPythonVisitor<QPWBC>>
{
    template <class PyClassQPWBC>
    void visit(PyClassQPWBC& cl) const
    {
        cl.def(bp::init<>(bp::arg(""), "Default constructor."))

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

BOOST_PYTHON_MODULE(libquadruped_reactive_walking)
{
    boost::python::def("add", gepetto::example::add);
    boost::python::def("sub", gepetto::example::sub);

    eigenpy::enableEigenPy();

    exposeMPC();
    exposePlanner();
    exposeStatePlanner();
    exposeGait();
    exposeInvKin();
    exposeQPWBC();
}