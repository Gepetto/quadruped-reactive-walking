#include "quadruped-reactive-walking/gepadd.hpp"
#include "quadruped-reactive-walking/MPC.hpp"
#include "quadruped-reactive-walking/Planner.hpp"
#include "quadruped-reactive-walking/InvKin.hpp"
#include "quadruped-reactive-walking/QPWBC.hpp"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>

namespace bp = boost::python;

template <typename MPC>
struct MPCPythonVisitor : public bp::def_visitor<MPCPythonVisitor<MPC> > {
  template <class PyClassMPC>
  void visit(PyClassMPC& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def(bp::init<double, int, double>(bp::args("dt_in", "n_steps_in", "T_gait_in"),
                                           "Constructor with parameters."))

        // Run MPC from Python
        .def("run", &MPC::run, bp::args("num_iter", "xref_in", "fsteps_in"), "Run MPC from Python.\n")
        .def("get_latest_result", &MPC::get_latest_result,
             "Get latest result (predicted trajectory + forces to apply).\n")
        .def("get_gait", &MPC::get_gait, "Get gait matrix.\n")
        .def("get_Sgait", &MPC::get_Sgait, "Get S_gait matrix.\n");
  }

  static void expose() {
    bp::class_<MPC>("MPC", bp::no_init).def(MPCPythonVisitor<MPC>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
  }
};

void exposeMPC() { MPCPythonVisitor<MPC>::expose(); }

template <typename Planner>
struct PlannerPythonVisitor : public bp::def_visitor<PlannerPythonVisitor<Planner> > {
  template <class PyClassPlanner>
  void visit(PyClassPlanner& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def(bp::init<double, double, double, double, int, bool, double, const Eigen::MatrixXd&>(
            bp::args("dt_in", "dt_tsid_in", "T_gait_in", "T_mpc_in", "k_mpc_in", "on_solo8_in", "h_ref_in",
                     "fsteps_in"),
            "Constructor with parameters."))

        .def("get_xref", &Planner::get_xref, "Get xref matrix.\n")
        .def("get_fsteps", &Planner::get_fsteps, "Get fsteps matrix.\n")
        .def("get_gait", &Planner::get_gait, "Get gait matrix.\n")
        .def("get_goals", &Planner::get_goals, "Get position goals matrix.\n")
        .def("get_vgoals", &Planner::get_vgoals, "Get velocity goals matrix.\n")
        .def("get_agoals", &Planner::get_agoals, "Get acceleration goals matrix.\n")
        //.add_property("xref", &Planner::get_xref)

        // Run Planner from Python
        .def("run_planner", &Planner::run_planner, bp::args("k", "q", "v", "b_vref", "h_estim", "z_average", "joystick_code"),
             "Run Planner from Python.\n");
  }

  static void expose() {
    bp::class_<Planner>("Planner", bp::no_init).def(PlannerPythonVisitor<Planner>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
  }
};

void exposePlanner() { PlannerPythonVisitor<Planner>::expose(); }

template <typename InvKin>
struct InvKinPythonVisitor : public bp::def_visitor<InvKinPythonVisitor<InvKin> > {
  template <class PyClassInvKin>
  void visit(PyClassInvKin& cl) const {
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

  static void expose() {
    bp::class_<InvKin>("InvKin", bp::no_init).def(InvKinPythonVisitor<InvKin>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
  }
};

void exposeInvKin() { InvKinPythonVisitor<InvKin>::expose(); }

template <typename QPWBC>
struct QPWBCPythonVisitor : public bp::def_visitor<QPWBCPythonVisitor<QPWBC> > {
  template <class PyClassQPWBC>
  void visit(PyClassQPWBC& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_f_res", &QPWBC::get_f_res, "Get velocity goals matrix.\n")
        .def("get_ddq_res", &QPWBC::get_ddq_res, "Get acceleration goals matrix.\n")
        .def("get_H", &QPWBC::get_H, "Get H weight matrix.\n")

        // Run QPWBC from Python
        .def("run", &QPWBC::run, bp::args("M", "Jc", "f_cmd", "RNEA", "k_contacts"), "Run QPWBC from Python.\n");
  }

  static void expose() {
    bp::class_<QPWBC>("QPWBC", bp::no_init).def(QPWBCPythonVisitor<QPWBC>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
  }
};

void exposeQPWBC() { QPWBCPythonVisitor<QPWBC>::expose(); }

BOOST_PYTHON_MODULE(libquadruped_reactive_walking) {
  boost::python::def("add", gepetto::example::add);
  boost::python::def("sub", gepetto::example::sub);

  eigenpy::enableEigenPy();

  exposeMPC();
  exposePlanner();
  exposeInvKin();
  exposeQPWBC();
}