#include "mpc-wbc-cpp/gepadd.hpp"
#include "mpc-wbc-cpp/MPC.hpp"
#include "mpc-wbc-cpp/Planner.hpp"

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
        .def("run_planner", &Planner::run_planner, bp::args("k", "q", "v", "b_vref", "h_estim", "z_average"),
             "Run Planner from Python.\n");
  }

  static void expose() {
    bp::class_<Planner>("Planner", bp::no_init).def(PlannerPythonVisitor<Planner>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
  }
};

void exposePlanner() { PlannerPythonVisitor<Planner>::expose(); }

BOOST_PYTHON_MODULE(libmpc_wbc_cpp) {
  boost::python::def("add", gepetto::example::add);
  boost::python::def("sub", gepetto::example::sub);

  eigenpy::enableEigenPy();

  exposeMPC();
  exposePlanner();
}