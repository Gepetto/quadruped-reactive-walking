#include "qrw/MPC.hpp"

#include "bindings/python.hpp"

template <typename MPC>
struct MPCVisitor : public bp::def_visitor<MPCVisitor<MPC>> {
  template <class PyClassMPC>
  void visit(PyClassMPC& mpc) const {
    mpc.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def(bp::init<Params&>(bp::args("params"), "Constructor with parameters."))
        .def("run", &MPC::run, bp::args("num_iter", "xref_in", "fsteps_in", "nle"), "Run MPC from Python.\n")
        .def("get_latest_result", &MPC::get_latest_result,
             "Get latest result (predicted trajectory  forces to apply).\n")
        .def("get_gait", &MPC::get_gait, "Get gait matrix.\n")
        .def("get_Sgait", &MPC::get_Sgait, "Get S_gait matrix.\n")
        .def("retrieve_cost", &MPC::retrieve_cost, "retrieve the cost.\n");
  }

  static void expose() {
    bp::class_<MPC>("MPC", bp::no_init).def(MPCVisitor<MPC>());
  }
};

void exposeMPC() { MPCVisitor<MPC>::expose(); }