#include "qrw/MpcWrapper.hpp"

#include "bindings/python.hpp"

template <typename MpcWrapper>
struct MpcWrapperVisitor : public bp::def_visitor<MpcWrapperVisitor<MpcWrapper>> {
  template <class PyClassMpcWrapper>
  void visit(PyClassMpcWrapper& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def("initialize", &MpcWrapper::initialize, bp::args("params"), "Initialize Gait from Python.\n")
        .def("solve", &MpcWrapper::solve, bp::args("k", "xref_in", "fsteps_in", "gait_in"),
             "Run MpcWrapper from Python.\n")
        .def("get_latest_result", &MpcWrapper::get_latest_result,
             "Get latest result (predicted trajectory  forces to apply).\n")
        .def_readonly("t_mpc_solving_duration", &MpcWrapper::get_t_mpc_solving_duration,
             "Solving time of latest MPC iteration.\n");
  }

  static void expose() { bp::class_<MpcWrapper>("MpcWrapper", bp::no_init).def(MpcWrapperVisitor<MpcWrapper>()); }
};

void exposeMpcWrapper() { MpcWrapperVisitor<MpcWrapper>::expose(); }
