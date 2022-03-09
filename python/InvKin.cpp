#include "qrw/InvKin.hpp"

#include "bindings/python.hpp"

template <typename InvKin>
struct InvKinVisitor : public bp::def_visitor<InvKinVisitor<InvKin>> {
  template <class PyClassInvKin>
  void visit(PyClassInvKin& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("initialize", &InvKin::initialize, bp::args("params"), "Initialize InvKin from Python.\n")

        .def("get_q_step", &InvKin::get_q_step, "Get position step of inverse kinematics.\n")
        .def("get_q_cmd", &InvKin::get_q_cmd, "Get position command.\n")
        .def("get_dq_cmd", &InvKin::get_dq_cmd, "Get velocity command.\n")
        .def("get_ddq_cmd", &InvKin::get_ddq_cmd, "Get acceleration command.\n")
        .def("get_foot_id", &InvKin::get_foot_id, bp::args("i"), "Get food frame id.\n")

        // Run InvKin from Python
        .def("run_InvKin", &InvKin::run_InvKin, bp::args("contacts", "pgoals", "vgoals", "agoals", "x_cmd"),
             "Run InvKin from Python.\n");
  }

  static void expose() {
    bp::class_<InvKin>("InvKin", bp::no_init).def(InvKinVisitor<InvKin>());
  }
};

void exposeInvKin() { InvKinVisitor<InvKin>::expose(); }
