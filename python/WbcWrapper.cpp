#include "qrw/WbcWrapper.hpp"

#include "bindings/python.hpp"

template <typename WbcWrapper>
struct WbcWrapperVisitor : public bp::def_visitor<WbcWrapperVisitor<WbcWrapper>> {
  template <class PyClassWbcWrapper>
  void visit(PyClassWbcWrapper& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("initialize", &WbcWrapper::initialize, bp::args("params"), "Initialize WbcWrapper from Python.\n")

        .def("get_bdes", &WbcWrapper::get_bdes, "Get bdes_.\n")
        .def("get_qdes", &WbcWrapper::get_qdes, "Get qdes_.\n")
        .def("get_vdes", &WbcWrapper::get_vdes, "Get vdes_.\n")
        .def("get_tau_ff", &WbcWrapper::get_tau_ff, "Get tau_ff_.\n")
        .def("get_tasks_acc", &WbcWrapper::get_tasks_acc, "Get tasks acceleration.\n")
        .def("get_tasks_vel", &WbcWrapper::get_tasks_vel, "Get tasks velocity.\n")
        .def("get_tasks_err", &WbcWrapper::get_tasks_err, "Get tasks error.\n")

        .def_readonly("bdes", &WbcWrapper::get_bdes)
        .def_readonly("qdes", &WbcWrapper::get_qdes)
        .def_readonly("vdes", &WbcWrapper::get_vdes)
        .def_readonly("tau_ff", &WbcWrapper::get_tau_ff)
        .def_readonly("ddq_cmd", &WbcWrapper::get_ddq_cmd)
        .def_readonly("dq_cmd", &WbcWrapper::get_dq_cmd)
        .def_readonly("q_cmd", &WbcWrapper::get_q_cmd)
        .def_readonly("f_with_delta", &WbcWrapper::get_f_with_delta)
        .def_readonly("ddq_with_delta", &WbcWrapper::get_ddq_with_delta)
        .def_readonly("nle", &WbcWrapper::get_nle)
        .def_readonly("feet_pos", &WbcWrapper::get_feet_pos)
        .def_readonly("feet_err", &WbcWrapper::get_feet_err)
        .def_readonly("feet_vel", &WbcWrapper::get_feet_vel)
        .def_readonly("feet_pos_target", &WbcWrapper::get_feet_pos_target)
        .def_readonly("feet_vel_target", &WbcWrapper::get_feet_vel_target)
        .def_readonly("feet_acc_target", &WbcWrapper::get_feet_acc_target)
        .def_readonly("Mddq", &WbcWrapper::get_Mddq)
        .def_readonly("NLE", &WbcWrapper::get_NLE)
        .def_readonly("JcTf", &WbcWrapper::get_JcTf)
        .def_readonly("Mddq_out", &WbcWrapper::get_Mddq_out)
        .def_readonly("JcTf_out", &WbcWrapper::get_JcTf_out)

        .def("compute", &WbcWrapper::compute,
             bp::args("q", "dq", "f_cmd", "contacts", "pgoals", "vgoals", "agoals", "xgoals"),
             "Run WbcWrapper from Python.\n");
  }

  static void expose() { bp::class_<WbcWrapper>("WbcWrapper", bp::no_init).def(WbcWrapperVisitor<WbcWrapper>()); }
};
void exposeWbcWrapper() { WbcWrapperVisitor<WbcWrapper>::expose(); }
