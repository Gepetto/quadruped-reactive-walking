#include "qrw/QPWBC.hpp"

#include "bindings/python.hpp"

template <typename QPWBC>
struct QPWBCVisitor : public bp::def_visitor<QPWBCVisitor<QPWBC>> {
  template <class PyClassQPWBC>
  void visit(PyClassQPWBC& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("initialize", &QPWBC::initialize, bp::args("params"),
             "Initialize QPWBC from Python.\n")

        .def("get_f_res", &QPWBC::get_f_res, "Get velocity goals matrix.\n")
        .def("get_ddq_res", &QPWBC::get_ddq_res,
             "Get acceleration goals matrix.\n")
        .def("get_H", &QPWBC::get_H, "Get H weight matrix.\n")

        .def("run", &QPWBC::run,
             bp::args("M", "Jc", "f_cmd", "RNEA", "k_contacts"),
             "Run QPWBC from Python.\n");
  }

  static void expose() {
    bp::class_<QPWBC>("QPWBC", bp::no_init).def(QPWBCVisitor<QPWBC>());
  }
};

void exposeQPWBC() { QPWBCVisitor<QPWBC>::expose(); }
