#include "qrw/StatePlanner.hpp"

#include "bindings/python.hpp"

template <typename StatePlanner>
struct StatePlannerVisitor : public bp::def_visitor<StatePlannerVisitor<StatePlanner>> {
  template <class PyClassStatePlanner>
  void visit(PyClassStatePlanner& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("getReferenceStates", &StatePlanner::getReferenceStates, "Get xref matrix.\n")
        .def("getNumberStates", &StatePlanner::getNumberStates, "Get number of steps in prediction horizon.\n")

        .def("initialize", &StatePlanner::initialize, bp::args("params"), "Initialize StatePlanner from Python.\n")

        // Run StatePlanner from Python
        .def("computeReferenceStates", &StatePlanner::computeReferenceStates, bp::args("q", "v", "b_vref"),
             "Run StatePlanner from Python.\n");
  }

  static void expose() {
    bp::class_<StatePlanner>("StatePlanner", bp::no_init).def(StatePlannerVisitor<StatePlanner>());
  }
};

void exposeStatePlanner() { StatePlannerVisitor<StatePlanner>::expose(); }