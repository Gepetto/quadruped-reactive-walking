#include "qrw/StatePlanner3D.hpp"

#include "bindings/python.hpp"

template <typename StatePlanner3D>
struct StatePlanner3DVisitor : public bp::def_visitor<StatePlanner3DVisitor<StatePlanner3D>> {
  template <class PyClassStatePlanner3D>
  void visit(PyClassStatePlanner3D& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("getReferenceStates", &StatePlanner3D::getReferenceStates, "Get xref matrix.\n")
        .def("getNumberStates", &StatePlanner3D::getNumberStates, "Get number of steps in prediction horizon.\n")
        .def("getConfigurations", &StatePlanner3D::getConfigurations, "Get conf vector.\n")

        .def("initialize", &StatePlanner3D::initialize, bp::args("params"), "Initialize StatePlanner3D from Python.\n")

        // Run StatePlanner3D from Python
        .def("computeReferenceStates", &StatePlanner3D::computeReferenceStates, bp::args("q", "v", "b_vref"),
             "Run StatePlanner from Python.\n")
        .def("getFit", &StatePlanner3D::getFit, "Get the fitted surface.\n")
        .def("updateSurface", &StatePlanner3D::updateSurface, bp::args("q", "b_vref"),
             "Update the average surface from heightmap and positions.\n");
  }

  static void expose() {
    bp::class_<StatePlanner3D>("StatePlanner3D", bp::no_init).def(StatePlanner3DVisitor<StatePlanner3D>());
  }
};

void exposeStatePlanner3D() { StatePlanner3DVisitor<StatePlanner3D>::expose(); }