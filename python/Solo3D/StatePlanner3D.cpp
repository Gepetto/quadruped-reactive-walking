#include "qrw/Solo3D/StatePlanner3D.hpp"

#include "bindings/python.hpp"

template <typename StatePlanner3D>
struct StatePlanner3DVisitor
    : public bp::def_visitor<StatePlanner3DVisitor<StatePlanner3D>> {
  template <class PyClassStatePlanner3D>
  void visit(PyClassStatePlanner3D& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_reference_states", &StatePlanner3D::getReferenceStates,
             "Get xref matrix.\n")
        .def("get_configurations", &StatePlanner3D::getConfigurations,
             "Get conf vector.\n")

        .def("initialize", &StatePlanner3D::initialize, bp::args("params"),
             "Initialize StatePlanner3D from Python.\n")

        // Run StatePlanner3D from Python
        .def("compute_reference_states",
             &StatePlanner3D::computeReferenceStates,
             bp::args("q", "v", "b_vref"), "Run StatePlanner from Python.\n")
        .def("get_fit", &StatePlanner3D::getFit, "Get the fitted surface.\n")
        .def("update_surface", &StatePlanner3D::updateSurface,
             bp::args("q", "b_vref"),
             "Update the average surface from heightmap and positions.\n");
  }

  static void expose() {
    bp::class_<StatePlanner3D>("StatePlanner3D", bp::no_init)
        .def(StatePlanner3DVisitor<StatePlanner3D>());
  }
};

void exposeStatePlanner3D() { StatePlanner3DVisitor<StatePlanner3D>::expose(); }
