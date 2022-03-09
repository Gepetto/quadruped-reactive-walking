#include "qrw/Gait.hpp"

#include "bindings/python.hpp"

template <typename Gait>
struct GaitVisitor : public bp::def_visitor<GaitVisitor<Gait>> {
  template <class PyClassGait>
  void visit(PyClassGait& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("getPastGait", &Gait::getPastGait, "Get currentGait_ matrix.\n")
        .def("getCurrentGait", &Gait::getCurrentGait, "Get currentGait_ matrix.\n")
        .def("getDesiredGait", &Gait::getDesiredGait, "Get currentGait_ matrix.\n")
        .def("isNewPhase", &Gait::isNewPhase, "Get newPhase_ boolean.\n")
        .def("getIsStatic", &Gait::getIsStatic, "Get is_static_ boolean.\n")

        .def("initialize", &Gait::initialize, bp::args("params"), "Initialize Gait from Python.\n")

        // Update current gait matrix from Python
        .def("updateGait", &Gait::updateGait, bp::args("k", "k_mpc", "joystickCode"),
             "Update current gait matrix from Python.\n")

        // Set current gait matrix from Python
        .def("setGait", &Gait::setGait, bp::args("gaitMatrix"), "Set current gait matrix from Python.\n");
  }

  static void expose() { bp::class_<Gait>("Gait", bp::no_init).def(GaitVisitor<Gait>()); }
};

void exposeGait() { GaitVisitor<Gait>::expose(); }