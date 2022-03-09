#include "qrw/Joystick.hpp"

#include "bindings/python.hpp"

template <typename Joystick>
struct JoystickVisitor : public bp::def_visitor<JoystickVisitor<Joystick>> {
  template <class PyClassJoystick>
  void visit(PyClassJoystick& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("initialize", &Joystick::initialize, bp::args("params"), "Initialize Joystick from Python.\n")

        .def("update_v_ref", &Joystick::update_v_ref, bp::args("k", "velID", "gait_is_static", "h_v"),
             "Update joystick values.")
        .def("getPRef", &Joystick::getPRef, "Get Reference Position")
        .def("getVRef", &Joystick::getVRef, "Get Reference Velocity")
        .def("getJoystickCode", &Joystick::getJoystickCode, "Get Joystick Code")
        .def("getStart", &Joystick::getStart, "Get Joystick Start")
        .def("getStop", &Joystick::getStop, "Get Joystick Stop")
        .def("getCross", &Joystick::getCross, "Get Joystick Cross status")
        .def("getCircle", &Joystick::getCircle, "Get Joystick Circle status")
        .def("getTriangle", &Joystick::getTriangle, "Get Joystick Triangle status")
        .def("getSquare", &Joystick::getSquare, "Get Joystick Square status")
        .def("getL1", &Joystick::getL1, "Get Joystick L1 status")
        .def("getR1", &Joystick::getR1, "Get Joystick R1 status");
  }

  static void expose() { bp::class_<Joystick>("Joystick", bp::no_init).def(JoystickVisitor<Joystick>()); }
};

void exposeJoystick() { JoystickVisitor<Joystick>::expose(); }