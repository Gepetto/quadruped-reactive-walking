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

        .def("get_p_ref", &Joystick::getPRef, "Get Reference Position")
        .def("get_v_ref", &Joystick::getVRef, "Get Reference Velocity")
        .def("get_joystick_code", &Joystick::getJoystickCode, "Get Joystick Code")
        .def("get_start", &Joystick::getStart, "Get Joystick Start")
        .def("get_stop", &Joystick::getStop, "Get Joystick Stop")
        .def("get_cross", &Joystick::getCross, "Get Joystick Cross status")
        .def("get_circle", &Joystick::getCircle, "Get Joystick Circle status")
        .def("get_triangle", &Joystick::getTriangle, "Get Joystick Triangle status")
        .def("get_square", &Joystick::getSquare, "Get Joystick Square status")
        .def("get_l1", &Joystick::getL1, "Get Joystick L1 status")
        .def("get_r1", &Joystick::getR1, "Get Joystick R1 status");
  }

  static void expose() { bp::class_<Joystick>("Joystick", bp::no_init).def(JoystickVisitor<Joystick>()); }
};

void exposeJoystick() { JoystickVisitor<Joystick>::expose(); }