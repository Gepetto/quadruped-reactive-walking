#include "qrw/Gait.hpp"

#include "bindings/python.hpp"

template <typename Gait>
struct GaitVisitor : public bp::def_visitor<GaitVisitor<Gait>> {
  double (Gait::*getPhaseDuration0)(int) = &Gait::getPhaseDuration;
  double (Gait::*getPhaseDuration1)(int, int) = &Gait::getPhaseDuration;
  double (Gait::*getElapsedTime0)(int) = &Gait::getElapsedTime;
  double (Gait::*getElapsedTime1)(int, int) = &Gait::getElapsedTime;
  double (Gait::*getRemainingTime0)(int) = &Gait::getRemainingTime;
  double (Gait::*getRemainingTime1)(int, int) = &Gait::getRemainingTime;

  template <class PyClassGait>
  void visit(PyClassGait& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_past_gait", &Gait::getPastGait, "Get past gait matrix.\n")
        .def("get_gait_matrix", &Gait::getCurrentGait, "Get gait matrix.\n")
        .def("get_gait_coeff", &Gait::getCurrentGaitCoeff, bp::args("i", "j"),
             "Get gait coefficient.\n")
        .def("get_desired_gait", &Gait::getDesiredGait,
             "Get desired gait matrix.\n")
        .def("is_new_step", &Gait::isNewPhase,
             "True if new phase of the gait.\n")
        .def("is_static", &Gait::getIsStatic, "True if static gait.\n")
        .def("get_phase_duration", getPhaseDuration0, "Get phase duration.\n")
        .def("get_phase_duration", getPhaseDuration1, "Get phase duration.\n")
        .def("get_elapsed_time", getElapsedTime0,
             "Get elapsed time of the last computed phase.\n")
        .def("get_elapsed_time", getElapsedTime1,
             "Get elapsed time of the last computed phase.\n")
        .def("get_remaining_time", getRemainingTime0,
             "Get remaining time of the last computed phase.\n")
        .def("get_remaining_time", getRemainingTime1,
             "Get remaining time of the last computed phase.\n")

        .def("initialize", &Gait::initialize, bp::args("params"),
             "Initialize Gait from Python.\n")

        .def("update", &Gait::update, bp::args("k", "k_mpc", "joystickCode"),
             "Update current gait matrix from Python.\n")

        .def("set_new_phase", &Gait::setNewPhase, bp::args("value"),
             "Set value of newPhase_ from Python.\n")
        .def("set_late", &Gait::setLate, bp::args("i"),
             "Set value of isLate_ from Python.\n")

        .def("set_current_gait", &Gait::setCurrentGait, bp::args("gaitMatrix"),
             "Set current gait matrix from Python.\n")

        .def("set_past_gait", &Gait::setPastGait, bp::args("gaitMatrix"),
             "Set past gait matrix from Python.\n")

        .def("set_desired_gait", &Gait::setDesiredGait, bp::args("gaitMatrix"),
             "Set desired gait matrix from Python.\n")

        .add_property(
            "matrix",
            bp::make_function(&Gait::getCurrentGait,
                              bp::return_value_policy<bp::return_by_value>()));
  }

  static void expose() {
    bp::class_<Gait>("Gait", bp::no_init).def(GaitVisitor<Gait>());
  }
};

void exposeGait() { GaitVisitor<Gait>::expose(); }
