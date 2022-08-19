#include "qrw/Gait.hpp"

#include "bindings/python.hpp"

template <typename Gait>
struct GaitVisitor : public bp::def_visitor<GaitVisitor<Gait>> {
  template <class PyClassGait>
  void visit(PyClassGait& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_past_gait", &Gait::getPastGait, "Get past gait matrix.\n")
        .def("get_gait_matrix", &Gait::getCurrentGait, "Get gait matrix.\n")
        .def("get_gait_coeff", &Gait::getCurrentGaitCoeff, bp::args("i", "j"), "Get gait coefficient.\n")
        .def("get_desired_gait", &Gait::getDesiredGait, "Get desired gait matrix.\n")
        .def("is_new_step", &Gait::isNewPhase, "True if new phase of the gait.\n")
        .def("is_static", &Gait::getIsStatic, "True if static gait.\n")
        .def("get_phase_duration", &Gait::getPhaseDuration, bp::args("i", "j", "value"), "Get phase duration.\n")
        .def("get_remaining_time", &Gait::getRemainingTime, "Get remaining time of the last computed phase.\n")

        .def("initialize", &Gait::initialize, bp::args("params"), "Initialize Gait from Python.\n")

        .def("update", &Gait::update, bp::args("k", "k_mpc", "joystickCode"),
             "Update current gait matrix from Python.\n")

        .def("set_new_phase", &Gait::setNewPhase, bp::args("value"), "Set value of newPhase_ from Python.\n")
        .def("set_late", &Gait::setLate, bp::args("i"), "Set value of isLate_ from Python.\n")

        .add_property("matrix",
                      bp::make_function(&Gait::getCurrentGait, bp::return_value_policy<bp::return_by_value>()));
  }

  static void expose() { bp::class_<Gait>("Gait", bp::no_init).def(GaitVisitor<Gait>()); }
};

void exposeGait() { GaitVisitor<Gait>::expose(); }