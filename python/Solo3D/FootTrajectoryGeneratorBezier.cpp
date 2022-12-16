#include "qrw/Solo3D/FootTrajectoryGeneratorBezier.hpp"

#include "bindings/python.hpp"

template <typename FootTrajectoryGeneratorBezier>
struct FootTrajectoryGeneratorBezierVisitor
    : public bp::def_visitor<
          FootTrajectoryGeneratorBezierVisitor<FootTrajectoryGeneratorBezier>> {
  template <class PyClassFootTrajectoryGeneratorBezier>
  void visit(PyClassFootTrajectoryGeneratorBezier& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_foot_position",
             &FootTrajectoryGeneratorBezier::getFootPosition,
             "Get position_ matrix.\n")
        .def("get_foot_velocity",
             &FootTrajectoryGeneratorBezier::getFootVelocity,
             "Get velocity_ matrix.\n")
        .def("get_foot_acceleration",
             &FootTrajectoryGeneratorBezier::getFootAcceleration,
             "Get acceleration_ matrix.\n")
        .def("get_foot_jerk", &FootTrajectoryGeneratorBezier::getFootJerk,
             "Get jerk_ matrix.\n")
        .def("evaluate_bezier", &FootTrajectoryGeneratorBezier::evaluateBezier,
             "Evaluate Bezier curve by foot.\n")
        .def("evaluate_polynom", &FootTrajectoryGeneratorBezier::evaluatePoly,
             "Evaluate Bezier curve by foot.\n")

        .def("initialize", &FootTrajectoryGeneratorBezier::initialize,
             bp::args("params", "gaitIn"),
             "Initialize FootTrajectoryGeneratorBezier from Python.\n")

        .add_property(
            "t0s",
            bp::make_function(&FootTrajectoryGeneratorBezier::get_t0s,
                              bp::return_value_policy<bp::return_by_value>()))
        .add_property(
            "t_swing",
            bp::make_function(&FootTrajectoryGeneratorBezier::get_t_swing,
                              bp::return_value_policy<bp::return_by_value>()))

        .def("update", &FootTrajectoryGeneratorBezier::update,
             bp::args("k", "targetFootstep", "surfaces", "q"),
             "Compute target location of footsteps from Python.\n");
  }

  static void expose() {
    bp::class_<FootTrajectoryGeneratorBezier>("FootTrajectoryGeneratorBezier",
                                              bp::no_init)
        .def(FootTrajectoryGeneratorBezierVisitor<
             FootTrajectoryGeneratorBezier>());
  }
};

void exposeFootTrajectoryGeneratorBezier() {
  FootTrajectoryGeneratorBezierVisitor<FootTrajectoryGeneratorBezier>::expose();
}
