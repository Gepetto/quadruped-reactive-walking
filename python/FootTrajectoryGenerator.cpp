#include "qrw/FootTrajectoryGenerator.hpp"

#include "bindings/python.hpp"

template <typename FootTrajectoryGenerator>
struct FootTrajectoryGeneratorVisitor
    : public bp::def_visitor<
          FootTrajectoryGeneratorVisitor<FootTrajectoryGenerator>> {
  template <class PyClassFootTrajectoryGenerator>
  void visit(PyClassFootTrajectoryGenerator& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_target_position", &FootTrajectoryGenerator::getTargetPosition,
             "Get targetFootstep_ matrix.\n")
        .def("get_foot_position", &FootTrajectoryGenerator::getFootPosition,
             "Get position_ matrix.\n")
        .def("get_foot_velocity", &FootTrajectoryGenerator::getFootVelocity,
             "Get velocity_ matrix.\n")
        .def("get_foot_acceleration",
             &FootTrajectoryGenerator::getFootAcceleration,
             "Get acceleration_ matrix.\n")
        .def("get_foot_jerk", &FootTrajectoryGenerator::getFootJerk,
             "Get jerk_ matrix.\n")

        .def("initialize", &FootTrajectoryGenerator::initialize,
             bp::args("params", "gaitIn"),
             "Initialize FootTrajectoryGenerator from Python.\n")

        .def("update", &FootTrajectoryGenerator::update,
             bp::args("k", "targetFootstep"),
             "Compute target location of footsteps from Python.\n")

        .def("get_elapsed_durations", &FootTrajectoryGenerator::getT0s,
             "Get the current timings of the flying feet.\n")
        .def("get_phase_durations", &FootTrajectoryGenerator::getTswing,
             "Get the flying period of the feet.\n")
        .def("get_trajectory_to_target",
             &FootTrajectoryGenerator::getTrajectoryToTarget, bp::args("j"),
             "Get the whole swing trajectory from current position to target "
             "on the ground.\n");
  }

  static void expose() {
    bp::class_<FootTrajectoryGenerator>("FootTrajectoryGenerator", bp::no_init)
        .def(FootTrajectoryGeneratorVisitor<FootTrajectoryGenerator>());
  }
};

void exposeFootTrajectoryGenerator() {
  FootTrajectoryGeneratorVisitor<FootTrajectoryGenerator>::expose();
}
