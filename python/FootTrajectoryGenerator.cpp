#include "qrw/FootTrajectoryGenerator.hpp"

#include "bindings/python.hpp"

template <typename FootTrajectoryGenerator>
struct FootTrajectoryGeneratorVisitor
    : public bp::def_visitor<FootTrajectoryGeneratorVisitor<FootTrajectoryGenerator>> {
  template <class PyClassFootTrajectoryGenerator>
  void visit(PyClassFootTrajectoryGenerator& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("getFootPositionBaseFrame", &FootTrajectoryGenerator::getFootPositionBaseFrame, bp::args("R", "T"),
             "Get position_ matrix in base frame.\n")
        .def("getFootVelocityBaseFrame", &FootTrajectoryGenerator::getFootVelocityBaseFrame,
             bp::args("R", "v_ref", "w_ref"), "Get velocity_ matrix in base frame.\n")
        .def("getFootAccelerationBaseFrame", &FootTrajectoryGenerator::getFootAccelerationBaseFrame,
             bp::args("R", "w_ref", "a_ref"), "Get acceleration_ matrix in base frame.\n")

        .def("getTargetPosition", &FootTrajectoryGenerator::getTargetPosition, "Get targetFootstep_ matrix.\n")
        .def("getFootPosition", &FootTrajectoryGenerator::getFootPosition, "Get position_ matrix.\n")
        .def("getFootVelocity", &FootTrajectoryGenerator::getFootVelocity, "Get velocity_ matrix.\n")
        .def("getFootAcceleration", &FootTrajectoryGenerator::getFootAcceleration, "Get acceleration_ matrix.\n")
        .def("getFootJerk", &FootTrajectoryGenerator::getFootJerk, "Get jerk_ matrix.\n")

        .def("initialize", &FootTrajectoryGenerator::initialize, bp::args("params", "gaitIn"),
             "Initialize FootTrajectoryGenerator from Python.\n")

        .def("update", &FootTrajectoryGenerator::update, bp::args("k", "targetFootstep"),
             "Compute target location of footsteps from Python.\n")

        .def("getT0s", &FootTrajectoryGenerator::getT0s, "Get the current timings of the flying feet.\n")
        .def("getTswing", &FootTrajectoryGenerator::getTswing, "Get the flying period of the feet.\n");
  }

  static void expose() {
    bp::class_<FootTrajectoryGenerator>("FootTrajectoryGenerator", bp::no_init)
        .def(FootTrajectoryGeneratorVisitor<FootTrajectoryGenerator>());
  }
};

void exposeFootTrajectoryGenerator() { FootTrajectoryGeneratorVisitor<FootTrajectoryGenerator>::expose(); }
