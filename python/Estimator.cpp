#include "qrw/Estimator.hpp"

#include "bindings/python.hpp"

template <typename Estimator>
struct EstimatorVisitor : public bp::def_visitor<EstimatorVisitor<Estimator>> {
  template <class PyClassEstimator>
  void visit(PyClassEstimator& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("initialize", &Estimator::initialize, bp::args("params"), "Initialize Estimator from Python.\n")
        .def("security_check", &Estimator::security_check, bp::args("tau_ff"), "Run security check.\n")
        .def("updateState", &Estimator::updateState, bp::args("joystick_v_ref", "gait"), "Update robot state.\n")

        .def("getQFilt", &Estimator::getQFilt, "Get filtered configuration.\n")
        .def("getVFilt", &Estimator::getVFilt, "Get filtered velocity.\n")
        .def("getVSecu", &Estimator::getVSecu, "Get filtered velocity for security check.\n")
        .def("getVFiltBis", &Estimator::getVFiltBis, "Get filtered velocity.\n")
        .def("getRPY", &Estimator::getRPY, "Get Roll Pitch Yaw.\n")
        .def("getFeetStatus", &Estimator::getFeetStatus, "")
        .def("getFeetGoals", &Estimator::getFeetGoals, "")
        .def("getFKLinVel", &Estimator::getFKLinVel, "")
        .def("getFKXYZ", &Estimator::getFKXYZ, "")
        .def("getXYZMeanFeet", &Estimator::getXYZMeanFeet, "")
        .def("getFiltLinVel", &Estimator::getFiltLinVel, "")
        .def("getFilterVelX", &Estimator::getFilterVelX, "")
        .def("getFilterVelDX", &Estimator::getFilterVelDX, "")
        .def("getFilterVelAlpha", &Estimator::getFilterVelAlpha, "")
        .def("getFilterVelFiltX", &Estimator::getFilterVelFiltX, "")
        .def("getFilterPosX", &Estimator::getFilterPosX, "")
        .def("getFilterPosDX", &Estimator::getFilterPosDX, "")
        .def("getFilterPosAlpha", &Estimator::getFilterPosAlpha, "")
        .def("getFilterPosFiltX", &Estimator::getFilterPosFiltX, "")
        .def("getQUpdated", &Estimator::getQUpdated, "")
        .def("getVUpdated", &Estimator::getVUpdated, "")
        .def("getVRef", &Estimator::getVRef, "")
        .def("getARef", &Estimator::getARef, "")
        .def("getHV", &Estimator::getHV, "")
        .def("getHVWindowed", &Estimator::getHVWindowed, "")
        .def("getoRb", &Estimator::getoRb, "")
        .def("getoRh", &Estimator::getoRh, "")
        .def("gethRb", &Estimator::gethRb, "")
        .def("getoTh", &Estimator::getoTh, "")
        .def("getYawEstim", &Estimator::getYawEstim, "")

        .def("run_filter", &Estimator::run_filter,
             bp::args("gait", "goals", "baseLinearAcceleration", "baseAngularVelocity", "baseOrientation", "q_mes",
                      "v_mes", "dummyPos", "b_baseVel"),
             "Run Estimator from Python.\n");
  }

  static void expose() { bp::class_<Estimator>("Estimator", bp::no_init).def(EstimatorVisitor<Estimator>()); }
};

void exposeEstimator() { EstimatorVisitor<Estimator>::expose(); }
