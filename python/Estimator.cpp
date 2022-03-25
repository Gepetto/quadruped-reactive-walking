#include "qrw/Estimator.hpp"

#include "bindings/python.hpp"

template <typename Estimator>
struct EstimatorVisitor : public bp::def_visitor<EstimatorVisitor<Estimator>> {
  template <class PyClassEstimator>
  void visit(PyClassEstimator& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("initialize", &Estimator::initialize, bp::args("params"), "Initialize Estimator from Python.\n")
        .def("update_reference_state", &Estimator::updateReferenceState, bp::args("v_ref"), "Update robot state.\n")

        .def("get_q_estimate", &Estimator::getQEstimate, "Get filtered configuration.\n")
        .def("get_v_estimate", &Estimator::getVEstimate, "Get filtered velocity.\n")
        .def("get_v_security", &Estimator::getVSecurity, "Get filtered velocity for security check.\n")
        .def("get_feet_status", &Estimator::getFeetStatus, "")
        .def("get_feet_targets", &Estimator::getFeetTargets, "")
        .def("get_base_velocity_FK", &Estimator::getBaseVelocityFK, "")
        .def("get_base_position_FK", &Estimator::getBasePositionFK, "")
        .def("get_feet_position_barycenter", &Estimator::getFeetPositionBarycenter, "")
        .def("get_b_base_velocity", &Estimator::getBBaseVelocity, "")
        .def("get_filter_vel_X", &Estimator::getFilterVelX, "")
        .def("get_filter_vel_DX", &Estimator::getFilterVelDX, "")
        .def("get_filter_vel_Alpha", &Estimator::getFilterVelAlpha, "")
        .def("get_filter_vel_FiltX", &Estimator::getFilterVelFiltX, "")
        .def("get_filter_pos_X", &Estimator::getFilterPosX, "")
        .def("get_filter_pos_DX", &Estimator::getFilterPosDX, "")
        .def("get_filter_pos_Alpha", &Estimator::getFilterPosAlpha, "")
        .def("get_filter_pos_FiltX", &Estimator::getFilterPosFiltX, "")
        .def("get_q_reference", &Estimator::getQReference, "")
        .def("get_v_reference", &Estimator::getVReference, "")
        .def("get_base_vel_ref", &Estimator::getBaseVelRef, "")
        .def("get_base_acc_ref", &Estimator::getBaseAccRef, "")
        .def("get_h_v", &Estimator::getHV, "")
        .def("get_v_filtered", &Estimator::getVFiltered, "Get filtered velocity.\n")
        .def("get_h_v_filtered", &Estimator::getHVFiltered, "")
        .def("get_oRh", &Estimator::getoRh, "")
        .def("get_hRb", &Estimator::gethRb, "")
        .def("get_oTh", &Estimator::getoTh, "")

        .def("run", &Estimator::run,
             bp::args("gait", "goals", "baseLinearAcceleration", "baseAngularVelocity", "baseOrientation", "q_mes",
                      "v_mes", "base_position", "b_base_velocity"),
             "Run Estimator from Python.\n");
  }

  static void expose() { bp::class_<Estimator>("Estimator", bp::no_init).def(EstimatorVisitor<Estimator>()); }
};

void exposeEstimator() { EstimatorVisitor<Estimator>::expose(); }
