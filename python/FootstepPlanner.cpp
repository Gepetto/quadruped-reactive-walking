#include "qrw/FootstepPlanner.hpp"

#include "bindings/python.hpp"

template <typename FootstepPlanner>
struct FootstepPlannerVisitor : public bp::def_visitor<FootstepPlannerVisitor<FootstepPlanner>> {
  template <class PyClassFootstepPlanner>
  void visit(PyClassFootstepPlanner& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_footsteps", &FootstepPlanner::getFootsteps, "Get footsteps_ matrix.\n")
        .def("get_target_footsteps", &FootstepPlanner::getTargetFootsteps, "Get footsteps_ matrix.\n")

        .def("initialize", &FootstepPlanner::initialize, bp::args("params", "gaitIn"),
             "Initialize FootstepPlanner from Python.\n")

        // Compute target location of footsteps from Python
        .def("update_footsteps", &FootstepPlanner::updateFootsteps, bp::args("refresh", "k", "q", "b_v", "b_vref"),
             "Update and compute location of footsteps from Python.\n");
  }

  static void expose() {
    bp::class_<FootstepPlanner>("FootstepPlanner", bp::no_init).def(FootstepPlannerVisitor<FootstepPlanner>());
  }
};

void exposeFootstepPlanner() { FootstepPlannerVisitor<FootstepPlanner>::expose(); }