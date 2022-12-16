#include "qrw/Solo3D/FootstepPlannerQP.hpp"

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "bindings/python.hpp"

template <typename FootstepPlannerQP>
struct FootstepPlannerQPVisitor
    : public bp::def_visitor<FootstepPlannerQPVisitor<FootstepPlannerQP>> {
  template <class PyClassFootstepPlannerQP>
  void visit(PyClassFootstepPlannerQP& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))

        .def("get_footsteps", &FootstepPlannerQP::getFootsteps,
             "Get footsteps_ matrix.\n")
        .def("get_target_footsteps", &FootstepPlannerQP::getTargetFootsteps,
             "Get footsteps_ matrix.\n")

        .def("initialize", &FootstepPlannerQP::initialize,
             bp::args("params", "gaitIn"),
             "Initialize FootstepPlanner from Python.\n")

        // Compute target location of footsteps from Python
        .def("update_footsteps", &FootstepPlannerQP::updateFootsteps,
             bp::args("refresh", "k", "q", "b_v", "b_vref"),
             "Update and compute location of footsteps from Python.\n")
        .def("update_surfaces", &FootstepPlannerQP::updateSurfaces,
             bp::args("potential_surfaces", "selected_surfaces", "status",
                      "iterations"),
             "Update the surfaces from surface planner.\n");
  }

  static void expose() {
    bp::class_<SurfaceVector>("SurfaceVector")
        .def(bp::vector_indexing_suite<SurfaceVector>());
    bp::class_<SurfaceVectorVector>("SurfaceVectorVector")
        .def(bp::vector_indexing_suite<SurfaceVectorVector>());
    bp::class_<FootstepPlannerQP>("FootstepPlannerQP", bp::no_init)
        .def(FootstepPlannerQPVisitor<FootstepPlannerQP>());
  }
};

void exposeFootstepPlannerQP() {
  FootstepPlannerQPVisitor<FootstepPlannerQP>::expose();
}
