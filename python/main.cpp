#include "bindings/python.hpp"

BOOST_PYTHON_MODULE(quadruped_reactive_walking) {
  exposeMPC();
  exposeFilter();
  exposeStatePlanner();
  exposeGait();
  exposeFootstepPlanner();
  exposeFootTrajectoryGenerator();
  exposeInvKin();
  exposeQPWBC();
  exposeWbcWrapper();
  exposeEstimator();
  exposeJoystick();
  exposeParams();
  exposeSurface();
  exposeFootTrajectoryGeneratorBezier();
  exposeFootstepPlannerQP();
  exposeStatePlanner3D();
}