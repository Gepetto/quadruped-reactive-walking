#ifndef __PYTHON_ADDER__
#define __PYTHON_ADDER__

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <boost/python/scope.hpp>

#undef BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;

void exposeMPC();
void exposeMpcWrapper();
void exposeFilter();
void exposeStatePlanner();
void exposeGait();
void exposeFootstepPlanner();
void exposeFootTrajectoryGenerator();
void exposeInvKin();
void exposeQPWBC();
void exposeWbcWrapper();
void exposeEstimator();
void exposeJoystick();
void exposeParams();
void exposeSurface();
void exposeFootTrajectoryGeneratorBezier();
void exposeFootstepPlannerQP();
void exposeStatePlanner3D();

#endif