set(${PY_NAME}_SOURCES
  main.cpp
  Estimator.cpp
  Filter.cpp
  MPC.cpp
  FootstepPlanner.cpp
  FootTrajectoryGenerator.cpp
  Gait.cpp
  InvKin.cpp
  Joystick.cpp
  Params.cpp
  QPWBC.cpp
  StatePlanner.cpp
  WbcWrapper.cpp
  Solo3D/FootstepPlannerQP.cpp
  Solo3D/FootTrajectoryGeneratorBezier.cpp
  Solo3D/StatePlanner3D.cpp
  Solo3D/Surface.cpp
)

add_library(${PY_NAME} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(${PY_NAME} PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX})
TARGET_LINK_BOOST_PYTHON(${PY_NAME} PRIVATE)
install(TARGETS ${PY_NAME} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
  __init__.py
  )

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()