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
  Controller.py
  main_solo12_control.py
  main_solo12_replay.py
  MPC_Wrapper.py
  )

set(${PY_NAME}_CROCODDYL_PYTHON
  __init__.py
  MPC_crocoddyl.py
  MPC_crocoddyl_2.py
  MPC_crocoddyl_planner.py
  )

set(${PY_NAME}_SOLO3D_PYTHON
  __init__.py
  heightmap_generator.py
  heightmap_tools.py
  pyb_environment_3D.py
  SurfacePlanner.py
  SurfacePlannerWrapper.py
  utils.py
  )

set(${PY_NAME}_TOOLS_PYTHON
  ForceMonitor.py
  gamepadClient.py
  __init__.py
  LoggerControl.py
  LoggerSensors.py
  place_com_with_invkin.py
  PyBulletSimulator.py
  qualisysClient.py
  utils_mpc.py
  )

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()

foreach(python ${${PY_NAME}_CROCODDYL_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/Crocoddyl ${python})
endforeach()

foreach(python ${${PY_NAME}_SOLO3D_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/solo3D ${python})
endforeach()

foreach(python ${${PY_NAME}_TOOLS_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/tools ${python})
endforeach()
