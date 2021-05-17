set(${PY_NAME}_SOURCES
  module.cpp
  )

add_library(${PY_NAME} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
set_target_properties(${PY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PY_NAME})
TARGET_LINK_BOOST_PYTHON(${PY_NAME} PUBLIC)
install(TARGETS ${PY_NAME} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
  Controller.py
  Estimator.py
  FootTrajectoryGenerator.py
  ForceMonitor.py
  gamepadClient.py
  __init__.py
  Joystick.py
  LoggerControl.py
  Logger.py
  LoggerSensors.py
  main_mcapi_solo8.py
  main_minimal_controler.py
  main_solo12_control.py
  main_solo12_demo_estimator.py
  main_solo12_replay.py
  MPC_Wrapper.py
  Planner.py
  plot_comparison_fb.py
  plot_IMU_mocap_result_bis.py
  plot_IMU_mocap_result.py
  PyBulletSimulator.py
  QP_WBC.py
  solo12InvKin.py
  utils_mpc.py
  )

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()
