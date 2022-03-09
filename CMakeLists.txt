cmake_minimum_required(VERSION 3.14)

# Project properties
set(PROJECT_NAMESPACE gepetto)
set(PROJECT_NAME quadruped-reactive-walking)
set(PROJECT_DESCRIPTION "Reactive walking control architecture for quadrupeds")
set(PROJECT_URL https://gitlab.laas.fr/${PROJECT_NAMESPACE}/${PROJECT_NAME})

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(INSTALL_PYTHON_INTERFACE_ONLY "Install *ONLY* the python bindings" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)

# Project configuration
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  set(PROJECT_USE_CMAKE_EXPORT TRUE)
endif()

# JRL-cmakemodule setup
include(cmake/base.cmake)
include(cmake/boost.cmake)
include(cmake/python.cmake)

# Disable -Werror on Unix.
SET(CXX_DISABLE_WERROR True)

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
CHECK_MINIMAL_CXX_STANDARD(11 REQUIRED)

# Project dependencies
set(BOOST_COMPONENTS unit_test_framework)
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED)
ADD_PROJECT_DEPENDENCY(example-robot-data REQUIRED)

if(BUILD_PYTHON_INTERFACE)
  FINDPYTHON()
  ADD_PROJECT_DEPENDENCY(eigenpy REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
  set(BOOST_COMPONENTS ${BOOST_COMPONENTS} python)
endif()

SEARCH_FOR_BOOST()

# Main Library
set(${PROJECT_NAME}_HEADERS
  include/bindings/python.hpp
  include/qrw/Heightmap.hpp
  include/qrw/MPC.hpp
  include/qrw/Gait.hpp
  include/qrw/FootTrajectoryGenerator.hpp
  include/qrw/FootTrajectoryGeneratorBezier.hpp
  include/qrw/FootstepPlanner.hpp
  include/qrw/FootstepPlannerQP.hpp
  include/qrw/StatePlanner.hpp
  include/qrw/StatePlanner3D.hpp
  include/qrw/Types.h
  include/qrw/InvKin.hpp
  include/qrw/QPWBC.hpp
  include/qrw/WbcWrapper.hpp
  include/qrw/Params.hpp
  include/qrw/Estimator.hpp
  include/qrw/Joystick.hpp
  include/qrw/Filter.hpp
  include/qrw/Controller.hpp
  include/qrw/MpcWrapper.hpp
  include/qrw/FakeRobot.hpp
  include/other/st_to_cc.hpp
  include/qrw/FootTrajectoryGeneratorBezier.hpp
  include/qrw/FootstepPlannerQP.hpp
  include/qrw/Surface.hpp
  )

set(${PROJECT_NAME}_SOURCES
  src/st_to_cc.cpp
  src/Heightmap.cpp
  src/MPC.cpp
  src/Gait.cpp
  src/FootTrajectoryGenerator.cpp
  src/FootTrajectoryGeneratorBezier.cpp
  src/FootstepPlanner.cpp
  src/FootstepPlannerQP.cpp
  src/StatePlanner.cpp
  src/StatePlanner3D.cpp
  src/InvKin.cpp
  src/QPWBC.cpp
  src/WbcWrapper.cpp
  src/Params.cpp
  src/Estimator.cpp
  src/Joystick.cpp
  src/Filter.cpp
  src/Controller.cpp
  src/MpcWrapper.cpp
  src/FootTrajectoryGeneratorBezier.cpp
  src/FootstepPlannerQP.cpp
  src/Surface.cpp
  )

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include/qrw>)
target_link_libraries(${PROJECT_NAME} PUBLIC example-robot-data::example-robot-data)

# Include Eigen3 directories
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})

# Find pinocchio library
add_project_dependency(pinocchio REQUIRED)

# Link pinocchio library
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)

# Link odri_control_interface library
find_package(odri_control_interface REQUIRED)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC odri_control_interface::odri_control_interface)

# Link master board library
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC master_board_sdk::master_board_sdk)

# Find parametric curves library and headers
add_project_dependency(ndcurves REQUIRED)

# Link parametric curves library
target_link_libraries(${PROJECT_NAME} PUBLIC ndcurves::ndcurves)

# Find eiquadprog library and headers
add_project_dependency(eiquadprog REQUIRED)

# Link eiquadprog library
target_link_libraries(${PROJECT_NAME} PUBLIC eiquadprog::eiquadprog)

# Find OSQP library and headers
find_package(osqp REQUIRED)

# Link the OSQP shared library
target_link_libraries(${PROJECT_NAME} PUBLIC osqp::osqp)

#find_package(yaml-cpp REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp::yaml-cpp)
add_project_dependency(yaml-cpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${YAML_CPP_LIBRARIES})

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# target_compile_options(${PROJECT_NAME} PUBLIC PRINTING)
# target_compile_definitions(${PROJECT_NAME} PUBLIC PRINTING)

target_compile_options(${PROJECT_NAME} PUBLIC -DNDEBUG -O3)

target_compile_definitions(${PROJECT_NAME} PUBLIC CONFIG_SOLO12_YAML="${PROJECT_SOURCE_DIR}/scripts/config_solo12.yaml")
target_compile_definitions(${PROJECT_NAME} PUBLIC WALK_PARAMETERS_YAML="${PROJECT_SOURCE_DIR}/src/walk_parameters.yaml")


# MQTT Interface
add_subdirectory (paho.mqtt.c)
add_library(mqtt-interface SHARED include/qrw/mqtt-interface.hpp src/mqtt-interface.cpp)
target_link_libraries(mqtt-interface PRIVATE paho-mqtt3c paho-mqtt3a)
target_include_directories(mqtt-interface PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

# Main Executable
add_executable(${PROJECT_NAMESPACE}-${PROJECT_NAME} src/control_solo12.cpp)
target_link_libraries(${PROJECT_NAMESPACE}-${PROJECT_NAME} ${PROJECT_NAME} mqtt-interface)

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# Unit tests
add_subdirectory(tests)

# Installation
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
  install(TARGETS ${PROJECT_NAMESPACE}-${PROJECT_NAME} RUNTIME DESTINATION bin)
endif()
