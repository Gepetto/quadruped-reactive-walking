cmake_minimum_required(VERSION 3.14)

# Project properties
set(PROJECT_NAMESPACE gepetto)
set(PROJECT_NAME quadruped-reactive-walking)
set(PROJECT_DESCRIPTION "Reactive walking control architecture for quadrupeds")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(INSTALL_PYTHON_INTERFACE_ONLY "Install *ONLY* the python bindings" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(BUILD_WITH_MQTT_SUPPORT "Add support for MQTT interface" OFF)

# Project configuration
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  set(PROJECT_USE_CMAKE_EXPORT TRUE)
endif()

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  message(STATUS "JRL cmakemodules not found. Let's fetch it.")
  include(FetchContent)
  FetchContent_Declare(
    "jrl-cmakemodules"
    GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
  FetchContent_MakeAvailable("jrl-cmakemodules")
  FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")

# Disable -Werror on Unix.
SET(CXX_DISABLE_WERROR True)

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

# Project dependencies
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(ndcurves REQUIRED)
ADD_PROJECT_DEPENDENCY(example-robot-data REQUIRED)
add_project_dependency(eiquadprog REQUIRED)
add_project_dependency(yaml-cpp CONFIG REQUIRED)
find_package(odri_control_interface REQUIRED)
find_package(osqp REQUIRED)

if(BUILD_TESTING)
  set_boost_default_options()
  export_boost_default_options()
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
  if(BUILD_PYTHON_INTERFACE)
    find_package(quadruped-walkgen REQUIRED)
  endif()
endif()

if(BUILD_PYTHON_INTERFACE)
  ADD_PROJECT_DEPENDENCY(eigenpy REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

# Main Library
set(${PROJECT_NAME}_HEADERS
  include/bindings/python.hpp
  include/qrw/Solo3D/Heightmap.hpp
  include/qrw/Solo3D/FootTrajectoryGeneratorBezier.hpp
  include/qrw/Solo3D/FootstepPlannerQP.hpp
  include/qrw/Solo3D/StatePlanner3D.hpp
  include/qrw/Solo3D/Surface.hpp
  include/qrw/MPC.hpp
  include/qrw/Gait.hpp
  include/qrw/FootTrajectoryGenerator.hpp
  include/qrw/FootstepPlanner.hpp
  include/qrw/StatePlanner.hpp
  include/qrw/Types.h
  include/qrw/InvKin.hpp
  include/qrw/QPWBC.hpp
  include/qrw/WbcWrapper.hpp
  include/qrw/Params.hpp
  include/qrw/Estimator.hpp
  include/qrw/ComplementaryFilter.hpp
  include/qrw/Joystick.hpp
  include/qrw/Filter.hpp
  include/qrw/Controller.hpp
  include/qrw/MpcWrapper.hpp
  include/qrw/FakeRobot.hpp
  include/other/st_to_cc.hpp
  )

set(${PROJECT_NAME}_SOURCES
  src/st_to_cc.cpp
  src/Solo3D/Heightmap.cpp
  src/Solo3D/FootTrajectoryGeneratorBezier.cpp
  src/Solo3D/FootstepPlannerQP.cpp
  src/Solo3D/StatePlanner3D.cpp
  src/Solo3D/Surface.cpp
  src/MPC.cpp
  src/Gait.cpp
  src/FootTrajectoryGenerator.cpp
  src/FootstepPlanner.cpp
  src/StatePlanner.cpp
  src/InvKin.cpp
  src/QPWBC.cpp
  src/WbcWrapper.cpp
  src/Params.cpp
  src/Estimator.cpp
  src/ComplementaryFilter.cpp
  src/Joystick.cpp
  src/Filter.cpp
  src/Controller.cpp
  src/MpcWrapper.cpp
  )

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include/qrw>)
target_link_libraries(${PROJECT_NAME} PUBLIC example-robot-data::example-robot-data
pinocchio::pinocchio
odri_control_interface::odri_control_interface
ndcurves::ndcurves
osqp::osqp
eiquadprog::eiquadprog)

#TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC master_board_sdk::master_board_sdk)

#find_package(yaml-cpp REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp::yaml-cpp)
#target_link_libraries(${PROJECT_NAME} PUBLIC ${YAML_CPP_LIBRARIES})

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# target_compile_options(${PROJECT_NAME} PUBLIC PRINTING)
# target_compile_definitions(${PROJECT_NAME} PUBLIC PRINTING)

target_compile_definitions(${PROJECT_NAME} PUBLIC CONFIG_SOLO12_YAML="${PROJECT_SOURCE_DIR}/config/config_solo12.yaml")
target_compile_definitions(${PROJECT_NAME} PUBLIC WALK_PARAMETERS_YAML="${PROJECT_SOURCE_DIR}/config/walk_parameters.yaml")

# Main Executable
add_executable(${PROJECT_NAMESPACE}-${PROJECT_NAME} src/control_solo12.cpp)
target_link_libraries(${PROJECT_NAMESPACE}-${PROJECT_NAME} ${PROJECT_NAME})

# MQTT Interface
if(BUILD_WITH_MQTT_SUPPORT)
  add_subdirectory (paho.mqtt.c)
  add_library(mqtt-interface SHARED include/qrw/mqtt-interface.hpp src/mqtt-interface.cpp)
  target_link_libraries(mqtt-interface PRIVATE paho-mqtt3c paho-mqtt3a)
  target_include_directories(mqtt-interface PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

  target_link_libraries(${PROJECT_NAMESPACE}-${PROJECT_NAME} mqtt-interface)
  target_compile_definitions(${PROJECT_NAMESPACE}-${PROJECT_NAME} PRIVATE -DQRW_WITH_MQTT)
endif()

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# Unit tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Installation
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
  install(TARGETS ${PROJECT_NAMESPACE}-${PROJECT_NAME} RUNTIME DESTINATION bin)
endif()
