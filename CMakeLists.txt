cmake_minimum_required(VERSION 3.1)

# Project properties
set(PROJECT_NAMESPACE gepetto)
set(PROJECT_NAME example-adder)
set(PROJECT_DESCRIPTION "This is an example project, to show how to use Gepetto's tools")
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

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

# Project dependencies
set(BOOST_COMPONENTS unit_test_framework)
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED)

if(BUILD_PYTHON_INTERFACE)
  FINDPYTHON()
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
  set(BOOST_COMPONENTS ${BOOST_COMPONENTS} python)
endif()

SEARCH_FOR_BOOST()

# Main Library
set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/gepadd.hpp
  include/${PROJECT_NAME}/MPC.hpp
  include/osqp_folder/include/osqp.h
  include/osqp_folder/include/cs.h
  )

set(${PROJECT_NAME}_SOURCES
  src/gepadd.cpp
  src/MPC.cpp
  )

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})

# Find OSQP library and headers
# find_package(osqp REQUIRED)
# Link the OSQP shared library
target_link_libraries(${PROJECT_NAME} PUBLIC /home/palex/Documents/Test_Cpp/example-adder/include/osqp_folder/lib/libosqp.so)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# Main Executable
add_executable(${PROJECT_NAMESPACE}-${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAMESPACE}-${PROJECT_NAME} ${PROJECT_NAME})

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
