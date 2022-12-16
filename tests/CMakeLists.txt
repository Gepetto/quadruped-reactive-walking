add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

add_unit_test(test_estimator test_estimator.cpp)
target_link_libraries(test_estimator PRIVATE ${PROJECT_NAME})

add_unit_test(boost_estimator boost_estimator.cpp)
target_link_libraries(boost_estimator PRIVATE ${PROJECT_NAME}
                                              Boost::unit_test_framework)

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
