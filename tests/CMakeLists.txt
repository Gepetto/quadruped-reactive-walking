add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

ADD_UNIT_TEST(add add)
target_link_libraries(add ${PROJECT_NAME})

ADD_UNIT_TEST(boost_add boost_add)
target_link_libraries(boost_add ${PROJECT_NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
