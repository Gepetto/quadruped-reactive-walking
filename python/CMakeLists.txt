set(${PY_NAME}_SOURCES
  gepadd.cpp
  )

add_library(${PY_NAME} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
set_target_properties(${PY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PY_NAME})
TARGET_LINK_BOOST_PYTHON(${PY_NAME} PUBLIC)
install(TARGETS ${PY_NAME} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
  __init__.py
  )

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()
