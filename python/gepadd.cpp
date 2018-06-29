#include "example-adder/gepadd.hpp"

#include <boost/python.hpp>

BOOST_PYTHON_MODULE(libexample_adder) {
  boost::python::def("add", gepetto::example::add);
  boost::python::def("sub", gepetto::example::sub);
}
