#include "example-adder/gepadd.hpp"
#include "example-adder/MPC.hpp"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>

namespace bp = boost::python;

template <typename MPC>
struct MPCPythonVisitor : public bp::def_visitor<MPCPythonVisitor<MPC> > {

  // call macro for all ContactPhase methods that can be overloaded
  // BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(isConsistent_overloads, ContactPhase::isConsistent, 0, 1)

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def(bp::init<double, int, double>(bp::args("dt_in", "n_steps_in", "T_gait_in"), "Constructor."))
        
        // Run MPC from Python
        .def("run_python", &MPC::run_python,
             bp::args("xref_in", "fsteps_in"),
             "Run MPC from Python.\n");
  }

  static void expose() {
    bp::class_<MPC>("MPC", bp::no_init)
        .def(MPCPythonVisitor<MPC>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
    //ENABLE_SPECIFIC_MATRIX_TYPE(typeXREF);
    //ENABLE_SPECIFIC_MATRIX_TYPE(typeFSTEPS);
  }

};

void exposeMPC() {
    MPCPythonVisitor<MPC>::expose();
}

BOOST_PYTHON_MODULE(libexample_adder) {
  boost::python::def("add", gepetto::example::add);
  boost::python::def("sub", gepetto::example::sub);

  eigenpy::enableEigenPy();

  exposeMPC();
}