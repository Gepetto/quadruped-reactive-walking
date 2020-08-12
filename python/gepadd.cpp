#include "example-adder/gepadd.hpp"
#include "example-adder/MPC.hpp"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>

namespace bp = boost::python;

template <typename MPC>
struct MPCPythonVisitor : public bp::def_visitor<MPCPythonVisitor<MPC> > {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def(bp::init<double, int, double>(bp::args("dt_in", "n_steps_in", "T_gait_in"),
                                           "Constructor with parameters."))

        // Run MPC from Python
        .def("run_python", &MPC::run_python, bp::args("num_iter", "xref_in", "fsteps_in"), "Run MPC from Python.\n");
  }

  static void expose() {
    bp::class_<MPC>("MPC", bp::no_init).def(MPCPythonVisitor<MPC>());

    ENABLE_SPECIFIC_MATRIX_TYPE(matXd);
  }
};

void exposeMPC() { MPCPythonVisitor<MPC>::expose(); }

BOOST_PYTHON_MODULE(libexample_adder) {
  boost::python::def("add", gepetto::example::add);
  boost::python::def("sub", gepetto::example::sub);

  eigenpy::enableEigenPy();

  exposeMPC();
}