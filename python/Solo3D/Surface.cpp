#include "qrw/Surface.hpp"

#include "bindings/python.hpp"

template <typename Surface>
struct SurfaceVisitor : public bp::def_visitor<SurfaceVisitor<Surface>> {
  template <class PyClassSurface>
  void visit(PyClassSurface& cl) const {
    cl.def(bp::init<>(bp::arg(""), "Default constructor."))
        .def(bp::init<MatrixN, VectorN, MatrixN>(bp::args("A", "b", "vertices"), "Constructor with parameters."))

        .def("get_vertices", &Surface::getVertices, "get the vertices of the surface.\n")
        .def("get_A", &Surface::getA, "get A vector of inequalities.\n")
        .def("get_b", &Surface::getb, "get b vector of inequalities.\n")

        .add_property("A", bp::make_function(&Surface::getA, bp::return_value_policy<bp::return_by_value>()))
        .add_property("b", bp::make_function(&Surface::getb, bp::return_value_policy<bp::return_by_value>()))
        .add_property("vertices",
                      bp::make_function(&Surface::getVertices, bp::return_value_policy<bp::return_by_value>()))

        .def("getHeight", &Surface::getHeight, bp::args("point"), "get the height of a point of the surface.\n")
        .def("has_point", &Surface::hasPoint, bp::args("point"), "return true if the point is in the surface.\n");
  }

  static void expose() { bp::class_<Surface>("Surface", bp::no_init).def(SurfaceVisitor<Surface>()); }
};

void exposeSurface() { SurfaceVisitor<Surface>::expose(); }