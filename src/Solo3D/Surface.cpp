#include "qrw/Solo3D/Surface.hpp"

Surface::Surface() {
  // Empty
}

Surface::Surface(const MatrixN& A_in, const VectorN& b_in, const MatrixN& vertices_in)
    : A_{A_in}, b_{b_in}, vertices_{vertices_in} {
  // Empty
}

MatrixN Surface::getA() const { return A_; }

void Surface::setA(MatrixN const& A_in) { A_ = A_in; }

VectorN Surface::getb() const { return b_; }

void Surface::setb(VectorN const& b_in) { b_ = b_in; }

MatrixN Surface::getVertices() const { return vertices_; }

void Surface::setVertices(const MatrixN& vertices_in) { vertices_ = vertices_in; }

double Surface::getHeight(Vector2 const& point) const {
  int id = A_.rows() - 1;
  return abs(b_(id) - point(0) * A_(id, 0) / A_(id, 2) - point(1) * A_(id, 1) / A_(id, 2));
}

bool Surface::hasPoint(Vector2 const& point) const {
  VectorN Ax = A_.block(0, 0, A_.rows() - 2, 2) * point;
  for (int i; i < b_.size() - 2; i++) {
    if (Ax(i) > b_(i)) {
      return false;
    }
  }
  return true;
}