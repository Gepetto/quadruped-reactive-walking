///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Surface class
///
/// \details Surface data structure
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SURFACE_H_INCLUDED
#define SURFACE_H_INCLUDED

#include "qrw/Types.h"

class Surface {
 public:
  // Constructor
  Surface();
  Surface(MatrixN const& A_in, VectorN const& b_in, MatrixN const& vertices_in);

  bool operator==(const Surface& other) const {
    return A_ == other.A_ && b_ == other.b_ && vertices_ == other.vertices_;
  }

  bool operator!=(const Surface& other) const {
    return A_ != other.A_ or b_ != other.b_ or vertices_ != other.vertices_;
  }

  // Destructor
  ~Surface() {}

  // Usefull for python binding
  MatrixN getA() const;
  void setA(MatrixN const& A_in);

  VectorN getb() const;
  void setb(VectorN const& b_in);

  MatrixN getVertices() const;
  void setVertices(MatrixN const& vertices_in);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief For a given X,Y point that belongs to the surface, return the
  /// height
  ///        d/c -a/c*x -b/c*y
  ///
  /// \param[in] point Vecto3 [x, y, z]
  ///
  //////////////////////////////////////////////////////////////////////////////////////////////////
  double getHeight(Vector2 const& point) const;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief For a given X,Y point return true if the point is in the surface
  ///
  /// \param[in] point Vecto3 [x, y]
  ///
  //////////////////////////////////////////////////////////////////////////////////////////////////
  bool hasPoint(Vector2 const& point) const;

  MatrixN vertices_;

 private:
  MatrixN A_;
  VectorN b_;
};

#endif  // SURFACE_H_INCLUDED
