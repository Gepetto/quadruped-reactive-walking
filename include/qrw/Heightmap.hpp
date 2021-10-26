///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Heightmap class
///
/// \details This class loads a binary file, return the height and compute surface of the heightmap
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HEIGHTMAP_H_INCLUDED
#define HEIGHTMAP_H_INCLUDED

#include <stdio.h>
#include <fstream>
#include <iostream>
#include "qrw/Types.h"
#include "qrw/Params.hpp"
#include <Eigen/Dense>
#include "eiquadprog/eiquadprog-fast.hpp"

using namespace std;

struct Header {
  int size_x;
  int size_y;
  double x_init;
  double x_end;
  double y_init;
  double y_end;
};

class Heightmap {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Heightmap();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize with given data
  ///
  /// \param[in] file_name name of the binary file containing the data of the heightmap
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(const std::string& file_name);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Heightmap() {}  // Empty constructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Get the i indice of a given x-axis position in the heightmap
  ///
  /// \param[in] x x-axis position
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int map_x(double x);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Get the j indice of a given y-axis position in the heightmap
  ///
  /// \param[in] y y-axis position
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  int map_y(double y);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Get height given a 2D position
  ///
  /// \param[in] x x-axis position
  /// \param[in] y y-axis position
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  double get_height(double x, double y);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Update the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
  /// for a given 2D  position
  ///
  /// \param[in] x x-axis position
  /// \param[in] y y-axis position
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector3 compute_mean_surface(double x, double y);

  MatrixN z_;
  MatrixN x_;
  MatrixN y_;
  VectorN surface_eq = VectorN::Zero(3); // [a,b,c], such as ax + by -z + c = 0

 private:
  Header header_;  // Contain the size and parameters of the heightmap

  double dx_;  // interval size x-axis
  double dy_;  //  interval size y-axis

  double FIT_SIZE_X;  // size around x-axis the robot to detect the surface
  double FIT_SIZE_Y;  // size around x-axis the robot to detect the surface
  int FIT_NX;         // Number of point for the QP, to compute the surface, x-axis
  int FIT_NY;         // Number of point for the QP, to compute the surface, y-axis

  // min. 1/2 * x' C_ x + q_' x
  // s.t. C_ x + d_ = 0
  //      G_ x + h_ >= 0
  MatrixN P_= MatrixN::Zero(3, 3);
  VectorN q_= VectorN::Zero(3);

  MatrixN G_= MatrixN::Zero(3, 3);
  VectorN h_= VectorN::Zero(3);

  MatrixN C_= MatrixN::Zero(3, 3);
  VectorN d_= VectorN::Zero(3);

  // qp solver
  eiquadprog::solvers::EiquadprogFast_status expected = eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL;
  eiquadprog::solvers::EiquadprogFast_status status;
  eiquadprog::solvers::EiquadprogFast qp;

  MatrixN A;
  VectorN b;
};
#endif  // HEIGHTMAP_H_INCLUDED
