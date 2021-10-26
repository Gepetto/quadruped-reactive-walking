#include "qrw/Heightmap.hpp"

Heightmap::Heightmap() {
  // empty
}

void Heightmap::initialize(const std::string& file_name) {
  // Open the binary file
  std::ifstream iF(file_name, std::ios::in | std::ios::out | std::ios::binary);
  if (!iF) {
    throw std::runtime_error("Error while opening heighmap binary file");
  }

  // Extract header from binary file
  iF.read(reinterpret_cast<char*>(&header_), sizeof header_);

  // Resize matrix and vector according to header
  z_ = MatrixN::Zero(header_.size_x, header_.size_y);
  x_ = VectorN::LinSpaced(header_.size_x, header_.x_init, header_.x_end);
  y_ = VectorN::LinSpaced(header_.size_y, header_.y_init, header_.y_end);

  dx_ = std::abs((header_.x_init - header_.x_end) / (header_.size_x - 1));
  dy_ = std::abs((header_.y_init - header_.y_end) / (header_.size_y - 1));

  FIT_SIZE_X = 0.3;
  FIT_SIZE_Y = 0.3;
  FIT_NX = 16;
  FIT_NY = 6;

  int i = 0;
  int j = 0;
  double read;
  // Read the file and extract heightmap matrix
  while (i < header_.size_x && !iF.eof()) {
    j = 0;
    while (j < header_.size_y && !iF.eof()) {
      iF.read(reinterpret_cast<char*>(&read), sizeof read);
      z_(i, j) = read;
      j++;
    }
    i++;
  }

  A = MatrixN::Zero(FIT_NX * FIT_NY, 3);
  b = VectorN::Zero(FIT_NX * FIT_NY);
  surface_eq = Vector3::Zero();
}

int Heightmap::map_x(double x) {
  if (x < header_.x_init || x > header_.x_end) {
    return -10;
  } else {
    return (int)std::round((x - header_.x_init) / dx_);
  }
}

int Heightmap::map_y(double y) {
  if (y < header_.y_init || y > header_.y_end) {
    return -10;
  } else {
    return (int)std::round((y - header_.y_init) / dy_);
  }
}

double Heightmap::get_height(double x, double y) {
  int index_x = map_x(x);
  int index_y = map_y(y);
  if (index_x == -10 || index_y == -10) {
    return 0.0;
  } else {
    return z_(index_x, index_y);
  }
}

Vector3 Heightmap::compute_mean_surface(double x, double y) {
  VectorN x_surface = VectorN::LinSpaced(FIT_NX, x - FIT_SIZE_X / 2, x + FIT_SIZE_X / 2);
  VectorN y_surface = VectorN::LinSpaced(FIT_NY, y - FIT_SIZE_Y / 2, x + FIT_SIZE_Y / 2);

  int i_pb = 0;
  for (int i = 0; i < FIT_NX; i++) {
    for (int j = 0; j < FIT_NY; j++) {
      A.block(i_pb, 0, 1, 3) << x_surface[i],y_surface[j], 1.0;
      b.block(i_pb, 0, 1, 1) << get_height(x_surface[i],y_surface[j]);
      i_pb += 1;
    }
  }

  qp.reset(3, 0, 0);
  P_ = A.transpose() * A;
  q_ = -A.transpose() * b;
  status = qp.solve_quadprog(P_, q_, C_, d_, G_, h_, surface_eq);
  return surface_eq;
}
