#include "qrw/Solo3D/Heightmap.hpp"

Heightmap::Heightmap()
    : z_(),
      fit_(VectorN::Zero(3)),
      p(),
      A_(MatrixN::Ones(p.heightmap_fit_size * p.heightmap_fit_size, 3)),
      b_(VectorN::Zero(p.heightmap_fit_size * p.heightmap_fit_size)) {
  // empty
}

void Heightmap::initialize(const std::string &fileName) {
  // Open the binary file
  std::ifstream file(fileName, std::ios::in | std::ios::out | std::ios::binary);
  if (!file) {
    throw std::runtime_error("Error while opening heighmap binary file");
  }

  // Extract header from binary file
  file.read(reinterpret_cast<char *>(&map_), sizeof map_);

  // Resize matrix and vector according to header
  z_ = MatrixN::Zero(map_.size_x, map_.size_y);

  dx_ = std::abs((map_.x_init - map_.x_end) / (map_.size_x - 1));
  dy_ = std::abs((map_.y_init - map_.y_end) / (map_.size_y - 1));

  // Read the file and extract heightmap matrix
  int i = 0;
  int j = 0;
  double read;
  while (i < map_.size_x && !file.eof()) {
    j = 0;
    while (j < map_.size_y && !file.eof()) {
      file.read(reinterpret_cast<char *>(&read), sizeof read);
      z_(i, j) = read;
      j++;
    }
    i++;
  }
}

int Heightmap::xIndex(double x) {
  return (x < map_.x_init || x > map_.x_end)
             ? -1
             : (int)std::round((x - map_.x_init) / dx_);
}

int Heightmap::yIndex(double y) {
  return (y < map_.y_init || y > map_.y_end)
             ? -1
             : (int)std::round((y - map_.y_init) / dy_);
}

double Heightmap::getHeight(double x, double y) {
  int iX = xIndex(x);
  int iY = yIndex(y);
  return (iX == -1 || iY == -1) ? 0. : z_(iX, iY);
}

Vector3 Heightmap::fitSurface_(double x, double y) {
  VectorN xVector =
      VectorN::LinSpaced(p.heightmap_fit_size, x - p.heightmap_fit_length,
                         x + p.heightmap_fit_length);
  VectorN yVector =
      VectorN::LinSpaced(p.heightmap_fit_size, y - p.heightmap_fit_length,
                         y + p.heightmap_fit_length);

  int index = 0;
  for (int i = 0; i < p.heightmap_fit_size; i++) {
    for (int j = 0; j < p.heightmap_fit_size; j++) {
      A_.block(index, 0, 1, 2) << xVector[i], yVector[j];
      b_(index) = getHeight(xVector[i], yVector[j]);
      index++;
    }
  }

  qp.reset(3, 0, 0);
  P_ = A_.transpose() * A_;
  q_ = -A_.transpose() * b_;
  status = qp.solve_quadprog(P_, q_, C_, d_, G_, h_, fit_);

  return fit_;
}
