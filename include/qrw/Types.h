#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>

using Vector2 = Eigen::Matrix<double, 2, 1>;
using Vector3 = Eigen::Matrix<double, 3, 1>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Vector5 = Eigen::Matrix<double, 5, 1>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
using Vector11 = Eigen::Matrix<double, 11, 1>;
using Vector12 = Eigen::Matrix<double, 12, 1>;
using Vector19 = Eigen::Matrix<double, 1, 1>;
using VectorN = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Matrix3 = Eigen::Matrix<double, 3, 3>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;
using Matrix34 = Eigen::Matrix<double, 3, 4>;
using Matrix64 = Eigen::Matrix<double, 6, 4>;
using Matrix3N = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix6N = Eigen::Matrix<double, 6, Eigen::Dynamic>;
using MatrixN = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

#endif  // TYPES_H_INCLUDED
