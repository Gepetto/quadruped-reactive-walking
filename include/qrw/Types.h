#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <Eigen/Core>
#include <Eigen/Dense>

using Vector1 = Eigen::Matrix<double, 1, 1>;
using Vector2 = Eigen::Matrix<double, 2, 1>;
using Vector3 = Eigen::Matrix<double, 3, 1>;
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Vector5 = Eigen::Matrix<double, 5, 1>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
using Vector8 = Eigen::Matrix<double, 8, 1>;
using Vector11 = Eigen::Matrix<double, 11, 1>;
using Vector12 = Eigen::Matrix<double, 12, 1>;
using Vector18 = Eigen::Matrix<double, 18, 1>;
using Vector19 = Eigen::Matrix<double, 19, 1>;
using Vector24 = Eigen::Matrix<double, 24, 1>;
using VectorN = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using VectorNi = Eigen::Matrix<int, Eigen::Dynamic, 1>;

using RowVector4 = Eigen::Matrix<double, 1, 4>;

using Matrix2 = Eigen::Matrix<double, 2, 2>;
using Matrix3 = Eigen::Matrix<double, 3, 3>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;

using Matrix34 = Eigen::Matrix<double, 3, 4>;
using Matrix43 = Eigen::Matrix<double, 4, 3>;
using Matrix64 = Eigen::Matrix<double, 6, 4>;
using Matrix74 = Eigen::Matrix<double, 7, 4>;
using MatrixN4 = Eigen::Matrix<double, Eigen::Dynamic, 4>;
using MatrixN4i = Eigen::Matrix<int, Eigen::Dynamic, 4>;
using Matrix3N = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix6N = Eigen::Matrix<double, 6, Eigen::Dynamic>;
using Matrix12N = Eigen::Matrix<double, 12, Eigen::Dynamic>;
using MatrixN = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixNi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

#endif  // TYPES_H_INCLUDED
