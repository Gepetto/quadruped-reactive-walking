#include "qrw/ComplementaryFilter.hpp"

ComplementaryFilter::ComplementaryFilter()
    : dt_(0.),
      HighPass_(Vector3::Zero()),
      LowPass_(Vector3::Zero()),
      alpha_(Vector3::Zero()),
      x_(Vector3::Zero()),
      dx_(Vector3::Zero()),
      filteredX_(Vector3::Zero()) {}

ComplementaryFilter::ComplementaryFilter(double dt, Vector3 HighPass,
                                         Vector3 LowPass)
    : dt_(dt),
      HighPass_(HighPass),
      LowPass_(LowPass),
      alpha_(Vector3::Zero()),
      x_(Vector3::Zero()),
      dx_(Vector3::Zero()),
      filteredX_(Vector3::Zero()) {}

void ComplementaryFilter::initialize(double dt, Vector3 HighPass,
                                     Vector3 LowPass) {
  dt_ = dt;
  HighPass_ = HighPass;
  LowPass_ = LowPass;
}

Vector3 ComplementaryFilter::compute(Vector3 const& x, Vector3 const& dx,
                                     Vector3 const& alpha) {
  alpha_ = alpha;
  x_ = x;
  dx_ = dx;

  HighPass_ = alpha.cwiseProduct(HighPass_ + dx_ * dt_);
  LowPass_ =
      alpha.cwiseProduct(LowPass_) + (Vector3::Ones() - alpha).cwiseProduct(x_);
  filteredX_ = HighPass_ + LowPass_;

  return filteredX_;
}
