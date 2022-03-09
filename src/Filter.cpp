#include "qrw/Filter.hpp"

Filter::Filter()
    : b_(0.),
      a_(Vector2::Zero()),
      x_(Vector6::Zero()),
      y_(VectorN::Zero(6, 1)),
      accum_(Vector6::Zero()),
      init_(false) {
  // Empty
}

void Filter::initialize(Params& params) {
  const double fc = 15.0;
  b_ = (2 * M_PI * params.dt_wbc * fc) / (2 * M_PI * params.dt_wbc * fc + 1.0);

  a_ << 1.0, -(1.0 - b_);

  x_queue_.resize(1, Vector6::Zero());
  y_queue_.resize(a_.rows() - 1, Vector6::Zero());
}

VectorN Filter::filter(Vector6 const& x, bool check_modulo) {
  // Retrieve measurement
  x_ = x;

  // Handle modulo for orientation
  if (check_modulo) {
    // Handle 2 pi modulo for roll, pitch and yaw
    // Should happen sometimes for yaw but now for roll and pitch except
    // if the robot rolls over
    for (int i = 3; i < 6; i++) {
      if (std::abs(x_(i, 0) - y_(i, 0)) > 1.5 * M_PI) {
        handle_modulo(i, x_(i, 0) - y_(i, 0) > 0);
      }
    }
  }

  // Initialisation of the value in the queues to the first measurement
  if (!init_) {
    init_ = true;
    std::fill(x_queue_.begin(), x_queue_.end(), x_.head(6));
    std::fill(y_queue_.begin(), y_queue_.end(), x_.head(6));
  }

  // Store measurement in x queue
  x_queue_.pop_back();
  x_queue_.push_front(x_.head(6));

  // Compute result (y/x = b/a for the transfert function)
  accum_ = b_ * x_queue_[0];
  for (int i = 1; i < a_.rows(); i++) {
    accum_ -= a_[i] * y_queue_[i - 1];
  }

  // Store result in y queue for recursion
  y_queue_.pop_back();
  y_queue_.push_front(accum_ / a_[0]);

  // Filtered result is stored in y_queue_.front()
  // Assigned to dynamic-sized vector for binding purpose
  y_ = y_queue_.front();

  return y_;
}

void Filter::handle_modulo(int a, bool dir) {
  // Add or remove 2 PI to all elements in the queues
  x_queue_[0](a, 0) += dir ? 2.0 * M_PI : -2.0 * M_PI;
  for (int i = 1; i < a_.rows(); i++) {
    (y_queue_[i - 1])(a, 0) += dir ? 2.0 * M_PI : -2.0 * M_PI;
  }
}
