#include "qrw/Filter.hpp"

Filter::Filter()
  : b_(Vector5::Zero())
  , a_(Vector5::Zero())
  , x_(Vector6::Zero())
  , y_(VectorN::Zero(6, 1))
  , accum_(Vector6::Zero())
  , init_(false)
{
  // Empty
}

void Filter::initialize(Params& params)
{
  if (params.dt_wbc == 0.001)
  {
    b_ << 3.12389769e-5, 1.24955908e-4, 1.87433862e-4, 1.24955908e-4, 3.12389769e-5;
    a_ << 1., -3.58973389, 4.85127588, -2.92405266, 0.66301048;
  }
  else if (params.dt_wbc == 0.002)
  {
    b_ << 0.0004166, 0.0016664, 0.0024996, 0.0016664, 0.0004166;
    a_ << 1., -3.18063855, 3.86119435, -2.11215536, 0.43826514;
  }
  else
  {
    throw std::runtime_error("No coefficients defined for this time step.");
  }
  
  x_queue_.resize(b_.rows(), Vector6::Zero());
  y_queue_.resize(a_.rows()-1, Vector6::Zero());
}

void Filter::filter(VectorN const& x)
{
  // If x is position + quaternion then we convert quaternion to RPY
  if (x.rows() == 7)
  {
    x_.head(3) = x.head(3);
    Eigen::Quaterniond quat(x(6, 0), x(3, 0), x(4, 0), x(5, 0));  // w, x, y, z
    x_.tail(3) = pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());

    // Handle 2 pi modulo for roll, pitch and yaw
    // Should happen sometimes for yaw but now for roll and pitch except
    // if the robot rolls over
    for (int i = 3; i < 6; i++) 
    {
      if (std::abs(x_(i, 0) - y_(i, 0)) > 1.5 * M_PI)
      {
        std::cout << "Modulo for " << i << std::endl;
        handle_modulo(i, x_(i, 0) - y_(i, 0) > 0);
      }
    }
  }
  else  // Otherwise we can directly use x
  {
    x_ = x;
  }

  // Initialisation of the value in the queues to the first measurement
  if (!init_)
  {
    init_ = true;
    std::fill(x_queue_.begin(), x_queue_.end(), x_.head(6));
    std::fill(y_queue_.begin(), y_queue_.end(), x_.head(6));
  }

  // Store measurement in x queue
  x_queue_.pop_back();
  x_queue_.push_front(x_.head(6));

  // Compute result (y/x = b/a for the transfert function)
  accum_ = Vector6::Zero();
  for (int i = 0; i < b_.rows(); i++) 
  {
    accum_ += b_[i] * x_queue_[i];
  }
  for (int i = 1; i < a_.rows(); i++) 
  {
    accum_ -= a_[i] * y_queue_[i-1];
  }
  
  // Store result in y queue for recursion
  y_queue_.pop_back();
  y_queue_.push_front(accum_ / a_[0]);

  // Filtered result is stored in y_queue_.front()
  // Assigned to dynamic-sized vector for binding purpose
  y_ = y_queue_.front();
}

void Filter::handle_modulo(int a, bool dir)
{
  // Add or remove 2 PI to all elements in the queues
  for (int i = 0; i < b_.rows(); i++) 
  {
    (x_queue_[i])(a, 0) += dir ? 2.0 * M_PI : -2.0 * M_PI;
  }
  for (int i = 1; i < a_.rows(); i++) 
  {
    (y_queue_[i-1])(a, 0) += dir ? 2.0 * M_PI : -2.0 * M_PI;
  }
}
