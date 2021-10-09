#include "qrw/Controller.hpp"

Controller::Controller()
    : P(Vector12::Zero()),
      D(Vector12::Zero()),
      q_des(Vector12::Zero()),
      v_des(Vector12::Zero()),
      tau_ff(Vector12::Zero()),
      FF(0.0),
      error(false) {}

void Controller::initialize(Params& params) {
  // Params store parameters
  params_ = &params;

}

void Controller::compute(std::shared_ptr<odri_control_interface::Robot> robot) {

  std::cout << "Pass" << std::endl;
}