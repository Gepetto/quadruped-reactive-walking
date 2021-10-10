#include "qrw/MpcWrapper.hpp"

MpcWrapper::MpcWrapper()
    : test(0) {}

void MpcWrapper::initialize(Params& params) {
  
  params_ = &params;
  mpc_ = MPC(params);

}

void MpcWrapper::solve(int k, MatrixN const& xref, MatrixN const& fsteps,
                       MatrixN const& gait, Matrix34 const& l_fsteps_target) {
  
  std::cout << "Pass" << std::endl;

}

Vector12 MpcWrapper::get_latest_result()
{
    return Vector12::Zero();
}
