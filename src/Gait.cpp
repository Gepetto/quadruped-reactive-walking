#include "qrw/Gait.hpp"

Gait::Gait()
    : pastGait_(),
      currentGait_(),
      desiredGait_(),
      dt_(0.0),
      remainingTime_(0.0),
      newPhase_(false),
      is_static_(false),
      switch_to_gait_(0),
      q_static_(VectorN::Zero(19)) {
  // Empty
}

void Gait::initialize(Params& params) {
  dt_ = params.dt_mpc;

  pastGait_ = MatrixN::Zero(params.gait.rows(), 4);
  currentGait_ = MatrixN::Zero(params.gait.rows(), 4);
  desiredGait_ = MatrixN::Zero(params.gait.rows(), 4);

  // Fill desired gait matrix with yaml gait
  desiredGait_ = params.gait;

  // Fill currrent gait matrix
  currentGait_ = desiredGait_;
  pastGait_ = desiredGait_.colwise().reverse();
}

void Gait::create_walk() {
  // Number of timesteps in 1/4th period of gait
  long int N = currentGait_.rows() / 4;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 1.0, 0.0, 1.0, 0.0;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 1.0, 0.0, 0.0, 1.0;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 0.0, 1.0;
  desiredGait_.block(2 * N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 1.0, 0.0;
  desiredGait_.block(3 * N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::create_trot() {
  // Number of timesteps in a half period of gait
  long int N = currentGait_.rows() / 2;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 1.0, 0.0, 0.0, 1.0;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 1.0, 0.0;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::create_walking_trot() {
  // Number of timesteps in a half period of gait
  long int N = currentGait_.rows() / 2;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  long int M = 8;
  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 1.0, 0.0, 0.0, 1.0;
  desiredGait_.block(0, 0, N-M, 4) = sequence.colwise().replicate(N);
  sequence << 1.0, 1.0, 1.0, 1.0;
  desiredGait_.block(N-M, 0, M, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 1.0, 0.0;
  desiredGait_.block(N, 0, N-M, 4) = sequence.colwise().replicate(N);
  sequence << 1.0, 1.0, 1.0, 1.0;
  desiredGait_.block(2*N-M, 0, M, 4) = sequence.colwise().replicate(N);
}

void Gait::create_pacing() {
  // Number of timesteps in a half period of gait
  long int N = currentGait_.rows() / 2;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 1.0, 0.0, 1.0, 0.0;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 0.0, 1.0;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::create_bounding() {
  // Number of timesteps in a half period of gait
  long int N = currentGait_.rows() / 2;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 1.0, 1.0, 0.0, 0.0;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 0.0, 1.0, 1.0;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::create_static() {
  desiredGait_.setOnes();
}

void Gait::create_transverse_gallop() {
  // Number of timesteps in a half period of gait
  long int N = currentGait_.rows() / 4;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 0.0, 0.0, 1.0, 0.0;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 1.0, 0.0, 0.0, 0.0;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 0.0, 0.0, 1.0;
  desiredGait_.block(2*N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 0.0, 0.0;
  desiredGait_.block(3*N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::create_custom_gallop() {
  // Number of timesteps in a half period of gait
  long int N = currentGait_.rows() / 4;

  desiredGait_ = MatrixN::Zero(currentGait_.rows(), 4);

  Eigen::Matrix<double, 1, 4> sequence;
  sequence << 1.0, 0.0, 1.0, 0.0;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 1.0, 0.0, 0.0, 1.0;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 0.0, 1.0;
  desiredGait_.block(2*N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0.0, 1.0, 1.0, 0.0;
  desiredGait_.block(3*N, 0, N, 4) = sequence.colwise().replicate(N);
}

double Gait::getPhaseDuration(int i, int j, double value) {
  double t_phase = 1;
  int a = i;

  // Looking for the end of the swing/stance phase in currentGait_
  while ((i + 1 < currentGait_.rows()) && (currentGait_(i + 1, j) == value)) {
    i++;
    t_phase++;
  }
  // If we reach the end of currentGait_ we continue looking for the end of the swing/stance phase in desiredGait_
  if (i + 1 == currentGait_.rows()) {
    int k = 0;
    while ((k < desiredGait_.rows()) && (desiredGait_(k, j) == value)) {
      k++;
      t_phase++;
    }
  }
  // We suppose that we found the end of the swing/stance phase either in currentGait_ or desiredGait_

  remainingTime_ = t_phase;

  // Looking for the beginning of the swing/stance phase in currentGait_
  while ((a > 0) && (currentGait_(a - 1, j) == value)) {
    a--;
    t_phase++;
  }
  // If we reach the end of currentGait_ we continue looking for the beginning of the swing/stance phase in pastGait_
  if (a == 0) {
    while ((a < pastGait_.rows()) && (pastGait_(a, j) == value)) {
      a++;
      t_phase++;
    }
  }
  // We suppose that we found the beginning of the swing/stance phase either in currentGait_ or pastGait_

  // TODO: Handle infinite swing / stance phases

  return t_phase * dt_;  // Take into account time step value
}

void Gait::updateGait(int const k, int const k_mpc, int const joystickCode) {
  changeGait(k, k_mpc, joystickCode);
  if (k % k_mpc == 0 && k > 0) rollGait();
}

bool Gait::changeGait(int const k, int const k_mpc, int const code) {
  if (code != 0 && switch_to_gait_ == 0) {
    switch_to_gait_ = code;
  }
  if (switch_to_gait_ != 0 && ((k - k_mpc) % (k_mpc * currentGait_.rows() / 2) == 0)) {
    is_static_ = false;
    switch (switch_to_gait_) {
      /*case 1:
        create_pacing();
        break;
      case 2:
        create_bounding();
        break;*/
      case 3:
        create_trot();
        break;
      case 1:
        is_static_ = true;
        create_static();
        break;
    }
    switch_to_gait_ = 0;
  }

  return is_static_;
}

void Gait::rollGait() {
  // Transfer current gait into past gait
  for (long int m = pastGait_.rows() - 1; m > 0; m--)  // TODO: Find optimized circular shift function
  {
    pastGait_.row(m).swap(pastGait_.row(m - 1));
  }
  pastGait_.row(0) = currentGait_.row(0);

  // Entering new contact phase, store positions of feet that are now in contact
  if (!currentGait_.row(0).isApprox(currentGait_.row(1))) {
    newPhase_ = true;
  } else {
    newPhase_ = false;
  }

  // Age current gait
  for (int index = 1; index < currentGait_.rows(); index++)
  {
    currentGait_.row(index - 1).swap(currentGait_.row(index));
  }

  // Insert a new line from desired gait into current gait
  currentGait_.row(currentGait_.rows() - 1) = desiredGait_.row(0);

  // Age desired gait
  for (int index = 1; index < currentGait_.rows(); index++)
  {
    desiredGait_.row(index - 1).swap(desiredGait_.row(index));
  }
}

bool Gait::setGait(MatrixN const& gaitMatrix) {
  std::cout << "Gait matrix received by setGait:" << std::endl;
  std::cout << gaitMatrix << std::endl;

  // Todo: Check if the matrix is a static gait (only ones)
  return false;
}
