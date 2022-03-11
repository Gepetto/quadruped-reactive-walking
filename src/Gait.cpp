#include "qrw/Gait.hpp"

#include <iostream>

Gait::Gait()
    : pastGait_(),
      currentGait_(),
      desiredGait_(),
      dt_(0.),
      nRows_(0),
      newPhase_(false),
      isStatic_(false),
      switchToGait_(0) {
  // Empty
}

void Gait::initialize(Params& params) {
  dt_ = params.dt_mpc;
  nRows_ = params.gait.rows();

  pastGait_ = MatrixN::Zero(nRows_, 4);
  currentGait_ = MatrixN::Zero(nRows_, 4);
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  currentGait_ = params.gait;
  desiredGait_ = currentGait_;
  pastGait_ = currentGait_;
}

void Gait::createStatic() { desiredGait_.setOnes(); }

void Gait::createWalk() {
  // Number of timesteps in 1/4th period of gait
  long int N = nRows_ / 4;

  desiredGait_ = MatrixN::Zero(nRows_, 4);

  RowVector4 sequence;
  sequence << 1., 0., 1., 0.;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 1., 0., 0., 1.;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 0., 1.;
  desiredGait_.block(2 * N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 1., 0.;
  desiredGait_.block(3 * N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::createTrot() {
  long int N = nRows_ / 2;
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  RowVector4 sequence;
  sequence << 1., 0., 0., 1.;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 1., 0.;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::createWalkingTrot() {
  long int N = nRows_ / 2;
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  long int M = 8;
  RowVector4 sequence;
  sequence << 1., 0., 0., 1.;
  desiredGait_.block(0, 0, N - M, 4) = sequence.colwise().replicate(N);
  sequence << 1., 1., 1., 1.;
  desiredGait_.block(N - M, 0, M, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 1., 0.;
  desiredGait_.block(N, 0, N - M, 4) = sequence.colwise().replicate(N);
  sequence << 1., 1., 1., 1.;
  desiredGait_.block(2 * N - M, 0, M, 4) = sequence.colwise().replicate(N);
}

void Gait::createPacing() {
  long int N = nRows_ / 2;
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  RowVector4 sequence;
  sequence << 1., 0., 1., 0.;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 0., 1.;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::createBounding() {
  long int N = nRows_ / 2;
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  RowVector4 sequence;
  sequence << 1., 1., 0., 0.;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 0., 1., 1.;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::createTransverseGallop() {
  long int N = nRows_ / 4;
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  RowVector4 sequence;
  sequence << 0., 0., 1., 0.;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 1., 0., 0., 0.;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 0., 0., 1.;
  desiredGait_.block(2 * N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 0., 0.;
  desiredGait_.block(3 * N, 0, N, 4) = sequence.colwise().replicate(N);
}

void Gait::createCustomGallop() {
  long int N = nRows_ / 4;
  desiredGait_ = MatrixN::Zero(nRows_, 4);

  RowVector4 sequence;
  sequence << 1., 0., 1., 0.;
  desiredGait_.block(0, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 1., 0., 0., 1.;
  desiredGait_.block(N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 0., 1.;
  desiredGait_.block(2 * N, 0, N, 4) = sequence.colwise().replicate(N);
  sequence << 0., 1., 1., 0.;
  desiredGait_.block(3 * N, 0, N, 4) = sequence.colwise().replicate(N);
}

double Gait::getPhaseDuration(int i, int j) { return getElapsedTime(i, j) + getRemainingTime(i, j); }

double Gait::getRemainingTime(int i, int j) {
  double state = currentGait_(i, j);
  double nPhase = 1;
  int row = i;

  while ((row < nRows_ - 1) && (currentGait_(row + 1, j) == state)) {
    row++;
    nPhase++;
  }

  if (row == nRows_ - 1) {
    row = 0;
    while ((row < nRows_) && (desiredGait_(row, j) == state)) {
      row++;
      nPhase++;
    }
  }
  return nPhase * dt_;
}

double Gait::getElapsedTime(int i, int j) {
  double state = currentGait_(i, j);
  double nPhase = 0;
  int row = i;

  while ((row > 0) && (currentGait_(row - 1, j) == state)) {
    row--;
    nPhase++;
  }

  if (row == 0) {
    row = nRows_;
    while ((row > 0) && (pastGait_(row - 1, j) == state)) {
      row--;
      nPhase++;
    }
  }
  return nPhase * dt_;
}

void Gait::update(int const k, int const k_mpc, int const joystickCode) {
  changeGait(k, k_mpc, joystickCode);
  if (k % k_mpc == 0 && k > 0) rollGait();
}

bool Gait::changeGait(int const k, int const k_mpc, int const code) {
  if (code != 0 && switchToGait_ == 0) {
    switchToGait_ = code;
  }
  if (switchToGait_ != 0 && ((k - k_mpc) % (k_mpc * nRows_ / 2) == 0)) {
    isStatic_ = false;
    switch (switchToGait_) {
      /*case 1:
        createPacing();
        break;
      case 2:
        createBounding();
        break;*/
      case 3:
        createTrot();
        break;
      case 1:
        isStatic_ = true;
        createStatic();
        break;
    }
    switchToGait_ = 0;
  }

  return isStatic_;
}

void Gait::rollGait() {
  // Age past gait
  pastGait_.topRows(nRows_ - 1) = pastGait_.bottomRows(nRows_ - 1);
  pastGait_.row(nRows_ - 1) = currentGait_.row(0);

  newPhase_ = !currentGait_.row(0).isApprox(currentGait_.row(1));

  // Age current gait
  currentGait_.topRows(nRows_ - 1) = currentGait_.bottomRows(nRows_ - 1);
  currentGait_.row(nRows_ - 1) = desiredGait_.row(0);

  // Age desired gait
  desiredGait_.topRows(nRows_ - 1) = desiredGait_.bottomRows(nRows_ - 1);
  desiredGait_.row(nRows_ - 1) = currentGait_.row(nRows_ - 1);
}