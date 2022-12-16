#include "qrw/MpcWrapper.hpp"

#include <chrono>
#include <mutex>
#include <thread>

// Shared global variables
Params* shared_params =
    nullptr;                 // Shared pointer to object that stores parameters
bool shared_running = true;  // Flag to stop the parallel thread
bool shared_newIn =
    false;  // Flag to indicate new data has been written by main loop for MPC
bool shared_newOut =
    false;  // Flag to indicate new data has been written by MPC for main loop
int shared_k;           // Numero of the current loop
MatrixN shared_xref;    // Desired state vector for the whole prediction horizon
MatrixN shared_fsteps;  // The [x, y, z]^T desired position of each foot for
                        // each time step of the horizon
MatrixN shared_result;  // Predicted state and desired contact forces resulting
                        // of the MPC

// Mutexes to protect the global variables
std::mutex mutexStop;  // To check if the thread should still run
std::mutex mutexIn;    // From main loop to MPC
std::mutex mutexOut;   // From MPC to main loop

std::thread parallel_thread;  // spawn new thread that runs MPC in parallel

void stop_thread() {
  const std::lock_guard<std::mutex> lockStop(mutexStop);
  shared_running = false;
}

bool check_stop_thread() {
  const std::lock_guard<std::mutex> lockStop(mutexStop);
  return shared_running;
}

void write_in(int& k, MatrixN& xref, MatrixN& fsteps) {
  const std::lock_guard<std::mutex> lockIn(mutexIn);
  shared_k = k;
  shared_xref = xref;
  shared_fsteps = fsteps;
  shared_newIn = true;  // New data is available
}

bool read_in(int& k, MatrixN& xref, MatrixN& fsteps) {
  const std::lock_guard<std::mutex> lockIn(mutexIn);
  if (shared_newIn) {
    k = shared_k;
    xref = shared_xref;
    fsteps = shared_fsteps;
    shared_newIn = false;
    return true;
  }
  return false;
}

void write_out(MatrixN& result) {
  const std::lock_guard<std::mutex> lockOut(mutexOut);
  shared_result = result;
  shared_newOut = true;  // New data is available
}

bool check_new_result() {
  const std::lock_guard<std::mutex> lockOut(mutexOut);
  if (shared_newOut) {
    shared_newOut = false;
    return true;
  }
  return false;
}

MatrixN read_out() {
  const std::lock_guard<std::mutex> lockOut(mutexOut);
  return shared_result;
}

void parallel_loop() {
  while (shared_params == nullptr) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Wait a bit
  }
  int k;
  MatrixN xref;
  MatrixN fsteps;
  MatrixN result;
  MPC loop_mpc =
      MPC(*shared_params);  // Create the MPC object running in parallel
  while (check_stop_thread()) {
    // Checking if new data is available to trigger the asynchronous MPC
    if (read_in(k, xref, fsteps)) {
      // std::cout << "NEW DATA AVAILABLE, LAUNCHING MPC" << std::endl;

      /*std::cout << "Parallel k" << std::endl << k << std::endl;
      std::cout << "Parallel xref" << std::endl << xref << std::endl;
      std::cout << "Parallel fsteps" << std::endl << fsteps << std::endl;*/

      // Run the asynchronous MPC with the data that as been retrieved
      loop_mpc.run(k, xref, fsteps);

      // Store the result (predicted state + desired forces) in the shared
      // memory MPC::get_latest_result() returns a matrix of size 24 x N
      // std::cout << "NEW RESULT AVAILABLE, WRITING OUT" << std::endl;
      result = loop_mpc.get_latest_result();
      write_out(result);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Wait a bit
    }
  }
}

MpcWrapper::MpcWrapper()
    : gait_past(RowVector4::Zero()),
      gait_next(RowVector4::Zero()),
      gait_mem_(RowVector4::Zero()),
      t_mpc_solving_duration_(0.0) {}

void MpcWrapper::initialize(Params& params) {
  params_ = &params;
  mpc_ = MPC(params);

  t_mpc_solving_start_ = std::chrono::system_clock::now();

  // Default result for first step
  last_available_result = MatrixN::Zero(24, params.gait.rows());
  last_available_result(2, 0) = params.h_ref;
  last_available_result.col(0).tail(12) =
      (Vector3(0.0, 0.0, 8.0)).replicate<4, 1>();

  // Initialize the shared memory
  shared_k = 42;
  shared_xref = MatrixN::Zero(12, params.gait.rows() + 1);
  shared_fsteps = MatrixN::Zero(params.gait.rows(), 12);
  shared_params = &params;

  // Start the parallel loop
  parallel_thread = std::thread(parallel_loop);
}

MpcWrapper::~MpcWrapper() {
  stop_thread();
  parallel_thread.join();
}

void MpcWrapper::solve(int k, MatrixN xref, MatrixN fsteps, MatrixN gait) {
  t_mpc_solving_start_ = std::chrono::system_clock::now();

  // std::cout << "NEW DATA AVAILABLE, WRITING IN" << std::endl;
  write_in(k, xref, fsteps);

  // Adaptation if gait has changed
  if (!gait_past.isApprox(gait.row(0)))  // If gait status has changed
  {
    if (gait_next.isApprox(
            gait.row(0)))  // If we're still doing what was planned the last
                           // time MPC was solved
    {
      last_available_result.col(0).tail(12) =
          last_available_result.col(1).tail(12);
    } else  // Otherwise use a default contact force command till we get the
            // actual result of the MPC for this new sequence
    {
      double F = 9.81 * params_->mass / gait.row(0).sum();
      for (int i = 0; i < 4; i++) {
        last_available_result.block(12 + 3 * i, 0, 3, 1) << 0.0, 0.0, F;
      }
    }
    last_available_result.col(1).tail(12).setZero();
    gait_past = gait.row(0);
  }
  gait_next = gait.row(1);
}

MatrixN MpcWrapper::get_latest_result() {
  // Retrieve data from parallel process if a new result is available
  bool refresh = false;
  if (check_new_result()) {
    t_mpc_solving_duration_ =
        ((std::chrono::duration<double>)(std::chrono::system_clock::now() -
                                         t_mpc_solving_start_))
            .count();
    last_available_result = read_out();
    refresh = true;
  }

  if (refresh) {
    // Handle changes of gait between moment when the MPC solving is launched
    // and result is retrieved
    for (int i = 0; i < 4; i++) {
      if (flag_change_[i]) {
        // Foot in contact but was not in contact when we launched MPC solving
        // (new detection) Fetch first non zero force in the prediction horizon
        // for this foot
        int k = 0;
        while (k < last_available_result.cols() &&
               last_available_result(14 + 3 * i, k) == 0) {
          k++;
        }
        last_available_result.block(12 + 3 * i, 0, 3, 1) =
            last_available_result.block(12 + 3 * i, k, 3, 1);
      }
    }
  }

  // std::cout << "get_latest_result: " << std::endl <<
  // last_available_result.transpose() << std::endl;
  return last_available_result;
}

void MpcWrapper::get_temporary_result(RowVector4 gait) {
  // Check if gait status has changed
  bool same = false;
  for (int i = 0; i < 4 && !same; i++) {
    same = (gait[i] != gait_mem_[i]);
  }
  // If gait status has changed
  if (!same) {
    for (int i = 0; i < 4; i++) {
      if (gait[i] == 0) {
        // Foot not in contact
        last_available_result.block(12 + 3 * i, 0, 3, 1).setZero();
      } else if (gait[i] == 1 && gait_mem_[i] == 1) {
        // Foot in contact and was already in contact
        continue;
      } else if (gait[i] == 1 && gait_mem_[i] == 0) {
        // Foot in contact but was not in contact (new detection)
        flag_change_[i] = true;
        // Fetch first non zero force in the prediction horizon for this foot
        int k = 0;
        while (k < last_available_result.cols() &&
               last_available_result(14 + 3 * i, k) == 0) {
          k++;
        }
        last_available_result.block(12 + 3 * i, 0, 3, 1) =
            last_available_result.block(12 + 3 * i, k, 3, 1);
      }
    }
    gait_mem_ = gait;
  }
}

void MpcWrapper::stop_parallel_loop() {
  stop_thread();
  // std::terminate(parallel_thread);
}
