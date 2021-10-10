#include "qrw/MpcWrapper.hpp"

MpcWrapper::MpcWrapper()
  : test(0),
    last_available_result(Eigen::Matrix<double, 24, 2>::Zero()),
    gait_past(Matrix14::Zero()),
    gait_next(Matrix14::Zero()) {}

void MpcWrapper::initialize(Params& params) {
  
  params_ = &params;
  mpc_ = MPC(params);

  // Default result for first step
  last_available_result(2, 0) = params.h_ref;
  last_available_result.col(0).tail(12) = (Vector3(0.0, 0.0, 8.0)).replicate<4, 1>();

}

void MpcWrapper::solve(int k, MatrixN const& xref, MatrixN const& fsteps,
                       MatrixN const& gait) {
  
  std::cout << "Pass mpcWrapper" << std::endl;

  // Run in parallel process
  run_MPC_asynchronous(k, xref, fsteps);

  // Adaptation if gait has changed
  if (!gait_past.isApprox(gait.row(0)))  // If gait status has changed
  {
    if (gait_next.isApprox(gait.row(0)))  // If we're still doing what was planned the last time MPC was solved
    {
      last_available_result.col(0).tail(12) = last_available_result.col(1).tail(12);
    }
    else  // Otherwise use a default contact force command till we get the actual result of the MPC for this new sequence
    {
      double F = 9.81 * params_->mass / gait.row(0).sum();
      for (int i = 0; i < 4; i++) 
      {
        last_available_result.block(12 + 3 * i, 0, 3, 0) << 0.0, 0.0, F;
      }
    }
    last_available_result.col(1).tail(12).setZero();
    gait_past = gait.row(0);
  }
  gait_next = gait.row(1);
}

void MpcWrapper::run_MPC_asynchronous(int k, MatrixN const& xref, MatrixN const& fsteps)
{
  // If this is the first iteration, creation of the parallel process
  /*
  if (k == 0)
  {
    p = Process(target=self.create_MPC_asynchronous, args=(
                self.newData, self.newResult, self.dataIn, self.dataOut, self.running))
    p.start()
  }
  */
            
  // Stacking data to send them to the parallel process
  // compress_dataIn(k, xref, fsteps)
  // newData = true

}

void MpcWrapper::create_MPC_asynchronous()
{
    /*
    while (running)
    {
        // Checking if new data is available to trigger the asynchronous MPC
        if (newData)
        {
            // Set the shared variable to false to avoid re-trigering the asynchronous MPC
            newData = false;

            // Retrieve data thanks to the decompression function and reshape it
            k, xref, fsteps = decompress_dataIn(dataIn);

            // Create the MPC object of the parallel process during the first iteration
            if (k == 0) { MPC loop_mpc = MPC(*params_); }

            // Run the asynchronous MPC with the data that as been retrieved
            loop_mpc.run(k, xref, fsteps);

            // Store the result (predicted state + desired forces) in the shared memory
            // MPC::get_latest_result() returns a matrix of size 24 x N and we want to
            // retrieve only the 2 first columns i.e. dataOut.block(0, 0, 24, 2)
            dataOut = loop_mpc.get_latest_result();

            // Set shared variable to true to signal that a new result is available
            newResult = true;
        }
    }
    */
}

Eigen::Matrix<double, 24, 2> MpcWrapper::get_latest_result()
{
  // Retrieve data from parallel process if a new result is available
  // if (there is new result) {last_available_result = retrieve data mpc i.e. dataOut.block(0, 0, 24, 2) }
  return last_available_result;
}
