#include "quadruped-reactive-walking/MPC.hpp"

MPC::MPC(double dt_in, int n_steps_in, double T_gait_in) {
  dt = dt_in;
  n_steps = n_steps_in;
  T_gait = T_gait_in;

  xref = Eigen::Matrix<double, 12, Eigen::Dynamic>::Zero(12, 1 + n_steps);
  x = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(12 * n_steps * 2, 1);
  S_gait = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(12 * n_steps, 1);
  warmxf = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(12 * n_steps * 2, 1);
  x_f_applied = Eigen::MatrixXd::Zero(24, n_steps);

  // Predefined variables
  mass = 2.50000279f;
  mu = 0.9f;
  cpt_ML = 0;
  cpt_P = 0;

  // Predefined matrices
  footholds << 0.19, 0.19, -0.19, -0.19, 0.15005, -0.15005, 0.15005, -0.15005, 0.0, 0.0, 0.0, 0.0;
  gI << 3.09249e-2, -8.00101e-7, 1.865287e-5, -8.00101e-7, 5.106100e-2, 1.245813e-4, 1.865287e-5, 1.245813e-4,
      6.939757e-2;
  q << 0.0f, 0.0f, 0.2027682f, 0.0f, 0.0f, 0.0f;
  h_ref = q(2, 0);
  g(8, 0) = -9.81f * dt;

  osqp_set_default_settings(settings);
}

MPC::MPC() { MPC(0.02, 32, 0.64); }

/*
Create the constraint matrices of the MPC (M.X = N and L.X <= K)
Create the weight matrices P and Q of the MPC solver (cost 1/2 x^T * P * X + X^T * Q)
*/
int MPC::create_matrices() {
  // Create the constraint matrices
  create_ML();
  create_NK();

  // Create the weight matrices
  create_weight_matrices();

  return 0;
}

/*
Add a new non-zero coefficient to the ML matrix by filling the triplet r_ML / c_ML / v_ML
*/
inline void MPC::add_to_ML(int i, int j, double v, int *r_ML, int *c_ML, double *v_ML) {
  r_ML[cpt_ML] = i;  // row index
  c_ML[cpt_ML] = j;  // column index
  v_ML[cpt_ML] = v;  // value of coefficient
  cpt_ML++;          // increment the counter
}

/*
Add a new non-zero coefficient to the P matrix by filling the triplet r_P / c_P / v_P
*/
inline void MPC::add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P) {
  r_P[cpt_P] = i;  // row index
  c_P[cpt_P] = j;  // column index
  v_P[cpt_P] = v;  // value of coefficient
  cpt_P++;         // increment the counter
}

/*
Create the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K
*/
int MPC::create_ML() {
  int *r_ML = new int[size_nz_ML];        // row indexes of non-zero values in matrix ML
  int *c_ML = new int[size_nz_ML];        // col indexes of non-zero values in matrix ML
  double *v_ML = new double[size_nz_ML];  // non-zero values in matrix ML

  std::fill_n(r_ML, size_nz_ML, 0);
  std::fill_n(c_ML, size_nz_ML, 0);
  std::fill_n(v_ML, size_nz_ML, 0.0);  // initialized to -1.0

  // Put identity matrices in M
  for (int k = 0; k < (12 * n_steps); k++) {
    add_to_ML(k, k, -1.0, r_ML, c_ML, v_ML);
  }

  // Fill matrix A (for other functions)
  A.block(0, 6, 6, 6) = dt * Eigen::Matrix<double, 6, 6>::Identity();

  // Put A matrices in M
  for (int k = 0; k < (n_steps - 1); k++) {
    for (int i = 0; i < 12; i++) {
      add_to_ML((k + 1) * 12 + i, (k * 12) + i, 1.0, r_ML, c_ML, v_ML);
    }
    for (int j = 0; j < 6; j++) {
      add_to_ML((k + 1) * 12 + j, (k * 12) + j + 6, dt, r_ML, c_ML, v_ML);
    }
  }

  // Put B matrices in M
  double div_tmp = dt / mass;
  for (int k = 0; k < n_steps; k++) {
    for (int i = 0; i < 4; i++) {
      add_to_ML(12 * k + 6, 12 * (n_steps + k) + 0 + 3 * i, div_tmp, r_ML, c_ML, v_ML);
      add_to_ML(12 * k + 7, 12 * (n_steps + k) + 1 + 3 * i, div_tmp, r_ML, c_ML, v_ML);
      add_to_ML(12 * k + 8, 12 * (n_steps + k) + 2 + 3 * i, div_tmp, r_ML, c_ML, v_ML);
    }
    for (int i = 0; i < 12; i++) {
      add_to_ML(12 * k + 9, 12 * (n_steps + k) + i, 8.0, r_ML, c_ML, v_ML);
      add_to_ML(12 * k + 10, 12 * (n_steps + k) + i, 8.0, r_ML, c_ML, v_ML);
      add_to_ML(12 * k + 11, 12 * (n_steps + k) + i, 8.0, r_ML, c_ML, v_ML);
    }
  }
  for (int i = 0; i < 4; i++) {
    B(6, 0 + 3 * i) = div_tmp;
    B(7, 1 + 3 * i) = div_tmp;
    B(8, 2 + 3 * i) = div_tmp;
    B(9, i) = 8.0;
    B(10, i) = 8.0;
    B(11, i) = 8.0;
  }

  // Add lines to enable/disable forces
  for (int i = 12 * n_steps; i < 12 * n_steps * 2; i++) {
    add_to_ML(i, i, 1.0, r_ML, c_ML, v_ML);
  }

  // Fill ML with F matrices
  int offset_L = 12 * n_steps * 2;
  for (int k = 0; k < n_steps; k++) {
    int di = offset_L + 20 * k;
    int dj = 12 * (n_steps + k);
    // Matrix F with top left corner at (di, dj) in ML
    for (int i = 0; i < 4; i++) {
      int dx = 5 * i;
      int dy = 3 * i;
      int a[9] = {0, 1, 2, 3, 0, 1, 2, 3, 4};
      int b[9] = {0, 0, 1, 1, 2, 2, 2, 2, 2};
      double c[9] = {1.0, -1.0, 1.0, -1.0, -mu, -mu, -mu, -mu, -1};
      // Matrix C with top left corner at (dx, dy) in F
      for (int j = 0; j < 9; j++) {
        add_to_ML(di + dx + a[j], dj + dy + b[j], c[j], r_ML, c_ML, v_ML);
      }
    }
  }

  // Creation of CSC matrix
  int *icc;                                  // row indices
  int *ccc;                                  // col indices
  double *acc;                               // coeff values
  int nst = cpt_ML;                          // number of non zero elements
  int ncc = st_to_cc_size(nst, r_ML, c_ML);  // number of CC values
  // int m = 12 * n_steps * 2 + 20 * n_steps;   // number of rows
  int n = 12 * n_steps * 2;                  // number of columns

  /*int i_min = i4vec_min(nst, r_ML);
  int i_max = i4vec_max(nst, r_ML);
  int j_min = i4vec_min(nst, c_ML);
  int j_max = i4vec_max(nst, c_ML);
  st_header_print(i_min, i_max, j_min, j_max, m, n, nst);*/

  // Get the CC indices.
  icc = (int *)malloc(ncc * sizeof(int));
  ccc = (int *)malloc((n + 1) * sizeof(int));
  st_to_cc_index(nst, r_ML, c_ML, ncc, n, icc, ccc);

  // Get the CC values.
  acc = st_to_cc_values(nst, r_ML, c_ML, v_ML, ncc, n, icc, ccc);

  // Assign values to the csc object
  ML = (csc *)c_malloc(sizeof(csc));
  ML->m = 12 * n_steps * 2 + 20 * n_steps;
  ML->n = 12 * n_steps * 2;
  ML->nz = -1;
  ML->nzmax = ncc;
  ML->x = acc;
  ML->i = icc;
  ML->p = ccc;

  // Free memory
  delete[] r_ML;
  delete[] c_ML;
  delete[] v_ML;

  // Print CC matrix
  // char rt_char[2] = {'R', 'T'};
  // char cc_char[2] = {'C', 'C'};
  // st_print(m, n, 25, r_ML, c_ML, v_ML, rt_char);
  // cc_print ( m, n, 25, icc, ccc, acc, cc_char);

  // Create indices list that will be used to update ML
  int i_x_tmp[12] = {6, 9, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 12; i++) {
      i_x_B[12 * k + i] = i_x_tmp[i];
      i_y_B[12 * k + i] = (12 * k + i) / 4;  // 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2...
    }
  }

  int i_start = 30 * n_steps - 18;
  int i_data[12] = {0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17};
  int i_foot[4] = {0 * 24, 1 * 24, 2 * 24, 3 * 24};
  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 12; i++) {
      i_update_B[12 * k + i] = i_start + i_data[i] + i_foot[k];
    }
  }

  // i_update_S here?

  // Update state of B
  for (int k = 0; k < n_steps; k++) {
    // Get inverse of the inertia matrix for time step k
    double c = cos(xref(5, k));
    double s = sin(xref(5, k));
    Eigen::Matrix<double, 3, 3> R;
    R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix<double, 3, 3> R_gI = R.transpose() * gI * R;
    Eigen::Matrix<double, 3, 3> I_inv = R_gI.inverse();

    // Get skew-symetric matrix for each foothold
    Eigen::Matrix<double, 3, 4> l_arms = footholds - (xref.block(0, k, 3, 1)).replicate<1, 4>();
    for (int i = 0; i < 4; i++) {
      B.block(9, 3 * i, 3, 3) = dt * (I_inv * getSkew(l_arms.col(i)));
    }

    int i_iter = 24 * 4 * k;
    for (int j = 0; j < 12 * 4; j++) {
      ML->x[i_update_B[j] + i_iter] = B(i_x_B[j], i_y_B[j]);
    }
  }

  // Update lines to enable/disable forces
  construct_S();

  Eigen::Matrix<int, 3, 1> i_tmp1;
  i_tmp1 << 3 + 4, 3 + 4, 6 + 4;
  Eigen::Matrix<int, Eigen::Dynamic, 1> i_tmp2 =
      Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(12 * n_steps, 1);  // i_tmp1.replicate<4,1>();
  for (int k = 0; k < 4 * n_steps; k++) {
    i_tmp2.block(3 * k, 0, 3, 1) = i_tmp1;
  }

  i_off = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(12 * n_steps, 1);
  i_off(0, 0) = 4;
  for (int k = 1; k < 12 * n_steps; k++) {
    i_off(k, 0) = i_off(k - 1, 0) + i_tmp2(k - 1, 0);
    // ML->x[i_off(k, 0)+ i_start] = S_gait(k, 0);
  }
  for (int k = 0; k < 12 * n_steps; k++) {
    ML->x[i_off(k, 0) + i_start] = S_gait(k, 0);
  }

  return 0;
}

/*
Create the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K
*/
int MPC::create_NK() {
  // Create NK matrix (upper and lower bounds)
  NK_up = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(12 * n_steps * 2 + 20 * n_steps, 1);
  NK_low = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(12 * n_steps * 2 + 20 * n_steps, 1);

  // Fill N matrix with g matrices
  for (int k = 0; k < n_steps; k++) {
    NK_up(12 * k + 8, 0) = -g(8, 0);  // only 8-th coeff is non zero
  }

  // Including - A*X0 in the first row of N
  NK_up.block(0, 0, 12, 1) += A * (-x0);

  // Create matrix D (third term of N) and put identity matrices in it
  D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(12 * n_steps, 12 * n_steps);

  // Put -A matrices in D
  for (int k = 0; k < n_steps - 1; k++) {
    for (int i = 0; i < 12; i++) {
      D((k + 1) * 12 + i, (k * 12) + i) = -1.0;
    }
    for (int i = 0; i < 6; i++) {
      D((k + 1) * 12 + i, (k * 12) + i + 6) = -dt;
    }
  }

  // Add third term to matrix N
  Eigen::Map<Eigen::MatrixXd> xref_col((xref.block(0, 1, 12, n_steps)).data(), 12 * n_steps, 1);
  NK_up.block(0, 0, 12 * n_steps, 1) += D * xref_col;

  // Lines to enable/disable forces are already initialized (0 values)
  // Matrix K is already initialized (0 values)
  Eigen::Matrix<double, Eigen::Dynamic, 1> inf_lower_bount =
      -std::numeric_limits<double>::infinity() * Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(20 * n_steps, 1);
  for (int k = 0; (4 + 5 * k) < (20 * n_steps); k++) {
    inf_lower_bount(4 + 5 * k, 0) = -25.0;
  }

  NK_low.block(0, 0, 12 * n_steps * 2, 1) = NK_up.block(0, 0, 12 * n_steps * 2, 1);
  NK_low.block(12 * n_steps * 2, 0, 20 * n_steps, 1) = inf_lower_bount;

  // Convert to c_double arrays
  /*std::vector<c_double> vec_up(NK_up.data(), NK_up.data() + NK_up.size());
  std::copy(vec_up.begin(), vec_up.end(), v_NK_up);
  std::vector<c_double> vec_low(NK_low.data(), NK_low.data() + NK_low.size());
  std::copy(vec_low.begin(), vec_low.end(), v_NK_low);*/

  Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&v_NK_up[0], NK_up.size()) = NK_up;
  Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&v_NK_low[0], NK_low.size()) = NK_low;

  return 0;
}

/*
Create the weight matrices P and q in the cost function x^T.P.x + x^T.q of the QP problem
*/
int MPC::create_weight_matrices() {
  int *r_P = new int[size_nz_P];        // row indexes of non-zero values in matrix P
  int *c_P = new int[size_nz_P];        // col indexes of non-zero values in matrix P
  double *v_P = new double[size_nz_P];  // non-zero values in matrix P

  std::fill_n(r_P, size_nz_P, 0);
  std::fill_n(c_P, size_nz_P, 0);
  std::fill_n(v_P, size_nz_P, 0.0);

  // Define weights for the x-x_ref components of the optimization vector
  // Hand-tuning of parameters if you want to give more weight to specific components
  // double w[12] = {10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 10.0f};
  double w[12] = {2.0f, 2.0f, 20.0f, 0.25f, 0.25f, 10.0f, 0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 0.3f};
  /*w[6] = 2.0f * sqrt(w[0]);
  w[7] = 2.0f * sqrt(w[1]);
  w[8] = 2.0f * sqrt(w[2]);
  w[9] = 0.05f * sqrt(w[3]);
  w[10] = 0.05f * sqrt(w[4]);
  w[11] = 0.05f * sqrt(w[5]);*/
  for (int k = 0; k < n_steps; k++) {
    for (int i = 0; i < 12; i++) {
      add_to_P(12 * k + i, 12 * k + i, w[i], r_P, c_P, v_P);
    }
  }

  // Define weights for the force components of the optimization vector
  for (int k = n_steps; k < (2 * n_steps); k++) {
    for (int i = 0; i < 4; i++) {
      add_to_P(12 * k + 3 * i + 0, 12 * k + 3 * i + 0, 1e-5f, r_P, c_P, v_P);
      add_to_P(12 * k + 3 * i + 1, 12 * k + 3 * i + 1, 1e-5f, r_P, c_P, v_P);
      add_to_P(12 * k + 3 * i + 2, 12 * k + 3 * i + 2, 1e-5f, r_P, c_P, v_P);
    }
  }

  // Creation of CSC matrix
  int *icc;                                // row indices
  int *ccc;                                // col indices
  double *acc;                             // coeff values
  int nst = cpt_P;                         // number of non zero elements
  int ncc = st_to_cc_size(nst, r_P, c_P);  // number of CC values
  // int m = 12 * n_steps * 2;                // number of rows
  int n = 12 * n_steps * 2;                // number of columns

  // Get the CC indices.
  icc = (int *)malloc(ncc * sizeof(int));
  ccc = (int *)malloc((n + 1) * sizeof(int));
  st_to_cc_index(nst, r_P, c_P, ncc, n, icc, ccc);

  // Get the CC values.
  acc = st_to_cc_values(nst, r_P, c_P, v_P, ncc, n, icc, ccc);

  // Assign values to the csc object
  P = (csc *)c_malloc(sizeof(csc));
  P->m = 12 * n_steps * 2;
  P->n = 12 * n_steps * 2;
  P->nz = -1;
  P->nzmax = ncc;
  P->x = acc;
  P->i = icc;
  P->p = ccc;

  // Free memory
  delete[] r_P;
  delete[] c_P;
  delete[] v_P;

  // Q is already created filled with zeros
  std::fill_n(Q, size_nz_Q, 0.0);

  // char t_char[1] = {'P'};
  // my_print_csc_matrix(P, t_char);

  return 0;
}

/*
Update the M, N, L and K constraint matrices depending on what happened
*/
int MPC::update_matrices(Eigen::MatrixXd fsteps) {
  /* M need to be updated between each iteration:
   - lever_arms changes since the robot moves
   - I_inv changes if the reference velocity vector is modified
   - footholds need to be enabled/disabled depending on the contact sequence */
  update_ML(fsteps);

  /* N need to be updated between each iteration:
   - X0 changes since the robot moves
   - Xk* changes since X0 is not the same */
  update_NK();

  // L matrix is constant
  // K matrix is constant

  return 0;
}

/*
Update the M and L constaint matrices depending on the current state of the gait

*/
int MPC::update_ML(Eigen::MatrixXd fsteps) {
  int j = 0;
  int k_cum = 0;
  // Iterate over all phases of the gait
  while (gait(j, 0) != 0) {
    for (int k = k_cum; k < (k_cum + gait(j, 0)); k++) {
      // Get inverse of the inertia matrix for time step k
      double c = cos(xref(5, k));
      double s = sin(xref(5, k));
      Eigen::Matrix<double, 3, 3> R;
      R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
      Eigen::Matrix<double, 3, 3> R_gI = R.transpose() * gI * R;
      Eigen::Matrix<double, 3, 3> I_inv = R_gI.inverse();

      // Get skew-symetric matrix for each foothold
      // Eigen::Map<Eigen::Matrix<double, 3, 4>> fsteps_tmp((fsteps.block(j, 1, 1, 12)).data(), 3, 4);
      footholds_tmp = fsteps.block(j, 1, 1, 12);
      // footholds = footholds_tmp.reshaped(3, 4);
      Eigen::Map<Eigen::MatrixXd> footholds_bis(footholds_tmp.data(), 3, 4);

      lever_arms = footholds_bis - (xref.block(0, k, 3, 1)).replicate<1, 4>();
      for (int i = 0; i < 4; i++) {
        B.block(9, 3 * i, 3, 3) = dt * (I_inv * getSkew(lever_arms.col(i)));
      }

      // Replace the coefficient directly in ML.data
      int i_iter = 24 * 4 * k;
      for (int i = 0; i < 12 * 4; i++) {
        ML->x[i_update_B[i] + i_iter] = B(i_x_B[i], i_y_B[i]);
      }
    }

    k_cum += gait(j, 0);
    j++;
  }

  // Construct the activation/desactivation matrix based on the current gait
  construct_S();

  // Update lines to enable/disable forces
  int i_start = 30 * n_steps - 18;
  for (int k = 0; k < 12 * n_steps; k++) {
    ML->x[i_off(k, 0) + i_start] = S_gait(k, 0);
  }

  return 0;
}

/*
Update the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K
*/
int MPC::update_NK() {
  // Matrix g is already created and not changed

  // Reset NK
  NK_up = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(12 * n_steps * 2 + 20 * n_steps, 1);

  // Fill N matrix with g matrices
  for (int k = 0; k < n_steps; k++) {
    NK_up(12 * k + 8, 0) = -g(8, 0);  // only 8-th coeff is non zero
  }

  // Including - A*X0 in the first row of N
  NK_up.block(0, 0, 12, 1) += A * (-x0);

  // Matrix D is already created and not changed
  // Add third term to matrix N
  Eigen::Map<Eigen::MatrixXd> xref_col((xref.block(0, 1, 12, n_steps)).data(), 12 * n_steps, 1);
  NK_up.block(0, 0, 12 * n_steps, 1) += D * xref_col;

  // Update upper bound c_double array (unrequired since Map is just pointers?)
  Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&v_NK_up[0], NK_up.size()) = NK_up;

  // Update lower bound c_double array
  NK_low.block(0, 0, 12 * n_steps * 2, 1) = NK_up.block(0, 0, 12 * n_steps * 2, 1);
  Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&v_NK_low[0], NK_low.size()) = NK_low;

  return 0;
}

/*
Create an initial guess and call the solver to solve the QP problem
*/
int MPC::call_solver(int k) {
  // Initial guess for forces (mass evenly supported by all legs in contact)
  warmxf.block(0, 0, 12 * (n_steps - 1), 1) = x.block(12, 0, 12 * (n_steps - 1), 1);
  warmxf.block(12 * n_steps, 0, 12 * (n_steps - 1), 1) = x.block(12 * (n_steps + 1), 0, 12 * (n_steps - 1), 1);
  warmxf.block(12 * (2 * n_steps - 1), 0, 12, 1) = x.block(12 * n_steps, 0, 12, 1);
  Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&v_warmxf[0], warmxf.size()) = warmxf;

  // Setup the solver (first iteration) then just update it
  if (k == 0)  // Setup the solver with the matrices
  {
    data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = 12 * n_steps * 2;                 // number of variables
    data->m = 12 * n_steps * 2 + 20 * n_steps;  // number of constraints
    data->P = P;             // the upper triangular part of the quadratic cost matrix P in csc format (size n x n)
    data->A = ML;            // linear constraints matrix A in csc format (size m x n)
    data->q = &Q[0];         // dense array for linear part of cost function (size n)
    data->l = &v_NK_low[0];  // dense array for lower bound (size m)
    data->u = &v_NK_up[0];   // dense array for upper bound (size m)

    /*save_csc_matrix(ML, "ML");
    save_csc_matrix(P, "P");
    save_dns_matrix(Q, 12 * n_steps * 2, "Q");
    save_dns_matrix(v_NK_low, 12 * n_steps * 2 + 20 * n_steps, "l");
    save_dns_matrix(v_NK_up, 12 * n_steps * 2 + 20 * n_steps, "u");*/

    // settings->rho = 0.1f;
    // settings->sigma = 1e-6f;
    // settings->max_iter = 4000;
    settings->eps_abs = (float)1e-5;
    settings->eps_rel = (float)1e-5;
    /*settings->eps_prim_inf = 1e-4f;
    settings->eps_dual_inf = 1e-4f;
    settings->alpha = 1.6f;
    settings->delta = 1e-6f;
    settings->polish = 0;
    settings->polish_refine_iter = 3;*/
    settings->adaptive_rho = (c_int)1;
    settings->adaptive_rho_interval = (c_int)200;
    settings->adaptive_rho_tolerance = (float)5.0;
    settings->adaptive_rho_fraction = (float)0.7;
    osqp_setup(&workspce, data, settings);

    /*self.prob.setup(P=self.P, q=self.Q, A=self.ML, l=self.NK_inf, u=self.NK.ravel(), verbose=False)
    self.prob.update_settings(eps_abs=1e-5)
    self.prob.update_settings(eps_rel=1e-5)*/
  } else  // Code to update the QP problem without creating it again
  {
    osqp_update_A(workspce, &ML->x[0], OSQP_NULL, 0);
    osqp_update_bounds(workspce, &v_NK_low[0], &v_NK_up[0]);
    // osqp_warm_start_x(workspce, &v_warmxf[0]);
  }

  // Run the solver to solve the QP problem
  osqp_solve(workspce);
  /*self.sol = self.prob.solve()
  self.x = self.sol.x*/
  // solution in workspce->solution->x

  return 0;
}

/*
Extract relevant information from the output of the QP solver
*/
int MPC::retrieve_result() {
  // Retrieve the "contact forces" part of the solution of the QP problem
  for (int i = 0; i < (n_steps); i++) {
    for (int k = 0; k < 12; k++) {
      x_f_applied(k, i) = (workspce->solution->x)[k + 12*i] + xref(k, 1+i);
      x_f_applied(k + 12, i) = (workspce->solution->x)[12 * (n_steps+i) + k];
    }
  }
  for (int k = 0; k < 12; k++) {
    x_next[k] = (workspce->solution->x)[k];
  }

  /*std::cout << "SOLUTION States" << std::endl;
  for (int k = 0; k < n_steps; k++) {
    for (int i = 0; i < 12; i++) {
      std::cout << (workspce->solution->x)[k * 12 + i] + xref(i, 1 + k) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "END" << std::endl;
  std::cout << "SOLUTION Forces" << std::endl;
  for (int k = 0; k < n_steps; k++) {
    for (int i = 0; i < 12; i++) {
      std::cout << (workspce->solution->x)[12 * n_steps + k * 12 + i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "END" << std::endl;*/

  return 0;
}

/*
Return the latest desired contact forces that have been computed
*/
Eigen::MatrixXd MPC::get_latest_result() { return x_f_applied; }

/*
Return the next predicted state of the base
*/
double *MPC::get_x_next() { return x_next; }

/*
Run function with arrays as input for compatibility between Python and C++
*/
void MPC::run_python(const matXd &xref_py, const matXd &fsteps_py) {
  printf("Trigger bindings \n");
  printf("xref: %f %f %f \n", xref_py(0, 0), xref_py(1, 0), xref_py(2, 0));
  printf("fsteps: %f %f %f \n", fsteps_py(0, 0), fsteps_py(1, 0), fsteps_py(2, 0));

  return;
}

/*
Run one iteration of the whole MPC by calling all the necessary functions (data retrieval,
update of constraint matrices, update of the solver, running the solver, retrieving result)
*/
int MPC::run(int num_iter, const Eigen::MatrixXd &xref_in, const Eigen::MatrixXd &fsteps_in) {
  // Recontruct the gait based on the computed footsteps
  construct_gait(fsteps_in);

  // Retrieve data required for the MPC
  xref = xref_in;
  x0 = xref_in.block(0, 0, 12, 1);

  // Create the constraint and weight matrices used by the QP solver
  // Minimize x^T.P.x + x^T.Q with constraints M.X == N and L.X <= K
  if (num_iter == 0) {
    create_matrices();
  } else {
    update_matrices(fsteps_in);
  }

  // Create an initial guess and call the solver to solve the QP problem
  call_solver(num_iter);

  // Extract relevant information from the output of the QP solver
  retrieve_result();

  return 0;
}

/*
Returns the skew matrix of a 3 by 1 column vector
*/
Eigen::Matrix<double, 3, 3> MPC::getSkew(Eigen::Matrix<double, 3, 1> v) {
  Eigen::Matrix<double, 3, 3> result;
  result << 0.0, -v(2, 0), v(1, 0), v(2, 0), 0.0, -v(0, 0), -v(1, 0), v(0, 0), 0.0;
  return result;
}

/*
Construct an array of size 12*N that contains information about the contact state of feet.
This matrix is used to enable/disable contact forces in the QP problem.
N is the number of time step in the prediction horizon.
*/
int MPC::construct_S() {
  int i = 0;
  int k = 0;

  Eigen::Matrix<int, 20, 5> inv_gait = Eigen::Matrix<int, 20, 5>::Ones() - gait;
  while (gait(i, 0) != 0) {
    // S_gait.block(k*12, 0, gait[i, 0]*12, 1) = (1 - (gait.block(i, 1, 1, 4)).transpose()).replicate<gait[i, 0], 1>()
    // not finished;
    for (int a = 0; a < gait(i, 0); a++) {
      for (int b = 0; b < 4; b++) {
        for (int c = 0; c < 3; c++) {
          S_gait(k * 12 + 12 * a + 3 * b + c, 0) = inv_gait(i, 1 + b);
        }
      }
    }
    k += gait(i, 0);
    i++;
  }

  return 0;
}

/*
Reconstruct the gait matrix based on the fsteps matrix since only the last one is received by the MPC
*/
int MPC::construct_gait(Eigen::MatrixXd fsteps_in) {
  // First column is identical
  gait.col(0) = fsteps_in.col(0).cast<int>();

  int k = 0;
  while (gait(k, 0) != 0) {
    for (int i = 0; i < 4; i++) {
      if (fsteps_in(k, 1 + i * 3) == 0.0) {
        gait(k, 1 + i) = 0;
      } else {
        gait(k, 1 + i) = 1;
      }
    }
    k++;
  }
  gait.row(k) << 0, 0, 0, 0, 0;
  return 0;
}

/*
Set all the parameters of the OSQP solver
*/
int set_settings() { return 0; }

void MPC::my_print_csc_matrix(csc *M, const char *name) {
  c_int j, i, row_start, row_stop;
  c_int k = 0;

  // Print name
  c_print("%s :\n", name);

  for (j = 0; j < M->n; j++) {
    row_start = M->p[j];
    row_stop = M->p[j + 1];

    if (row_start == row_stop)
      continue;
    else {
      for (i = row_start; i < row_stop; i++) {
        int a = (int)M->i[i];
        int b = (int)j;
        double c = M->x[k++];
        if ((a >= 12 * (n_steps - 1)) && (a < 12 * (n_steps - 1) + 24) && (b >= 12 * (n_steps - 1)) &&
            (b < 12 * (n_steps - 1) * 2)) {
          c_print("\t%3u [%3u,%3u] = %.3g\n", k - 1, a, b - 12 * n_steps, c);
        }
      }
    }
  }
}

void MPC::save_csc_matrix(csc *M, std::string filename) {
  c_int j, i, row_start, row_stop;
  c_int k = 0;

  // Open file
  std::ofstream myfile;
  myfile.open(filename + ".csv");

  for (j = 0; j < M->n; j++) {
    row_start = M->p[j];
    row_stop = M->p[j + 1];

    if (row_start == row_stop)
      continue;
    else {
      for (i = row_start; i < row_stop; i++) {
        int a = (int)M->i[i];
        int b = (int)j;
        double c = M->x[k++];
        myfile << a << "," << b << "," << c << "\n";
      }
    }
  }
  myfile.close();
}

void MPC::save_dns_matrix(double *M, int size, std::string filename) {
  // Open file
  std::ofstream myfile;
  myfile.open(filename + ".csv");

  for (int j = 0; j < size; j++) {
    myfile << j << "," << 0 << "," << M[j] << "\n";
  }

  myfile.close();
}

Eigen::MatrixXd MPC::get_gait() {
  // Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(20, 5);
  // tmp.block(0, 0, 20, 5) = gait.block(0, 0, 20, 5);
  Eigen::MatrixXd tmp;
  tmp = gait.cast<double>();
  return tmp;
}

Eigen::MatrixXd MPC::get_Sgait() {
  // Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(12 * n_steps, 1);
  // tmp.col(0) = S_gait.col(0);
  Eigen::MatrixXd tmp;
  tmp = S_gait.cast<double>();
  return tmp;
}