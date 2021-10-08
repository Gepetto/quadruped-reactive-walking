#include "qrw/QPWBC.hpp"

QPWBC::QPWBC() {
  /*
  Constructor of the QP solver. Initialization of matrices
  */

  // Slipping constraints
  Eigen::Matrix<double, 5, 3> SC = Eigen::Matrix<double, 5, 3>::Zero();
  int a[9] = {0, 1, 2, 3, 0, 1, 2, 3, 4};
  int b[9] = {0, 0, 1, 1, 2, 2, 2, 2, 2};
  double c[9] = {1.0, -1.0, 1.0, -1.0, -mu, -mu, -mu, -mu, -1};
  for (int i = 0; i <= 8; i++) {
    SC(a[i], b[i]) = -c[i];
  }

  // Add slipping constraints to inequality matrix
  for (int i = 0; i < 4; i++) {
    G.block(5 * i, 3 * i, 5, 3) = SC;
  }

  // Set OSQP settings to default
  osqp_set_default_settings(settings);
}

void QPWBC::initialize(Params &params) {
  params_ = &params;
  Q1 = params.Q1 * Eigen::Matrix<double, 6, 6>::Identity();
  Q2 = params.Q2 * Eigen::Matrix<double, 12, 12>::Identity();

  // Set the lower and upper limits of the box
  Fz_max = params_->Fz_max;
  Fz_min = params_->Fz_min;
  std::fill_n(v_NK_up, size_nz_NK, +std::numeric_limits<double>::infinity());
  std::fill_n(v_NK_low, size_nz_NK, -std::numeric_limits<double>::infinity());
}

int QPWBC::create_matrices(const Eigen::Matrix<double, 12, 6> &Jc, const Eigen::Matrix<double, 12, 1> &f_cmd,
                           const Eigen::Matrix<double, 6, 1> &RNEA) {
  // Create the constraint matrices
  create_ML();
  create_NK(Jc.transpose(), f_cmd, RNEA);

  // Create the weight matrices
  create_weight_matrices();

  return 0;
}

inline void QPWBC::add_to_ML(int i, int j, double v, int *r_ML, int *c_ML, double *v_ML) {
  r_ML[cpt_ML] = i;  // row index
  c_ML[cpt_ML] = j;  // column index
  v_ML[cpt_ML] = v;  // value of coefficient
  cpt_ML++;          // increment the counter
}

inline void QPWBC::add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P) {
  r_P[cpt_P] = i;  // row index
  c_P[cpt_P] = j;  // column index
  v_P[cpt_P] = v;  // value of coefficient
  cpt_P++;         // increment the counter
}

int QPWBC::create_ML() {
  int *r_ML = new int[size_nz_ML];        // row indexes of non-zero values in matrix ML
  int *c_ML = new int[size_nz_ML];        // col indexes of non-zero values in matrix ML
  double *v_ML = new double[size_nz_ML];  // non-zero values in matrix ML

  std::fill_n(r_ML, size_nz_ML, 0);
  std::fill_n(c_ML, size_nz_ML, 0);
  std::fill_n(v_ML, size_nz_ML, 0.0);

  // ML is the identity matrix of size 12
  // Friction cone constraints
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 12; j++) {
      if (G(i, j) != 0.0) {add_to_ML(i, 6 + j, G(i, j), r_ML, c_ML, v_ML);}
    }
  }
  // Dynamics equation constraints
  // Filling with mass matrix (all coeffs to 1.0 to initialize that part as dense)
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      add_to_ML(20 + i, j, 1.0, r_ML, c_ML, v_ML);  // 1.0 replaces M(i, j)
    }
  }
  // Filling with - contact Jacobian transposed (all coeffs to 1.0 to initialize that part as dense)
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 12; j++) {
      add_to_ML(20 + i, 6 + j, 1.0, r_ML, c_ML, v_ML);  // 1.0 replaces -JcT(i, j)
    }
  }

  // Creation of CSC matrix
  int *icc;                                  // row indices
  int *ccc;                                  // col indices
  double *acc;                               // coeff values
  int nst = cpt_ML;                          // number of non zero elements
  int ncc = st_to_cc_size(nst, r_ML, c_ML);  // number of CC values
  // int m = 20;   // number of rows
  int n = 18;  // number of columns

  // std::cout << "Number of CC values: " << ncc << std::endl;

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
  ML->m = (20 + 6);
  ML->n = 18;
  ML->nz = -1;
  ML->nzmax = ncc;
  ML->x = acc;
  ML->i = icc;
  ML->p = ccc;

  // Free memory
  delete[] r_ML;
  delete[] c_ML;
  delete[] v_ML;

  return 0;
}

int QPWBC::create_NK(const Eigen::Matrix<double, 6, 12> &JcT, const Eigen::Matrix<double, 12, 1> &f_cmd,
                     const Eigen::Matrix<double, 6, 1> &RNEA) {

  // Fill upper bound of the friction cone contraints
  for (int i = 0; i < 4; i++) {
    v_NK_up[5 * i + 0] = std::numeric_limits<double>::infinity();
    v_NK_up[5 * i + 1] = std::numeric_limits<double>::infinity();
    v_NK_up[5 * i + 2] = std::numeric_limits<double>::infinity();
    v_NK_up[5 * i + 3] = std::numeric_limits<double>::infinity();
    v_NK_up[5 * i + 4] = Fz_max - f_cmd(3 * i + 2, 0);
  }

  // Fill lower bound of the friction cone contraints
  for (int i = 0; i < 4; i++) {
    v_NK_low[5 * i + 0] = f_cmd(3 * i + 0, 0) - mu * f_cmd(3 * i + 2, 0);
    v_NK_low[5 * i + 1] = - f_cmd(3 * i + 0, 0) - mu * f_cmd(3 * i + 2, 0);
    v_NK_low[5 * i + 2] = f_cmd(3 * i + 1, 0) - mu * f_cmd(3 * i + 2, 0);
    v_NK_low[5 * i + 3] = - f_cmd(3 * i + 1, 0) - mu * f_cmd(3 * i + 2, 0);
    v_NK_low[5 * i + 4] = Fz_min - f_cmd(3 * i + 2, 0);
  }

  // Fill the remaining 6 lines for the dynamics
  Eigen::Matrix<double, 6, 1> dyn_cons = JcT * f_cmd - RNEA;
  for (int i = 0; i < 6; i++) {
    v_NK_up[20 + i] = dyn_cons(i, 0);
    v_NK_low[20 + i] = dyn_cons(i, 0);
  }

  return 0;
}

int QPWBC::create_weight_matrices() {
  int *r_P = new int[size_nz_P];        // row indexes of non-zero values in matrix P
  int *c_P = new int[size_nz_P];        // col indexes of non-zero values in matrix P
  double *v_P = new double[size_nz_P];  // non-zero values in matrix P

  std::fill_n(r_P, size_nz_P, 0);
  std::fill_n(c_P, size_nz_P, 0);
  std::fill_n(v_P, size_nz_P, 0.0);

  // Fill P with 1.0 so that the sparse creation process considers that all coeffs
  // can have a non zero value
  for (int i = 0; i < 6; i++) {
      add_to_P(i, i, Q1(i, i), r_P, c_P, v_P);
  }
  for (int i = 0; i < 12; i++) {
      add_to_P(6 + i, 6+i, Q2(i, i), r_P, c_P, v_P);
  }

  // Creation of CSC matrix
  int *icc;                                // row indices
  int *ccc;                                // col indices
  double *acc;                             // coeff values
  int nst = cpt_P;                         // number of non zero elements
  int ncc = st_to_cc_size(nst, r_P, c_P);  // number of CC values
  // int m = 12;                // number of rows
  int n = 18;  // number of columns

  // std::cout << "Number of CC values: " << ncc << std::endl;

  // Get the CC indices.
  icc = (int *)malloc(ncc * sizeof(int));
  ccc = (int *)malloc((n + 1) * sizeof(int));
  st_to_cc_index(nst, r_P, c_P, ncc, n, icc, ccc);

  // Get the CC values.
  acc = st_to_cc_values(nst, r_P, c_P, v_P, ncc, n, icc, ccc);

  // Assign values to the csc object
  P = (csc *)c_malloc(sizeof(csc));
  P->m = 18;
  P->n = 18;
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

  return 0;
}

int QPWBC::update_matrices(const Eigen::Matrix<double, 6, 6> &M, const Eigen::Matrix<double, 12, 6> &Jc,
                           const Eigen::Matrix<double, 12, 1> &f_cmd, const Eigen::Matrix<double, 6, 1> &RNEA) {
  // Updating M and L matrices
  update_ML(M, Jc.transpose());

  // Updating N and K matrices
  create_NK(Jc.transpose(), f_cmd, RNEA);  // We can use the create function to update NK

  // Weight matrices do not need to be update

  /*char t_char[1] = {'P'};
  my_print_csc_matrix(P, t_char);

  t_char[0] = 'M';
  my_print_csc_matrix(ML, t_char);

  std::cout << "Q" << std::endl;
  for (int j = 0; j < 18; j++) {
    std::cout << Q[j] << " ";
  }
  std::cout << std::endl;

  std::cout << "vlow" << std::endl;
  for (int j = 0; j < 26; j++) {
    std::cout << v_NK_low[j] << " ";
  }
  std::cout << std::endl;

  std::cout << "vup" << std::endl;
  for (int j = 0; j < 26; j++) {
    std::cout << v_NK_up[j] << " ";
  }
  std::cout << std::endl;*/

  return 0;
}

int QPWBC::update_ML(const Eigen::Matrix<double, 6, 6> &M, const Eigen::Matrix<double, 6, 12> &JcT) {

  // Update the part of ML that contains the dynamics constraint
  // Coefficients are stored in column order and we want to update the block (20, 0, 6, 18)
  // [0  fric
  //  M -JcT] with fric having [2 2 5 2 2 5 2 2 5 2 2 5] non zeros coefficient for each one of its 12 columns
  
  // Update M, no need to be careful because there is only zeros coefficients above M
  for (int j = 0; j < 6; j++) {
    for (int i = 0; i < 6; i++) {
      ML->x[6 * j + i] = M(i, j);
    }
  }

  // Update -JcT, need to be careful because there are non zeros coefficients before
  // M represents 36 non zero coefficients + [2 2 5 2 2 5 2 2 5 2 2 5] non zeros for friction cone
  for (int j = 0; j < 12; j++) {
    for (int i = 0; i < 6; i++) {
      ML->x[36 + fric_nz[j] + 6 * j + i] = -JcT(i, j);
    }
  }

  return 0;
}

int QPWBC::call_solver() {
  // Setup the solver (first iteration) then just update it
  if (not initialized)  // Setup the solver with the matrices
  {
    data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = 18;        // number of variables
    data->m = 26;        // number of constraints
    data->P = P;         // the upper triangular part of the quadratic cost matrix P in csc format (size n x n)
    data->A = ML;        // linear constraints matrix A in csc format (size m x n)
    data->q = Q;         // dense array for linear part of cost function (size n)
    data->l = v_NK_low;  // dense array for lower bound (size m)
    data->u = v_NK_up;   // dense array for upper bound (size m)

    /*std::cout << data->l << std::endl;
    std::cout << data->A << std::endl;
    std::cout << data->u << std::endl;*/

    // Tuning parameters of the OSQP solver
    // settings->rho = 0.1f;
    // settings->sigma = 1e-6f;
    // settings->max_iter = 4000;
    // settings->eps_abs = (float)1e-5;
    // settings->eps_rel = (float)1e-5;
    /*settings->eps_prim_inf = 1e-4f;
    settings->eps_dual_inf = 1e-4f;
    settings->alpha = 1.6f;
    settings->delta = 1e-6f;
    settings->polish = 0;
    settings->polish_refine_iter = 3;*/
    settings->adaptive_rho = (c_int)1;
    settings->adaptive_rho_interval = (c_int)200;
    settings->adaptive_rho_tolerance = (float)5.0;
    // settings->adaptive_rho_fraction = (float)0.7;
    settings->verbose = true;
    osqp_setup(&workspce, data, settings);

    initialized = true;
  } else  // Code to update the QP problem without creating it again
  {
    // Update P matrix of the OSQP solver
    // osqp_update_P(workspce, &P->x[0], OSQP_NULL, 0);

    // Update Q matrix of the OSQP solver
    // osqp_update_lin_cost(workspce, &Q[0]);

    // Update constraint matrix
    osqp_update_A(workspce, &ML->x[0], OSQP_NULL, 0);

    // Update lower and upper bounds
    osqp_update_bounds(workspce, &v_NK_low[0], &v_NK_up[0]);

    // Update upper bound of the OSQP solver
    // osqp_update_upper_bound(workspce, &v_NK_up[0]);
    // osqp_update_lower_bound(workspce, &v_NK_low[0]);
  }

  // Run the solver to solve the QP problem
  osqp_solve(workspce);

  // solution in workspce->solution->x

  return 0;
}

int QPWBC::retrieve_result(const Eigen::Matrix<double, 6, 1> &ddq_cmd, const Eigen::Matrix<double, 12, 1> &f_cmd) {
  // Retrieve the solution of the QP problem
  for (int k = 0; k < 6; k++) {
    ddq_res(k, 0) = (workspce->solution->x)[k];
  }
  for (int k = 0; k < 12; k++) {
    f_res(k, 0) = (workspce->solution->x)[6 + k];
  }

  // Computing delta ddq with delta f
  // ddq_res = A * f_res + gamma;

  // Adding reference contact forces to delta f
  // f_res += f_cmd;

  // Adding reference acceleration to delta ddq
  // ddq_res += ddq_cmd;

  /*std::cout << "f_cmd: " << std::endl << f_cmd.transpose() << std::endl;
  std::cout << "f_res: " << std::endl << f_res.transpose() << std::endl;
  std::cout << "ddq_cmd: " << std::endl << ddq_cmd.transpose() << std::endl;
  std::cout << "ddq_res: " << std::endl << ddq_res.transpose() << std::endl;*/

  return 0;
}

/*
Getters
*/
Eigen::MatrixXd QPWBC::get_f_res() { return f_res; }
Eigen::MatrixXd QPWBC::get_ddq_res() { return ddq_res; }
Eigen::MatrixXd QPWBC::get_H() {
  Eigen::MatrixXd Hxd = Eigen::MatrixXd::Zero(12, 12);
  Hxd = H;
  return Hxd;
}

int QPWBC::run(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &ddq_cmd,
               const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA, const Eigen::MatrixXd &k_contact) {
  

  // Create the constraint and weight matrices used by the QP solver
  // Minimize x^T.P.x + 2 x^T.Q with constraints M.X == N and L.X <= K
  if (not initialized) {
    create_matrices(Jc, f_cmd, RNEA);
    // std::cout << G << std::endl;
  }

  // Compute the different matrices involved in the box QP
  // compute_matrices(M, Jc, f_cmd, RNEA);

  // Update M, L, N and K matrices (M.X == N and L.X <= K)
  update_matrices(M, Jc, f_cmd, RNEA);

  // Update P and Q matrices of the cost function xT P x + 2 xT g
  //update_PQ();

  Eigen::Matrix<double, 20, 1> Gf = G * f_cmd;

  for (int i = 0; i < G.rows(); i++) {
    v_NK_low[i] = -Gf(i, 0) + params_->Fz_min;
    v_NK_up[i] = -Gf(i, 0) + params_->Fz_max;
  }

  // Limit max force when contact is activated
  /* const double k_max = 60.0;
  for (int i = 0; i < 4; i++) {
    if (k_contact(0, i) < k_max) {
      v_NK_up[5*i+4] -= params_->Fz_max * (1.0 - k_contact(0, i) / k_max);
    }
  }*/
  /*else if (k_contact(0, i) == (k_max+10))
  {
    //char t_char[1] = {'M'};
    //cc_print( (data->A)->m, (data->A)->n, (data->A)->nzmax, (data->A)->i, (data->A)->p, (data->A)->x, t_char);
    std::cout << " ### " << k_contact(0, i) << std::endl;

    for (int i = 0; i < data->m; i++) {
      std::cout << data->l[i] << " | " << data->u[i] << " | " << f_cmd(i, 0) << std::endl;
    }
  }*/

  //}

  // Create an initial guess and call the solver to solve the QP problem
  call_solver();

  // Extract relevant information from the output of the QP solver
  retrieve_result(ddq_cmd, f_cmd);

  /*Eigen::MatrixXd df = Eigen::MatrixXd::Zero(12, 1);
  df(0, 0) = 0.01;
  df(1, 0) = 0.01;
  df(2, 0) = 0.01;
  df(9, 0) = 0.01;
  df(10, 0) = 0.01;
  df(11, 0) = 0.01;
  std::cout << 0.5 * f_res.transpose() * H * f_res + f_res.transpose() * g << std::endl;
  std::cout << 0.5 * (f_res-df).transpose() * H * (f_res-df) + (f_res-df).transpose() * g << std::endl;
  std::cout << 0.5 * (f_res+df).transpose() * H * (f_res+df) + (f_res+df).transpose() * g << std::endl;

  std::cout << "A:" << std::endl << A << std::endl << "--" << std::endl;
  std::cout << "Xf:" << std::endl << (X * f_cmd) << std::endl << "--" << std::endl;
  std::cout << "RNEA:" << std::endl << RNEA << std::endl << "--" << std::endl;
  std::cout << "B:" << std::endl << gamma << std::endl << "--" << std::endl;
  std::cout << "AT Q1:" << std::endl << A.transpose() * Q1 << std::endl << "--" << std::endl;
  std::cout << "g:" << std::endl << g << std::endl << "--" << std::endl;
  std::cout << "H:" << std::endl << H << std::endl << "--" << std::endl;*/

  return 0;
}

void QPWBC::my_print_csc_matrix(csc *M, const char *name) {
  c_int j, i, row_start, row_stop;
  c_int k = 0;

  Eigen::Matrix<double, 26, 18> Ma = Eigen::Matrix<double, 26, 18>::Zero();

  // Print name
  printf("%s :\n", name);

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
        printf("\t%3u [%3u,%3u] = %.3g\n", k - 1, a, b, c);
        Ma(a, b) = c;
      }
    }
  }
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  std::cout << Ma << std::endl;
}

void QPWBC::save_csc_matrix(csc *M, std::string filename) {
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

void QPWBC::save_dns_matrix(double *M, int size, std::string filename) {
  // Open file
  std::ofstream myfile;
  myfile.open(filename + ".csv");

  for (int j = 0; j < size; j++) {
    myfile << j << "," << 0 << "," << M[j] << "\n";
  }

  myfile.close();
}

void QPWBC::compute_matrices(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd,
                             const Eigen::MatrixXd &RNEA) {
  Y = M.block(0, 0, 6, 6);
  X = Jc.block(0, 0, 12, 6).transpose();
  Yinv = pseudoInverse(Y);
  A = Yinv * X;
  gamma = Yinv * ((X * f_cmd) - RNEA);
  H = A.transpose() * Q1 * A + Q2;
  g = A.transpose() * Q1 * gamma;

  /*std::cout << "X" << std::endl;
  std::cout << X << std::endl;
  std::cout << "Yinv" << std::endl;
  std::cout << Yinv << std::endl;
  std::cout << "A" << std::endl;
  std::cout << A << std::endl;
  std::cout << "gamma" << std::endl;
  std::cout << gamma << std::endl;
  std::cout << "Q1" << std::endl;
  std::cout << Q1 << std::endl;
  std::cout << "A.transpose() * Q1" << std::endl;
  std::cout << A.transpose() * Q1 << std::endl;*/
  /*std::cout << "g" << std::endl;
  std::cout << g << std::endl;
  std::cout << "H" << std::endl;
  std::cout << H << std::endl;*/
}

void QPWBC::update_PQ() {
  // Update P matrix of min xT P x + 2 xT Q
  int cpt = 0;
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j <= i; j++) {
      P->x[cpt] = H(j, i);
      cpt++;
    }
  }

  // Update Q matrix of min xT P x + 2 xT Q
  for (int i = 0; i < 12; i++) {
    Q[i] = g(i, 0);
  }

  // std::cout << "Eigenvalues" << H.eigenvalues() << std::endl;

  /*char t_char[1] = {'P'};
  my_print_csc_matrix(P, t_char);*/
}

WbcWrapper::WbcWrapper()
    : M_(Eigen::Matrix<double, 18, 18>::Zero()),
      Jc_(Eigen::Matrix<double, 12, 6>::Zero()),
      k_since_contact_(Eigen::Matrix<double, 1, 4>::Zero()),
      bdes_(Vector7::Zero()),
      qdes_(Vector12::Zero()),
      vdes_(Vector12::Zero()),
      tau_ff_(Vector12::Zero()),
      q_wbc_(Vector19::Zero()),
      dq_wbc_(Vector18::Zero()),
      ddq_cmd_(Vector18::Zero()),
      f_with_delta_(Vector12::Zero()),
      ddq_with_delta_(Vector18::Zero()),
      nle_(Vector6::Zero()),
      log_feet_pos_target(Matrix34::Zero()),
      log_feet_vel_target(Matrix34::Zero()),
      log_feet_acc_target(Matrix34::Zero()),
      k_log_(0) {}

void WbcWrapper::initialize(Params &params) {
  // Params store parameters
  params_ = &params;

  // Path to the robot URDF (TODO: Automatic path)
  const std::string filename =
      std::string("/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf");

  // Build model from urdf (base is not free flyer)
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(), model_, false);

  // Construct data from model
  data_ = pinocchio::Data(model_);

  // Update all the quantities of the model
  VectorN q_tmp = VectorN::Zero(model_.nq);
  q_tmp(6, 0) = 1.0;  // Quaternion (0, 0, 0, 1)
  pinocchio::computeAllTerms(model_, data_, q_tmp, VectorN::Zero(model_.nv));

  // Initialize inverse kinematic and box QP solvers
  invkin_ = new InvKin();
  invkin_->initialize(params);
  box_qp_ = new QPWBC();
  box_qp_->initialize(params);

  // Initialize quaternion
  q_wbc_(6, 0) = 1.0;

  // Initialize joint positions
  qdes_.tail(12) = Vector12(params_->q_init.data());

  // Compute the upper triangular part of the joint space inertia matrix M by using the Composite Rigid Body Algorithm
  // Result is stored in data_.M
  pinocchio::crba(model_, data_, q_wbc_);

  // Make mass matrix symetric
  data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();
}

void WbcWrapper::compute(VectorN const &q, VectorN const &dq, VectorN const &f_cmd, MatrixN const &contacts,
                         MatrixN const &pgoals, MatrixN const &vgoals, MatrixN const &agoals, VectorN const &xgoals) {

  if (f_cmd.rows() != 12) {
    throw std::runtime_error("f_cmd should be a vector of size 12");
  }

  //  Update nb of iterations since contact
  k_since_contact_ += contacts;                                // Increment feet in stance phase
  k_since_contact_ = k_since_contact_.cwiseProduct(contacts);  // Reset feet in swing phase

  // Store target positions, velocities and acceleration for logging purpose
  log_feet_pos_target = pgoals;
  log_feet_vel_target = vgoals;
  log_feet_acc_target = agoals;

  // Retrieve configuration data
  q_wbc_.head(3) = q.head(3);
  q_wbc_.block(3, 0, 4, 1) = pinocchio::SE3::Quaternion(pinocchio::rpy::rpyToMatrix(q(3, 0), q(4, 0), q(5, 0))).coeffs();  // Roll, Pitch
  q_wbc_.tail(12) = q.tail(12);  // Encoders

  // Retrieve velocity data
  dq_wbc_ = dq;

  // Compute the upper triangular part of the joint space inertia matrix M by using the Composite Rigid Body Algorithm
  // Result is stored in data_.M
  pinocchio::crba(model_, data_, q_wbc_);

  // Make mass matrix symetric
  data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();

  // Compute Inverse Kinematics
  invkin_->run_InvKin(q_wbc_, dq_wbc_, contacts, pgoals.transpose(), vgoals.transpose(), agoals.transpose(), xgoals);
  ddq_cmd_ = invkin_->get_ddq_cmd();

  // TODO: Check if we can save time by switching MatrixXd to defined sized vector since they are
  // not called from python anymore

  // Retrieve feet jacobian
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i)) {
      Jc_.block(3 * i, 0, 3, 6) = invkin_->get_Jf().block(3 * i, 0, 3, 6);
    } else {
      Jc_.block(3 * i, 0, 3, 6).setZero();
    }
  }

  Eigen::Matrix<double, 12, 12> Jc_u_ = Eigen::Matrix<double, 12, 12>::Zero();
  for (int i = 0; i < 4; i++) {
    if (contacts(0, i)) {
      Jc_u_.block(3 * i, 0, 3, 12) = invkin_->get_Jf().block(3 * i, 6, 3, 12);
    } else {
      Jc_u_.block(3 * i, 0, 3, 12).setZero();
    }
  }

  // Compute the inverse dynamics, aka the joint torques according to the current state of the system,
  // the desired joint accelerations and the external forces, using the Recursive Newton Euler Algorithm.
  // Result is stored in data_.tau
  Vector18 ddq_test = Vector18::Zero();
  ddq_test.head(6) = ddq_cmd_.head(6);
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_test);
  Vector6 RNEA_without_joints = data_.tau.head(6);
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, VectorN::Zero(model_.nv));
  Vector6 RNEA_NLE = data_.tau.head(6);
  RNEA_NLE(2, 0) -= 24.8;

  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_cmd_);

  /*std::cout << "M inertia" << std::endl;
  std::cout << data_.M << std::endl;*/
  /*std::cout << "Jc" << std::endl;
  std::cout << Jc_ << std::endl;
  std::cout << "f_cmd" << std::endl;
  std::cout << f_cmd << std::endl;
  std::cout << "rnea" << std::endl;
  std::cout << data_.tau.head(6) << std::endl;
  std::cout << "k_since" << std::endl;
  std::cout << k_since_contact_ << std::endl;*/

  //std::cout << "Force compensation " << std::endl;
  Vector12 f_compensation = pseudoInverse(Jc_.transpose()) * (data_.tau.head(6) - RNEA_without_joints + RNEA_NLE);
  /*for (int i = 0; i < 4; i++) {
    f_compensation(3*i+2, 0) = 0.0;
  }*/
  //std::cout << f_compensation << std::endl;

  //std::cout << "agoals " << std::endl << agoals << std::endl; 
  //std::cout << "ddq_cmd_bis " << std::endl << ddq_cmd_.transpose() << std::endl; 

  // Solve the QP problem
  box_qp_->run(data_.M, Jc_, ddq_cmd_, f_cmd + f_compensation, data_.tau.head(6),
               k_since_contact_);

  // Add to reference quantities the deltas found by the QP solver
  f_with_delta_ = f_cmd + f_compensation + box_qp_->get_f_res();
  ddq_with_delta_.head(6) = ddq_cmd_.head(6) + box_qp_->get_ddq_res();
  ddq_with_delta_.tail(12) = ddq_cmd_.tail(12);

  Vector6 left = data_.M.block(0, 0, 6, 6) * box_qp_->get_ddq_res() - Jc_.transpose() * box_qp_->get_f_res();
  Vector6 right = - data_.tau.head(6) + Jc_.transpose() * (f_cmd + f_compensation);
  //std::cout << "RNEA: " << std::endl << data_.tau.head(6).transpose() << std::endl;
  Vector6 tmp_RNEA = data_.tau.head(6);
  //std::cout << "left: " << std::endl << left.transpose() << std::endl;
  //std::cout << "right: " << std::endl << right.transpose() << std::endl;
  //std::cout << "M: " << std::endl << data_.M.block(0, 0, 6, 6) << std::endl;
  
  //std::cout << "JcT: " << std::endl << Jc_.transpose() << std::endl;
  //std::cout << "M: " << std::endl << data_.M.block(0, 0, 3, 18) << std::endl;

  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, VectorN::Zero(model_.nv));
  Vector6 tmp_NLE = data_.tau.head(6);
  /*std::cout << "NLE: " << std::endl << data_.tau.head(6).transpose() << std::endl;
  std::cout << "M DDQ: " << std::endl << (tmp_RNEA - data_.tau.head(6)).transpose() << std::endl;
  std::cout << "JcT f_cmd: " << std::endl << (Jc_.transpose() * (f_cmd + f_compensation)).transpose() << std::endl;*/

  // std::cout << "Gravity ?: " << std::endl <<  (data_.M * ddq_cmd_ - data_.tau.head(6)).transpose() << std::endl;

  Mddq = tmp_RNEA - data_.tau.head(6);
  NLE = data_.tau.head(6);
  JcTf = Jc_.transpose() * (f_cmd + f_compensation);

  nle_ = data_.tau.head(6);

  // Compute joint torques from contact forces and desired accelerations
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_with_delta_);

  /*std::cout << "NLE Delta: " << std::endl << tmp_NLE.transpose() << std::endl;
  std::cout << "M DDQ Delta: " << std::endl << (data_.tau.head(6) - tmp_NLE).transpose() << std::endl;
  std::cout << "JcT f_cmd Delta: " << std::endl << (Jc_.transpose() * f_with_delta_).transpose() << std::endl;*/

  /*std::cout << "rnea delta" << std::endl;
  std::cout << data_.tau.tail(12) << std::endl;
  std::cout << "ddq del" << std::endl;
  std::cout << ddq_with_delta_ << std::endl;
  std::cout << "f del" << std::endl;
  std::cout << f_with_delta_ << std::endl;
  std::cout << "Jf" << std::endl;
  std::cout << invkin_->get_Jf().block(0, 6, 12, 12).transpose() << std::endl;*/

  // std::cout << " -- " << std::endl << invkin_->get_Jf().block(0, 6, 12, 12) << std::endl << " -- " << std::endl << Jc_u_ << std::endl;

  tau_ff_ = data_.tau.tail(12) - Jc_u_.transpose() * f_with_delta_;
  // tau_ff_ = - Jc_u_.transpose() * (f_cmd + f_compensation);

  // Retrieve desired positions and velocities
  vdes_ = invkin_->get_dq_cmd().tail(12);

  // std::cout << "GET:" << invkin_->get_q_cmd() << std::endl;

  qdes_ = invkin_->get_q_cmd().tail(12);
  bdes_ = invkin_->get_q_cmd().head(7);

  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_with_delta_);
  Vector6 tau_left = data_.tau.head(6); // data_.M * ddq_with_delta_.head(6) - Jc_.transpose() * f_with_delta_; //
  Vector6 tau_right = Jc_.transpose() * f_with_delta_;
  // std::cout << "tau_left: " << std::endl << tau_left.transpose() << std::endl;
  // std::cout << "tau_right: " << std::endl << tau_right.transpose() << std::endl;

  Mddq_out = data_.tau.head(6) - NLE;
  JcTf_out = Jc_.transpose() * f_with_delta_;

  /*std::cout << vdes_.transpose() << std::endl;
  std::cout << qdes_.transpose() << std::endl;*/

  /*std::cout << "----" << std::endl;
  std::cout << qdes_.transpose() << std::endl;
  std::cout << vdes_.transpose() << std::endl;
  std::cout << tau_ff_.transpose() << std::endl;*/

  // Compute joint torques from contact forces and desired accelerations
  /*Vector18 ddq_test = Vector18::Zero();
  ddq_test.head(6) = ddq_with_delta_.head(6);
  pinocchio::rnea(model_, data_, q_wbc_, dq_wbc_, ddq_test);
  std::cout << "M DDQ Delta Bis: " << std::endl << (data_.tau.head(6) - tmp_NLE).transpose() << std::endl;*/

  // Increment log counter
  k_log_++;
}
