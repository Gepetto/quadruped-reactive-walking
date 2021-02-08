#include "quadruped-reactive-walking/QPWBC.hpp"


QPWBC::QPWBC() {
  /* 
  Constructor of the QP solver. Initialization of matrices 
  */

  // Slipping constraints
  Eigen::Matrix<double, 5, 3> SC;
  int a[9] = {0, 1, 2, 3, 0, 1, 2, 3, 4};
  int b[9] = {0, 0, 1, 1, 2, 2, 2, 2, 2};
  double c[9] = {1.0, -1.0, 1.0, -1.0, -mu, -mu, -mu, -mu, -1};
  for (int i = 0; i <= 8; i++) {
    SC(a[i], b[i]) = -c[i];
  }

  // Add slipping constraints to inequality matrix
  for (int i = 0; i < 4; i++) {
    G.block(5*i, 3*i, 5, 3) = SC;
  }

  // Set the lower and upper limits of the box
  std::fill_n(v_NK_up, size_nz_NK, 25.0);
  std::fill_n(v_NK_low, size_nz_NK, 0.0);

  // Set OSQP settings to default
  osqp_set_default_settings(settings);

}

int QPWBC::create_matrices() {
  /*
  Create the constraint matrices (M.X = N and L.X <= K)
  Create the weight matrices P and Q (cost 1/2 x^T * P * X + X^T * Q)
  */

  // Create the constraint matrices
  create_ML();

  // Create the weight matrices
  create_weight_matrices();

  return 0;
}

inline void QPWBC::add_to_ML(int i, int j, double v, int *r_ML, int *c_ML, double *v_ML) {
  /*
  Add a new non-zero coefficient to the ML matrix by filling the triplet r_ML / c_ML / v_ML
  
  Args:
    - i (int): row index of the new entry
    - j (int): column index of the new entry
    - v (double): value of the new entry
    - r_ML (int*): table that contains row indexes
    - c_ML (int*): table that contains column indexes
    - v_ML (double*): table that contains values
  */
  
  r_ML[cpt_ML] = i;  // row index
  c_ML[cpt_ML] = j;  // column index
  v_ML[cpt_ML] = v;  // value of coefficient
  cpt_ML++;          // increment the counter
}

inline void QPWBC::add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P) {
  /*
  Add a new non-zero coefficient to the P matrix by filling the triplet r_P / c_P / v_P
  
  Args:
    - i (int): row index of the new entry
    - j (int): column index of the new entry
    - v (double): value of the new entry
    - r_P (int*): table that contains row indexes
    - c_P (int*): table that contains column indexes
    - v_P (double*): table that contains values
  */

  r_P[cpt_P] = i;  // row index
  c_P[cpt_P] = j;  // column index
  v_P[cpt_P] = v;  // value of coefficient
  cpt_P++;         // increment the counter
}

int QPWBC::create_ML() {
  /*
  Create the M and L matrices involved in the constraint equations
  the solution has to respect: M.X = N and L.X <= K
  */

  int *r_ML = new int[size_nz_ML];        // row indexes of non-zero values in matrix ML
  int *c_ML = new int[size_nz_ML];        // col indexes of non-zero values in matrix ML
  double *v_ML = new double[size_nz_ML];  // non-zero values in matrix ML

  std::fill_n(r_ML, size_nz_ML, 0);
  std::fill_n(c_ML, size_nz_ML, 0);
  std::fill_n(v_ML, size_nz_ML, 0.0);

  // ML is the identity matrix of size 12
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 12; j++) {
      add_to_ML(i, j, G(i, j), r_ML, c_ML, v_ML);
    }
  }

  // Creation of CSC matrix
  int *icc;                                  // row indices
  int *ccc;                                  // col indices
  double *acc;                               // coeff values
  int nst = cpt_ML;                          // number of non zero elements
  int ncc = st_to_cc_size(nst, r_ML, c_ML);  // number of CC values
  // int m = 20;   // number of rows
  int n = 12;   // number of columns

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
  ML->m = 20;
  ML->n = 12;
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


int QPWBC::create_weight_matrices() {
  /*
  Create the weight matrices P and Q in the cost function 
  1/2 x^T.P.x + x^T.q of the QP problem
  */

  int *r_P = new int[size_nz_P];        // row indexes of non-zero values in matrix P
  int *c_P = new int[size_nz_P];        // col indexes of non-zero values in matrix P
  double *v_P = new double[size_nz_P];  // non-zero values in matrix P

  std::fill_n(r_P, size_nz_P, 0);
  std::fill_n(c_P, size_nz_P, 0);
  std::fill_n(v_P, size_nz_P, 0.0);

  // Fill P with 1.0 so that the sparse creation process considers that all coeffs
  // can have a non zero value
  for (int i = 0; i < 12; i++) {
    for (int j = i; j < 12; j++) {
      add_to_P(i, j, 1.0, r_P, c_P, v_P);
    }
  }

  // Creation of CSC matrix
  int *icc;                                // row indices
  int *ccc;                                // col indices
  double *acc;                             // coeff values
  int nst = cpt_P;                         // number of non zero elements
  int ncc = st_to_cc_size(nst, r_P, c_P);  // number of CC values
  // int m = 12;                // number of rows
  int n = 12;                // number of columns

  // std::cout << "Number of CC values: " << ncc << std::endl;

  // Get the CC indices.
  icc = (int *)malloc(ncc * sizeof(int));
  ccc = (int *)malloc((n + 1) * sizeof(int));
  st_to_cc_index(nst, r_P, c_P, ncc, n, icc, ccc);

  // Get the CC values.
  acc = st_to_cc_values(nst, r_P, c_P, v_P, ncc, n, icc, ccc);

  // Assign values to the csc object
  P = (csc *)c_malloc(sizeof(csc));
  P->m = 12;
  P->n = 12;
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

int QPWBC::call_solver() {
  /*
  Initialize the solver (first iteration) or update it (next iterations)
  then call the OSQP solver to solve the QP problem
  */

  // Setup the solver (first iteration) then just update it
  if (not initialized)  // Setup the solver with the matrices
  {
    data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = 12;            // number of variables
    data->m = 20;            // number of constraints
    data->P = P;             // the upper triangular part of the quadratic cost matrix P in csc format (size n x n)
    data->A = ML;            // linear constraints matrix A in csc format (size m x n)
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
    settings->verbose = true;
    osqp_setup(&workspce, data, settings);

    initialized = true;
  } else  // Code to update the QP problem without creating it again
  {
    // Update P matrix of the OSQP solver
    osqp_update_P(workspce, &P->x[0], OSQP_NULL, 0);

    // Update Q matrix of the OSQP solver
    osqp_update_lin_cost(workspce, &Q[0]);

    // Update upper bound of the OSQP solver
    osqp_update_upper_bound(workspce, &v_NK_up[0]);
    osqp_update_lower_bound(workspce, &v_NK_low[0]);

  }

  // Run the solver to solve the QP problem
  osqp_solve(workspce);

  // solution in workspce->solution->x

  return 0;
}

int QPWBC::retrieve_result(const Eigen::MatrixXd &f_cmd) {
  /*
  Extract relevant information from the output of the QP solver

  Args:
    - f_cmd (Eigen::MatrixXd): reference contact forces received from the MPC
  */

  // Retrieve the solution of the QP problem
  for (int k = 0; k < 12; k++) {
    f_res(k, 0) = (workspce->solution->x)[k];
  }

  // Computing delta ddq with delta f
  ddq_res = A * f_res + gamma;

  // Adding reference contact forces to delta f
  f_res += f_cmd;

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

int QPWBC::run(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA,
               const Eigen::MatrixXd &k_contact) {
  /*
  Run one iteration of the whole WBC QP problem by calling all the necessary functions (data retrieval,
  update of constraint matrices, update of the solver, running the solver, retrieving result)
  
  Args:
    - M (Eigen::MatrixXd): joint space inertia matrix computed with crba
    - Jc (Eigen::MatrixXd): Jacobian of contact points
    - f_cmd (Eigen::MatrixXd): reference contact forces coming from the MPC
    - RNEA (Eigen::MatrixXd): joint torques according to the current state of the system and the desired joint accelerations
    - k_contact (Eigen::MatrixXd): nb of iterations since contact has been enabled for each foot
  */

  // Create the constraint and weight matrices used by the QP solver
  // Minimize x^T.P.x + 2 x^T.Q with constraints M.X == N and L.X <= K
  if (not initialized) {
    create_matrices();
    // std::cout << G << std::endl;
  }
  
  // Compute the different matrices involved in the box QP
  compute_matrices(M, Jc, f_cmd, RNEA);

  // Update P and Q matrices of the cost function xT P x + 2 xT g
  update_PQ();

  const double Nz_max = 20.0;
  Eigen::Matrix<double, 20, 1> Gf = G * f_cmd;
  
  for (int i = 0; i < G.rows(); i++) {
    v_NK_low[i] = - Gf(i, 0);
    v_NK_up[i] = - Gf(i, 0) + Nz_max;
  }

  const double k_max = 15.0;
  for (int i = 0; i < 4; i++) {
    if (k_contact(0, i) < k_max) {
      v_NK_up[5*i+4] -= Nz_max * (1.0 - k_contact(0, i) / k_max);
    }
    /*else if (k_contact(0, i) == (k_max+10))
    {
      //char t_char[1] = {'M'};
      //cc_print( (data->A)->m, (data->A)->n, (data->A)->nzmax, (data->A)->i, (data->A)->p, (data->A)->x, t_char);
      std::cout << " ### " << k_contact(0, i) << std::endl;

      for (int i = 0; i < data->m; i++) {
        std::cout << data->l[i] << " | " << data->u[i] << " | " << f_cmd(i, 0) << std::endl;
      }
    }*/
    
  }

  // Create an initial guess and call the solver to solve the QP problem
  call_solver();

  // Extract relevant information from the output of the QP solver
  retrieve_result(f_cmd);

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
  /*
  Print positions and value of coefficients in a csc matrix

  Args:
    - M (csc*): pointer to the csc matrix you want to print
    - name (char*): name that should be displayed for the matrix (one char)
  */

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
        c_print("\t%3u [%3u,%3u] = %.3g\n", k - 1, a, b, c);
        
      }
    }
  }
}

void QPWBC::save_csc_matrix(csc *M, std::string filename) {
  /*
  Save positions and value of coefficients of a csc matrix in a csc file

  Args:
    - M (csc*): pointer to the csc matrix you want to save
    - filename (string): name of the generated csv file
  */

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
  /*
  Save positions and value of coefficients of a dense matrix in a csc file

  Args:
    - M (double*): pointer to the dense matrix you want to save
    - size (int): size of the dense matrix
    - filename (string): name of the generated csv file
  */
  
  // Open file
  std::ofstream myfile;
  myfile.open(filename + ".csv");

  for (int j = 0; j < size; j++) {
    myfile << j << "," << 0 << "," << M[j] << "\n";
  }

  myfile.close();
}


void QPWBC::compute_matrices(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA) {
  /*
  Compute all matrices of the Box QP problem

  Args:
    - M (Eigen::MatrixXd): joint space inertia matrix computed with crba
    - Jc (Eigen::MatrixXd): Jacobian of contact points
    - f_cmd (Eigen::MatrixXd): reference contact forces coming from the MPC
    - RNEA (Eigen::MatrixXd): joint torques according to the current state of the system and the desired joint accelerations
  */

  Y = M.block(0, 0, 6, 6);
  X = Jc.block(0, 0, 12, 6).transpose();
  Yinv = pseudoInverse(Y);
  A = Yinv * X;
  gamma = Yinv * ((X * f_cmd) - RNEA);
  H = A.transpose() * Q1 * A + Q2;
  g = A.transpose() * Q1 * gamma;

}

void QPWBC::update_PQ() {
  /*
  Update P and Q matrices in the cost function xT P x + 2 xT Q
  */

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
