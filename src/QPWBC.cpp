#include "quadruped-reactive-walking/QPWBC.hpp"


QPWBC::QPWBC() {

    // Initialization of the generatrix G
    Eigen::Matrix<double, 3, 4> Gk;
    Gk.row(0) << mu,  mu, -mu, -mu;
    Gk.row(1) << mu, -mu,  mu, -mu;
    Gk.row(2) << 1.0, 1.0, 1.0, 1.0;
    for (int i = 0; i < 4; i++) {
        G.block(3*i, 4*i, 3, 4) = Gk;
    }

    // Set the lower and upper limits of the box
    std::fill_n(v_NK_up, size_nz_NK, 25.0);
    std::fill_n(v_NK_low, size_nz_NK, 0.0);

    // Set OSQP settings to default
    osqp_set_default_settings(settings);

    Q_qp.setZero();
    C_qp.setZero();
    Beq.setZero();

    Aineq.setZero();
    for (int i = 0; i < 16; i++) {
      Aineq(i, i) = 1.;
      Bineq(i) = 0.0;
    }
}

/*
Create the constraint matrices of the MPC (M.X = N and L.X <= K)
Create the weight matrices P and Q of the MPC solver (cost 1/2 x^T * P * X + X^T * Q)
*/
int QPWBC::create_matrices() {
  // Create the constraint matrices
  create_ML();

  // Create the weight matrices
  create_weight_matrices();

  return 0;
}

/*
Add a new non-zero coefficient to the ML matrix by filling the triplet r_ML / c_ML / v_ML
*/
inline void QPWBC::add_to_ML(int i, int j, double v, int *r_ML, int *c_ML, double *v_ML) {
  r_ML[cpt_ML] = i;  // row index
  c_ML[cpt_ML] = j;  // column index
  v_ML[cpt_ML] = v;  // value of coefficient
  cpt_ML++;          // increment the counter
}

/*
Add a new non-zero coefficient to the P matrix by filling the triplet r_P / c_P / v_P
*/
inline void QPWBC::add_to_P(int i, int j, double v, int *r_P, int *c_P, double *v_P) {
  r_P[cpt_P] = i;  // row index
  c_P[cpt_P] = j;  // column index
  v_P[cpt_P] = v;  // value of coefficient
  cpt_P++;         // increment the counter
}

/*
Create the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K
*/
int QPWBC::create_ML() {
  int *r_ML = new int[size_nz_ML];        // row indexes of non-zero values in matrix ML
  int *c_ML = new int[size_nz_ML];        // col indexes of non-zero values in matrix ML
  double *v_ML = new double[size_nz_ML];  // non-zero values in matrix ML

  std::fill_n(r_ML, size_nz_ML, 0);
  std::fill_n(c_ML, size_nz_ML, 0);
  std::fill_n(v_ML, size_nz_ML, 0.0);

  // ML is the identity matrix of size 12
  for (int k = 0; k < 16; k++) {
    add_to_ML(k, k, 1.0, r_ML, c_ML, v_ML);
  }

  // Creation of CSC matrix
  int *icc;                                  // row indices
  int *ccc;                                  // col indices
  double *acc;                               // coeff values
  int nst = cpt_ML;                          // number of non zero elements
  int ncc = st_to_cc_size(nst, r_ML, c_ML);  // number of CC values
  int m = 16;   // number of rows
  int n = 16;   // number of columns

  std::cout << "Number of CC values: " << ncc << std::endl;

  int i_min = i4vec_min(nst, r_ML);
  int i_max = i4vec_max(nst, r_ML);
  int j_min = i4vec_min(nst, c_ML);
  int j_max = i4vec_max(nst, c_ML);

  // st_header_print(i_min, i_max, j_min, j_max, m, n, nst);

  // Get the CC indices.
  icc = (int *)malloc(ncc * sizeof(int));
  ccc = (int *)malloc((n + 1) * sizeof(int));
  st_to_cc_index(nst, r_ML, c_ML, ncc, n, icc, ccc);

  // Get the CC values.
  acc = st_to_cc_values(nst, r_ML, c_ML, v_ML, ncc, n, icc, ccc);

  // Assign values to the csc object
  ML = (csc *)c_malloc(sizeof(csc));
  ML->m = 16;
  ML->n = 16;
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

/*
Create the weight matrices P and q in the cost function x^T.P.x + x^T.q of the QP problem
*/
int QPWBC::create_weight_matrices() {
  int *r_P = new int[size_nz_P];        // row indexes of non-zero values in matrix P
  int *c_P = new int[size_nz_P];        // col indexes of non-zero values in matrix P
  double *v_P = new double[size_nz_P];  // non-zero values in matrix P

  std::fill_n(r_P, size_nz_P, 0);
  std::fill_n(c_P, size_nz_P, 0);
  std::fill_n(v_P, size_nz_P, 0.0);

  // Fill P with 1.0 so that the sparse creation process considers that all coeffs
  // can have a non zero value
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      add_to_P(i, j, 1.0, r_P, c_P, v_P);
    }
  }

  // Creation of CSC matrix
  int *icc;                                // row indices
  int *ccc;                                // col indices
  double *acc;                             // coeff values
  int nst = cpt_P;                         // number of non zero elements
  int ncc = st_to_cc_size(nst, r_P, c_P);  // number of CC values
  int m = 16;                // number of rows
  int n = 16;                // number of columns

  std::cout << "Number of CC values: " << ncc << std::endl;

  // Get the CC indices.
  icc = (int *)malloc(ncc * sizeof(int));
  ccc = (int *)malloc((n + 1) * sizeof(int));
  st_to_cc_index(nst, r_P, c_P, ncc, n, icc, ccc);

  // Get the CC values.
  acc = st_to_cc_values(nst, r_P, c_P, v_P, ncc, n, icc, ccc);

  // Assign values to the csc object
  P = (csc *)c_malloc(sizeof(csc));
  P->m = 16;
  P->n = 16;
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
Create an initial guess and call the solver to solve the QP problem
*/
int QPWBC::call_solver() {
  // Initial guess for forces (mass evenly supported by all legs in contact)
  //warmxf.block(0, 0, 12 * (n_steps - 1), 1) = x.block(12, 0, 12 * (n_steps - 1), 1);
  //warmxf.block(12 * n_steps, 0, 12 * (n_steps - 1), 1) = x.block(12 * (n_steps + 1), 0, 12 * (n_steps - 1), 1);
  //warmxf.block(12 * (2 * n_steps - 1), 0, 12, 1) = x.block(12 * n_steps, 0, 12, 1);
  //Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(&v_warmxf[0], warmxf.size()) = warmxf;

  // Setup the solver (first iteration) then just update it
  if (not initialized)  // Setup the solver with the matrices
  {
    data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = 16;            // number of variables
    data->m = 16;            // number of constraints
    data->P = P;             // the upper triangular part of the quadratic cost matrix P in csc format (size n x n)
    data->A = ML;            // linear constraints matrix A in csc format (size m x n)
    data->q = Q;         // dense array for linear part of cost function (size n)
    data->l = v_NK_low;  // dense array for lower bound (size m)
    data->u = v_NK_up;   // dense array for upper bound (size m)

    std::cout << "PASS" << std::endl;

    for (int j = 0; j < 16; j++) {
      std::cout << Q[j] << " "; 
    }
    std::cout << std::endl;
    std::cout << "--" << std::endl;
    for (int j = 0; j < 16; j++) {
      std::cout << v_NK_low[j] << " "; 
    }
    std::cout << std::endl;
    std::cout << "--" << std::endl;
    for (int j = 0; j < 16; j++) {
      std::cout << v_NK_up[j] << " "; 
    }
    std::cout << std::endl;

    /*save_csc_matrix(ML, "ML");
    save_csc_matrix(P, "P");
    save_dns_matrix(Q, 12 * n_steps * 2, "Q");
    save_dns_matrix(v_NK_low, 12 * n_steps * 2 + 20 * n_steps, "l");
    save_dns_matrix(v_NK_up, 12 * n_steps * 2 + 20 * n_steps, "u");*/

    // settings->rho = 0.1f;
    // settings->sigma = 1e-6f;
    // settings->max_iter = 4000;
    /*settings->eps_abs = (float)1e-5;*/
    //settings->eps_rel = (float)1e-5;
    /*settings->eps_prim_inf = 1e-4f;
    settings->eps_dual_inf = 1e-4f;
    settings->alpha = 1.6f;
    settings->delta = 1e-6f;
    settings->polish = 0;
    settings->polish_refine_iter = 3;*/
    /*settings->adaptive_rho = (c_int)1;
    settings->adaptive_rho_interval = (c_int)200;
    settings->adaptive_rho_tolerance = (float)5.0;
    settings->adaptive_rho_fraction = (float)0.7;*/
    settings->verbose = true;
    int exitflag = 0;
    exitflag = osqp_setup(&workspce, data, settings);
    std::cout << "Setup exitflag: " << exitflag << std::endl;
    std::cout << "PASS 2" << std::endl;

    /*self.prob.setup(P=self.P, q=self.Q, A=self.ML, l=self.NK_inf, u=self.NK.ravel(), verbose=False)
    self.prob.update_settings(eps_abs=1e-5)
    self.prob.update_settings(eps_rel=1e-5)*/

    initialized = true;
  } else  // Code to update the QP problem without creating it again
  {
    std::cout << "PASS 3" << std::endl;
    osqp_update_P(workspce, &P->x[0], OSQP_NULL, 0);
    std::cout << "PASS 4" << std::endl;
    osqp_update_lin_cost(workspce, &Q[0]);
    // osqp_update_A(workspce, &ML->x[0], OSQP_NULL, 0);
    // osqp_update_bounds(workspce, &v_NK_low[0], &v_NK_up[0]);
    // osqp_warm_start_x(workspce, &v_warmxf[0]);
  }
  std::cout << "PASS 5" << std::endl;

  /*char t_char[1] = {'P'};
  my_print_csc_matrix(P, t_char);

  char tm_char[1] = {'M'};
  my_print_csc_matrix(ML, tm_char);*/
  double v_warmxf[16] = {};
  std::fill_n(v_warmxf, 16, 2.0);

  std::cout << "PASS 5.5" << std::endl;

  osqp_warm_start_x(workspce, v_warmxf);

  std::cout << "Warm" << std::endl;

  // Run the solver to solve the QP problem
  osqp_solve(workspce);

  std::cout << "PASS 6" << std::endl;
  /*self.sol = self.prob.solve()
  self.x = self.sol.x*/
  // solution in workspce->solution->x

  return 0;
}

/*
Extract relevant information from the output of the QP solver
*/
int QPWBC::retrieve_result(const Eigen::MatrixXd &f_cmd) {
  // Retrieve the "contact forces" part of the solution of the QP problem
  for (int k = 0; k < 16; k++) {
    lambdas(k, 0) = x_qp(k); // (workspce->solution->x)[k];
  }

  f_res = G * lambdas;
  ddq_res = A * (f_res - f_cmd) + gamma;

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
Return the next predicted state of the base
*/
Eigen::MatrixXd QPWBC::get_f_res() { return f_res; }
Eigen::MatrixXd QPWBC::get_ddq_res() { return ddq_res; }

/*
Run one iteration of the whole MPC by calling all the necessary functions (data retrieval,
update of constraint matrices, update of the solver, running the solver, retrieving result)
*/
int QPWBC::run(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA) {

  // Create the constraint and weight matrices used by the QP solver
  // Minimize x^T.P.x + x^T.Q with constraints M.X == N and L.X <= K
  // if (not initialized) {
  //   create_matrices();
  // }
  
  std::cout << "Creation done" << std::endl;

  compute_matrices(M, Jc, f_cmd, RNEA);

  std::cout << "compute_matrices done" << std::endl;

  update_PQ();
  
  std::cout << "update_PQ done" << std::endl;

  // Create an initial guess and call the solver to solve the QP problem
  //call_solver();
  qp.solve_quadprog(Q_qp, C_qp, Aeq, Beq, Aineq, Bineq, x_qp);

  std::cout << "A:" << std::endl << A << std::endl << "--" << std::endl;
  std::cout << "Xf:" << std::endl << (X * f_cmd) << std::endl << "--" << std::endl;
  std::cout << "RNEA:" << std::endl << RNEA << std::endl << "--" << std::endl;
  std::cout << "B:" << std::endl << gamma << std::endl << "--" << std::endl;
  std::cout << "AT Q1:" << std::endl << A.transpose() * Q1 << std::endl << "--" << std::endl;
  std::cout << "g:" << std::endl << g << std::endl << "--" << std::endl;
  std::cout << "H:" << std::endl << H << std::endl << "--" << std::endl;
  std::cout << Q_qp << std::endl;
  std::cout << C_qp << std::endl;
  std::cout << Aeq << std::endl;
  std::cout << Beq << std::endl;
  std::cout << Aineq << std::endl;
  std::cout << Bineq << std::endl;

  std::cout << "call_solver done" << std::endl;

  std::cout << "Raw result: " << std::endl << x_qp << std::endl;

  // Extract relevant information from the output of the QP solver
  retrieve_result(f_cmd);

  std::cout << "retrieve done" << std::endl;

  //char t_char[1] = {'P'};
  //my_print_csc_matrix(P, t_char);

  return 0;
}

void QPWBC::my_print_csc_matrix(csc *M, const char *name) {
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


void QPWBC::compute_matrices(const Eigen::MatrixXd &M, const Eigen::MatrixXd &Jc, const Eigen::MatrixXd &f_cmd, const Eigen::MatrixXd &RNEA) {

    // Compute all matrices of the Box QP problem
    Y = M.block(0, 0, 6, 6);
    X = Jc.block(0, 0, 12, 6).transpose();
    Yinv = pseudoInverse(Y);
    A = Yinv * X;
    gamma = Yinv * ((X * f_cmd) - RNEA);
    H = A.transpose() * Q1 * A + Q2;
    g = A.transpose() * Q1 * gamma;
    Pw = G.transpose() * H * G;
    Qw = (G.transpose() * g) - (G.transpose() * H * f_cmd);

}

void QPWBC::update_PQ() {

  // Update P and Q weight matrices
  /*for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
       P->x[i * 16 + j] = Pw(j, i);
    }
  }

  for (int i = 0; i < 16; i++) {
    Q[i] = Qw(i, 0);
  }*/

  // Update P and Q weight matrices
  Q_qp = Pw;
  for (int i = 0; i < 16; i++) {
    C_qp(i) = Qw(i, 0);
  }

}