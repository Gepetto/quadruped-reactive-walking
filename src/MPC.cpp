#include "example-adder/MPC.hpp"

MPC::MPC(float dt_in, int n_steps_in, float T_gait_in)
{
    dt = dt_in;
    n_steps = n_steps_in;
    T_gait = T_gait_in;

    xref = Eigen::Matrix<float, 12, Eigen::Dynamic>::Zero(12, 1+n_steps);
    x = Eigen::Matrix<float, 12, Eigen::Dynamic>::Zero(12, 1+n_steps);
    S_gait = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(12*n_steps, 1);
    warmxf = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(12*n_steps*2, 1);

    // Predefined variables
    mass = 2.50000279f;
    mu = 0.9f;
    cpt = 0;

    // Predefined matrices
    gI << 3.09249e-2f, -8.00101e-7f, 1.865287e-5f,
          -8.00101e-7f, 5.106100e-2f, 1.245813e-4f,
          1.865287e-5f, 1.245813e-4f, 6.939757e-2f;
    q << 0.0f, 0.0f, 0.2027682f, 0.0f, 0.0f, 0.0f;
    h_ref = q(2, 0);
}
 
/*
Create the constraint matrices of the MPC (M.X = N and L.X <= K)
Create the weight matrices P and Q of the MPC solver (cost 1/2 x^T * P * X + X^T * Q)
*/
int MPC::create_matrices()
{
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
inline void MPC::add_to_ML(int i, int j, float v)
{
    r_ML[cpt] = i; // row index
    c_ML[cpt] = j; // column index
    v_ML[cpt] = v; // value of coefficient
    cpt++; // increment the counter
}

/*
Create the M and L matrices involved in the MPC constraint equations M.X = N and L.X <= K
*/
int MPC::create_ML()
{
    
    std::fill_n(v_ML, size_nz_ML, -1.0); // initialized to -1.0


    // Create matrix ML
    ML = Eigen::MatrixXf::Zero(12*n_steps*2 + 20*n_steps, 12*n_steps*2);
    //int offset_L = 12*n_steps*2;

    // Put identity matrices in M
    ML.block(0, 0, 12*n_steps, 12*n_steps) = (-1.0f) * Eigen::MatrixXf::Identity(12*n_steps, 12*n_steps);
    // Eigen::VectorXi indexes = Eigen::VectorXi::LinSpaced(12*n_steps, 0, 12*n_steps+1);
    //ML[np.arange(0, 12*self.n_steps, 1), np.arange(0, 12*self.n_steps, 1)] = (-1) * Eigen::MatrixXf::Ones(12*n_steps)

    // Create matrix A
    A = Eigen::Matrix<float, 12, 12>::Identity();
    A.block(0, 6, 6, 6) = Eigen::Matrix<float, 6, 6>::Identity();

    // Put matrix A in M
    for (int k=0; k<n_steps-1; k++)
    {
        ML.block((k+1)*12, (k*12), 12, 12) = A;
    }

    // Create matrix B
    for (int k=0; k<4; k++)
    {
        B.block(6, 3*k, 3, 3) = (dt / mass) * Eigen::Matrix<float, 3, 3>::Identity();
    }
    
    // Put B matrices in M
    for (int k=0; k<n_steps-1; k++)
    {
        ML.block((k+1)*12, (k*12), 12, 12) = A;
    }

    c_int irow [2];
    c_int icol [2];
    c_float v_i [2];
    irow[0] = 0;
    icol[0] = 0;
    v_i[0] = 1.0;
    irow[1] = 1;
    icol[1] = 1;
    v_i[1] = 3.0;
    csc* test = csc_matrix(2, 2, 4, v_i, irow, icol);
    std::cout << test->x[0] << std::endl;
    std::cout << test->x[1] << std::endl;
    /*Eigen::Array<int, 6, 1> i_x;
    Eigen::Array<int, 6, 1> i_y;
    i_x << 0, 1, 2, 3, 4, 5;
    i_y << 6, 7, 8, 9, 10, 11;
    A(i_x, i_y) = dt * Eigen::Array<float, 6, 1>::Ones();*/

    return 0;
}

/*
Create the N and K matrices involved in the MPC constraint equations M.X = N and L.X <= K
*/       
int MPC::create_NK()
{
    return 0;
}


/*
Create the weight matrices P and q in the cost function x^T.P.x + x^T.q of the QP problem
*/
int MPC::create_weight_matrices()
{

    return 0;
}
