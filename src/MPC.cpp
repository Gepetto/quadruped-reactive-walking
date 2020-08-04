#include "example-adder/MPC.hpp"

MPC::MPC(float dt_in, int n_steps_in, float T_gait_in)
{
    dt = dt_in;
    n_steps = n_steps_in;
    T_gait = T_gait_in;

    xref = Eigen::Matrix<float, 12, Eigen::Dynamic>::Zero(12, 1+n_steps);
    x = Eigen::Matrix<float, 12, Eigen::Dynamic>::Zero(12, 1+n_steps);
    S_gait = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(12*n_steps, 1);
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

    // Put identity matrices in M
    for (int k=0; k<(12*n_steps); k++)
    {
        add_to_ML(k, k, -1.0);
    }
    
    // Put A matrices in M
    for (int k=0; k<n_steps-1; k++)
    {   
        for (int i=0; i<12; i++)
        {
            add_to_ML((k+1)*12 + i, (k*12) + i, 1.0);
        }
        for (int i=0; i<6; i++)
        {
            add_to_ML((k+1)*12 + i, (k*12) + i + 6, 1.0);
        }
    }

    // Put B matrices in M
    float div_tmp = dt / mass;
    for (int k=0; k<n_steps; k++)
    {   
        for (int i=0; i<4; i++)
        {
            add_to_ML(12*k + 6, 12*(n_steps+k) + 0+3*i, div_tmp);
            add_to_ML(12*k + 7, 12*(n_steps+k) + 1+3*i, div_tmp);
            add_to_ML(12*k + 8, 12*(n_steps+k) + 2+3*i, div_tmp);
            B(12*k + 6, 12*(n_steps+k) + 0+3*i) = div_tmp;
            B(12*k + 7, 12*(n_steps+k) + 1+3*i) = div_tmp;
            B(12*k + 8, 12*(n_steps+k) + 2+3*i) = div_tmp;
        }
        for (int i=0; i<12; i++)
        {
            add_to_ML(12*k + 9,  12*(n_steps+k) + i, 8.0);
            add_to_ML(12*k + 10, 12*(n_steps+k) + i, 8.0);
            add_to_ML(12*k + 11, 12*(n_steps+k) + i, 8.0);
            B(12*k + 9,  12*(n_steps+k) + i) = 8.0;
            B(12*k + 10, 12*(n_steps+k) + i) = 8.0;
            B(12*k + 11, 12*(n_steps+k) + i) = 8.0;
        }   
    }

    // Add lines to enable/disable forces
    for (int i=12*n_steps; i<12*n_steps*2; i++)
    {
        for (int j=12*n_steps; j<12*n_steps*2; j++)
        {
            add_to_ML(i, j, 1.0);
        } 
    }

    // Fill ML with F matrices
    int offset_L = 12*n_steps*2;
    for (int k=0; k<n_steps; k++)
    {   
        int di = offset_L+20*k;
        int dj = 12*(n_steps+k);
        // Matrix F with top left corner at (di, dj) in ML
        for (int i=0; i<4; i++)
        {
            int dx = 5*i;
            int dy = 3*i;
            int a [9] = {0, 1, 2, 3, 0, 1, 2, 3, 4};
            int b [9] = {0, 0, 1, 1, 2, 2, 2, 2, 2};
            float c [9] = {1.0, -1.0, 1.0, -1.0, -mu, -mu, -mu, -mu, -1};
            // Matrix C with top left corner at (dx, dy) in F
            for (int j=0; j<9; j++)
            {
                add_to_ML(di+dx+a[j], dj+dy+b[j], c[j]);
            }
        }
    }

    // Creation of CSC matrix
    ML = csc_matrix(12*n_steps*2 + 20*n_steps, 12*n_steps*2, size_nz_ML, 
                         v_ML, r_ML, c_ML);

    // Create indices list that will be used to update ML
    int i_x_tmp[12] = {6, 9, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
    for (int k=0; k<4; k++)
    {
        for (int i=0; i<12; i++)
        {
            i_x_B[12*k+i] = i_x_tmp[i];
            i_y_B[12*k+i] = (12*k+i) / 4; // 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2...
        }
    }

    int i_start = 30*n_steps-18;
    int i_data [12] = {0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17};
    int i_foot [4] = {0*24, 1*24, 2*24, 3*24};
    for (int k=0; k<4; k++)
    {
        for (int i=0; i<12; i++)
        {
            i_update_B[12*k+i] = i_start + i_data[i] + i_foot[k];
        }
    }

    // i_update_S here?

    // Update state of B
    for (int k=0; k<n_steps; k++)
    { 
        // Get inverse of the inertia matrix for time step k
        float c = cos(xref(5, k));
        float s = sin(xref(5, k));
        Eigen::Matrix<float, 3, 3> R;
        R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
        Eigen::Matrix<float, 3, 3> R_gI = R * gI;
        Eigen::Matrix<float, 3, 3> I_inv = R_gI.inverse(); 

        // Get skew-symetric matrix for each foothold
        Eigen::Matrix<float, 3, 4> l_arms = footholds - (xref.block(0, k, 3, 1)).replicate<1,4>();
        for (int i=0; i<4; i++)
        {
            B.block(9, 3*i, 3, 3) = dt * (I_inv * getSkew(l_arms.col(i)));
        }

        int i_iter = 24 * 4 * k;
        for (int j=0; j<12*4; j++)
        {
            ML->x[i_update_B[j] + i_iter] = B(i_x_B[j], i_y_B[j]);
        }
        
    }

    // Update lines to enable/disable forces
    construct_S();
    Eigen::Matrix<int, 3, 1> i_tmp1;
    i_tmp1 << 3 + 4, 3 + 4, 6 + 4;
    Eigen::Matrix<int, Eigen::Dynamic, 1> i_tmp2 = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(12*n_steps,1);//i_tmp1.replicate<4,1>();
    for (int k=1; k<4*n_steps; k++)
    {
        i_tmp2.block(3*k, 0, 3, 1) = i_tmp1;
    }
    Eigen::Matrix<int, Eigen::Dynamic, 1> i_off = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(12*n_steps,1);
    for (int k=1; k<12*n_steps; k++)
    {
        i_off(k, 1) = i_off(k-1, 1) + i_tmp2(k-1, 1);
        ML->x[i_off(k, 1)+ i_start] = S_gait(k, 1);
    }

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

/*
Returns the skew matrix of a 3 by 1 column vector
*/
Eigen::Matrix<float, 3, 3> MPC::getSkew(Eigen::Matrix<float, 3, 1> v)
{
    Eigen::Matrix<float, 3, 3> result;
    result << 0.0, -v(2, 0), v(1, 0), v(2, 0), 0.0, -v(0, 0), -v(1, 0), v(0, 0), 0.0;
    return result;
}

/*
Construct an array of size 12*N that contains information about the contact state of feet.
This matrix is used to enable/disable contact forces in the QP problem.
N is the number of time step in the prediction horizon.
*/
int MPC::construct_S()
{
    int i = 0;
    int k = 0;

    Eigen::Matrix<int, 20, 5> inv_gait = Eigen::Matrix<int, 20, 5>::Ones() - gait;
    while (gait(i, 0) != 0)
    {
        // S_gait.block(k*12, 0, gait[i, 0]*12, 1) = (1 - (gait.block(i, 1, 1, 4)).transpose()).replicate<gait[i, 0], 1>() not finished;
        for (int a=0; a<gait(i, 0); a++)
        {
            for (int b=0; b<4; b++)
            {
                for (int c=0; c<3; c++)
                {
                    S_gait(k*12, gait(i, 0)*12 + 12*a + 4*b + c) = inv_gait(i, 1+b);
                }
            }
        }
        k += gait(i, 0);
        i++;
    }

    return 0;
}