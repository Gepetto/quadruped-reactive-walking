#include "qrw/Gait.hpp"

Gait::Gait()
    : pastGait_(MatrixN::Zero(N0_gait, 5))
    , currentGait_(MatrixN::Zero(N0_gait, 5))
    , desiredGait_(MatrixN::Zero(N0_gait, 5))
    , dt_(0.0)
    , T_gait_(0.0)
    , T_mpc_(0.0)
    , remainingTime_(0.0)
    , is_static_(true)
    , q_static_(VectorN::Zero(19))
{
    // Empty
}


void Gait::initialize(double dt_in, double T_gait_in, double T_mpc_in)
{
    dt_ = dt_in;
    T_gait_ = T_gait_in;
    T_mpc_ = T_mpc_in;

    create_trot();
    create_gait_f();
}


int Gait::create_walk()
{
    // Number of timesteps in 1/4th period of gait
    int N = (int)std::lround(0.25 * T_gait_ / dt_);

    desiredGait_ = Eigen::Matrix<double, N0_gait, 5>::Zero();
    desiredGait_.block(0, 0, 4, 1) << N, N, N, N;

    // Set stance and swing phases
    // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
    // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
    desiredGait_.block(0, 1, 1, 4) << 0.0, 1.0, 1.0, 1.0;
    desiredGait_.block(1, 1, 1, 4) << 1.0, 0.0, 1.0, 1.0;
    desiredGait_.block(2, 1, 1, 4) << 1.0, 1.0, 0.0, 1.0;
    desiredGait_.block(3, 1, 1, 4) << 1.0, 1.0, 1.0, 0.0;

    return 0;
}

int Gait::create_trot()
{
    // Number of timesteps in a half period of gait
    int N = (int)std::lround(0.5 * T_gait_ / dt_);

    desiredGait_ = Eigen::Matrix<double, N0_gait, 5>::Zero();
    desiredGait_.block(0, 0, 2, 1) << N, N;

    // Set stance and swing phases
    // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
    // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
    desiredGait_(0, 1) = 1.0;
    desiredGait_(0, 4) = 1.0;
    desiredGait_(1, 2) = 1.0;
    desiredGait_(1, 3) = 1.0;

    return 0;
}

int Gait::create_pacing()
{
    // Number of timesteps in a half period of gait
    int N = (int)std::lround(0.5 * T_gait_ / dt_);

    desiredGait_ = Eigen::Matrix<double, N0_gait, 5>::Zero();
    desiredGait_.block(0, 0, 2, 1) << N, N;

    // Set stance and swing phases
    // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
    // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
    desiredGait_(0, 1) = 1.0;
    desiredGait_(0, 3) = 1.0;
    desiredGait_(1, 2) = 1.0;
    desiredGait_(1, 4) = 1.0;

    return 0;
}

int Gait::create_bounding()
{
    // Number of timesteps in a half period of gait
    int N = (int)std::lround(0.5 * T_gait_ / dt_);

    desiredGait_ = Eigen::Matrix<double, N0_gait, 5>::Zero();
    desiredGait_.block(0, 0, 2, 1) << N, N;

    // Set stance and swing phases
    // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
    // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
    desiredGait_(0, 1) = 1.0;
    desiredGait_(0, 2) = 1.0;
    desiredGait_(1, 3) = 1.0;
    desiredGait_(1, 4) = 1.0;

    return 0;
}

int Gait::create_static()
{
    // Number of timesteps in a half period of gait
    int N = (int)std::lround(T_gait_ / dt_);

    desiredGait_ = Eigen::Matrix<double, N0_gait, 5>::Zero();
    desiredGait_(0, 0) = N;

    // Set stance and swing phases
    // Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
    // Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
    desiredGait_(0, 1) = 1.0;
    desiredGait_(0, 2) = 1.0;
    desiredGait_(0, 3) = 1.0;
    desiredGait_(0, 4) = 1.0;

    return 0;
}

int Gait::create_gait_f()
{
    double sum = 0.0;
    double offset = 0.0;
    int i = 0;
    int j = 0;

    // Fill future gait matrix
    while (sum < (T_mpc_ / dt_))
    {
        currentGait_.row(j) = desiredGait_.row(i);
        sum += desiredGait_(i, 0);
        offset += desiredGait_(i, 0);
        i++;
        j++;
        if (desiredGait_(i, 0) == 0)
        {
            i = 0;
            offset = 0.0;
        }  // Loop back if T_mpc_ longer than gait duration
    }

    // Remove excess time steps
    currentGait_(j - 1, 0) -= sum - (T_mpc_ / dt_);
    offset -= sum - (T_mpc_ / dt_);

    // Age future desired gait to take into account what has been put in the future gait matrix
    j = 1;
    while (desiredGait_(j, 0) > 0.0)
    {
        j++;
    }

    for (double k = 0; k < offset; k++)
    {
        if ((desiredGait_.block(0, 1, 1, 4)).isApprox(desiredGait_.block(j - 1, 1, 1, 4)))
        {
            desiredGait_(j - 1, 0) += 1.0;
        }
        else
        {
            desiredGait_.row(j) = desiredGait_.row(0);
            desiredGait_(j, 0) = 1.0;
            j++;
        }
        if (desiredGait_(0, 0) == 1.0)
        {
            desiredGait_.block(0, 0, N0_gait - 1, 5) = desiredGait_.block(1, 0, N0_gait - 1, 5);
            j--;
        }
        else
        {
            desiredGait_(0, 0) -= 1.0;
        }
    }

    return 0;
}

double Gait::getPhaseDuration(int i, int j, double value)
{
    double t_phase = currentGait_(i, 0);
    int a = i;

    // Looking for the end of the swing/stance phase in currentGait_
    while ((currentGait_(i + 1, 0) > 0.0) && (currentGait_(i + 1, 1 + j) == value))
    {
        i++;
        t_phase += currentGait_(i, 0);
    }
    // If we reach the end of currentGait_ we continue looking for the end of the swing/stance phase in desiredGait_
    if (currentGait_(i + 1, 0) == 0.0)
    {
        int k = 0;
        while ((desiredGait_(k, 0) > 0.0) && (desiredGait_(k, 1 + j) == value))
        {
            t_phase += desiredGait_(k, 0);
            k++;
        }
    }
    // We suppose that we found the end of the swing/stance phase either in currentGait_ or desiredGait_

    remainingTime_ = t_phase;

    // Looking for the beginning of the swing/stance phase in currentGait_
    while ((a > 0) && (currentGait_(a - 1, 1 + j) == value))
    {
        a--;
        t_phase += currentGait_(a, 0);
    }
    // If we reach the end of currentGait_ we continue looking for the beginning of the swing/stance phase in pastGait_
    if (a == 0)
    {
        while ((pastGait_(a, 0) > 0.0) && (pastGait_(a, 1 + j) == value))
        {
            t_phase += pastGait_(a, 0);
            a++;
        }
    }
    // We suppose that we found the beginning of the swing/stance phase either in currentGait_ or pastGait_

    return t_phase * dt_;  // Take into account time step value
}

void Gait::roll(int k, Matrix34 const& footstep, Matrix34& currentFootstep)
{
    // Transfer current gait into past gait
    // If current gait is the same than the first line of past gait we just increment the counter
    if ((currentGait_.block(0, 1, 1, 4)).isApprox(pastGait_.block(0, 1, 1, 4)))
    {
        pastGait_(0, 0) += 1.0;
    }
    else
    {  // If current gait is not the same than the first line of past gait we have to insert it
        Eigen::Matrix<double, 5, 5> tmp = pastGait_.block(0, 0, N0_gait - 1, 5);
        pastGait_.block(1, 0, N0_gait - 1, 5) = tmp;
        pastGait_.row(0) = currentGait_.row(0);
        pastGait_(0, 0) = 1.0;
    }

    // Age future gait
    if (currentGait_(0, 0) == 1.0)
    {
        currentGait_.block(0, 0, N0_gait - 1, 5) = currentGait_.block(1, 0, N0_gait - 1, 5);
        // Entering new contact phase, store positions of feet that are now in contact
        if (k != 0)
        {
            for (int i = 0; i < 4; i++)
            {
                if (currentGait_(0, 1 + i) == 1.0)
                {
                    currentFootstep.col(i) = footstep.col(i);
                }
            }
        }
    }
    else
    {
        currentGait_(0, 0) -= 1.0;
    }

    // Get index of first empty line
    int i = 1;
    while (currentGait_(i, 0) > 0.0)
    {
        i++;
    }
    // Increment last gait line or insert a new line
    if ((currentGait_.block(i - 1, 1, 1, 4)).isApprox(desiredGait_.block(0, 1, 1, 4)))
    {
        currentGait_(i - 1, 0) += 1.0;
    }
    else
    {
        currentGait_.row(i) = desiredGait_.row(0);
        currentGait_(i, 0) = 1.0;
    }

    // Age future desired gait
    // Get index of first empty line
    int j = 1;
    while (desiredGait_(j, 0) > 0.0)
    {
        j++;
    }
    // Increment last gait line or insert a new line
    if ((desiredGait_.block(0, 1, 1, 4)).isApprox(desiredGait_.block(j - 1, 1, 1, 4)))
    {
        desiredGait_(j - 1, 0) += 1.0;
    }
    else
    {
        desiredGait_.row(j) = desiredGait_.row(0);
        desiredGait_(j, 0) = 1.0;
    }
    if (desiredGait_(0, 0) == 1.0)
    {
        desiredGait_.block(0, 0, N0_gait - 1, 5) = desiredGait_.block(1, 0, N0_gait - 1, 5);
    }
    else
    {
        desiredGait_(0, 0) -= 1.0;
    }
}

bool Gait::changeGait(int const code, VectorN const& q)
{
    is_static_ = false;
    if (code == 1)
    {
        create_pacing();
    }
    else if (code == 2)
    {
        create_bounding();
    }
    else if (code == 3)
    {
        create_trot();
    }
    else if (code == 4)
    {
        create_static();
        q_static_.head(7) = q.head(7);
        is_static_ = true;
    }
    return is_static_;
}
