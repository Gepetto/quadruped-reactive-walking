#include "qrw/FootTrajectoryGenerator.hpp"

// Trajectory generator functions (output reference pos, vel and acc of feet in swing phase)

FootTrajectoryGenerator::FootTrajectoryGenerator()
    : gait_(nullptr)
    , dt_tsid(0.0)
    , k_mpc(0)
    , maxHeight_(0.0)
    , lockTime_(0.0)
    , feet()
    , t0s(Vector4::Zero())
    , t_swing(Vector4::Zero())
    , targetFootstep_(Matrix34::Zero())
    , Ax(Matrix64::Zero())
    , Ay(Matrix64::Zero())
    , position_(Matrix34::Zero())
    , velocity_(Matrix34::Zero())
    , acceleration_(Matrix34::Zero())
{
}

void FootTrajectoryGenerator::initialize(double const maxHeightIn,
                                         double const lockTimeIn,
                                         Matrix34 const& targetFootstepIn,
                                         Matrix34 const& initialFootPosition,
                                         double const& dt_tsid_in,
                                         int const& k_mpc_in,
                                         std::shared_ptr<Gait> gaitIn)
{
    dt_tsid = dt_tsid_in;
    k_mpc = k_mpc_in;
    maxHeight_ = maxHeightIn;
    lockTime_ = lockTime_;
    targetFootstep_ = targetFootstepIn;
    position_ = initialFootPosition;
    gait_ = gaitIn;
}


void FootTrajectoryGenerator::updateFootPosition(int const j, Vector3 const& targetFootstep)
{
    double ddx0 = acceleration_(0, j);
    double ddy0 = acceleration_(1, j);
    double dx0 = velocity_(0, j);
    double dy0 = velocity_(1, j);
    double x0 = position_(0, j);
    double y0 = position_(1, j);

    double t = t0s[j];
    double d = t_swing[j];
    double dt = dt_tsid;

    if (t < d - lockTime_)
    {
        // compute polynoms coefficients for x and y
        Ax(0, j) = (ddx0 * std::pow(t, 2) - 2 * ddx0 * t * d - 6 * dx0 * t + ddx0 * std::pow(d, 2) + 6 * dx0 * d + 12 * x0 - 12 * targetFootstep[0]) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ax(1, j) = (30 * t * targetFootstep[0] - 30 * t * x0 - 30 * d * x0 + 30 * d * targetFootstep[0] - 2 * std::pow(t, 3) * ddx0 - 3 * std::pow(d, 3) * ddx0 + 14 * std::pow(t, 2) * dx0 - 16 * std::pow(d, 2) * dx0 + 2 * t * d * dx0 + 4 * t * std::pow(d, 2) * ddx0 + std::pow(t, 2) * d * ddx0) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ax(2, j) = (std::pow(t, 4) * ddx0 + 3 * std::pow(d, 4) * ddx0 - 8 * std::pow(t, 3) * dx0 + 12 * std::pow(d, 3) * dx0 + 20 * std::pow(t, 2) * x0 - 20 * std::pow(t, 2) * targetFootstep[0] + 20 * std::pow(d, 2) * x0 - 20 * std::pow(d, 2) * targetFootstep[0] + 80 * t * d * x0 - 80 * t * d * targetFootstep[0] + 4 * std::pow(t, 3) * d * ddx0 + 28 * t * std::pow(d, 2) * dx0 - 32 * std::pow(t, 2) * d * dx0 - 8 * std::pow(t, 2) * std::pow(d, 2) * ddx0) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ax(3, j) = -(std::pow(d, 5) * ddx0 + 4 * t * std::pow(d, 4) * ddx0 + 3 * std::pow(t, 4) * d * ddx0 + 36 * t * std::pow(d, 3) * dx0 - 24 * std::pow(t, 3) * d * dx0 + 60 * t * std::pow(d, 2) * x0 + 60 * std::pow(t, 2) * d * x0 - 60 * t * std::pow(d, 2) * targetFootstep[0] - 60 * std::pow(t, 2) * d * targetFootstep[0] - 8 * std::pow(t, 2) * std::pow(d, 3) * ddx0 - 12 * std::pow(t, 2) * std::pow(d, 2) * dx0) / (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ax(4, j) = -(2 * std::pow(d, 5) * dx0 - 2 * t * std::pow(d, 5) * ddx0 - 10 * t * std::pow(d, 4) * dx0 + std::pow(t, 2) * std::pow(d, 4) * ddx0 + 4 * std::pow(t, 3) * std::pow(d, 3) * ddx0 - 3 * std::pow(t, 4) * std::pow(d, 2) * ddx0 - 16 * std::pow(t, 2) * std::pow(d, 3) * dx0 + 24 * std::pow(t, 3) * std::pow(d, 2) * dx0 - 60 * std::pow(t, 2) * std::pow(d, 2) * x0 + 60 * std::pow(t, 2) * std::pow(d, 2) * targetFootstep[0]) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ax(5, j) = (2 * targetFootstep[0] * std::pow(t, 5) - ddx0 * std::pow(t, 4) * std::pow(d, 3) - 10 * targetFootstep[0] * std::pow(t, 4) * d + 2 * ddx0 * std::pow(t, 3) * std::pow(d, 4) + 8 * dx0 * std::pow(t, 3) * std::pow(d, 3) + 20 * targetFootstep[0] * std::pow(t, 3) * std::pow(d, 2) - ddx0 * std::pow(t, 2) * std::pow(d, 5) - 10 * dx0 * std::pow(t, 2) * std::pow(d, 4) - 20 * x0 * std::pow(t, 2) * std::pow(d, 3) + 2 * dx0 * t * std::pow(d, 5) + 10 * x0 * t * std::pow(d, 4) - 2 * x0 * std::pow(d, 5)) / (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));

        Ay(0, j) = (ddy0 * std::pow(t, 2) - 2 * ddy0 * t * d - 6 * dy0 * t + ddy0 * std::pow(d, 2) + 6 * dy0 * d + 12 * y0 - 12 * targetFootstep[1]) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ay(1, j) = (30 * t * targetFootstep[1] - 30 * t * y0 - 30 * d * y0 + 30 * d * targetFootstep[1] - 2 * std::pow(t, 3) * ddy0 - 3 * std::pow(d, 3) * ddy0 + 14 * std::pow(t, 2) * dy0 - 16 * std::pow(d, 2) * dy0 + 2 * t * d * dy0 + 4 * t * std::pow(d, 2) * ddy0 + std::pow(t, 2) * d * ddy0) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ay(2, j) = (std::pow(t, 4) * ddy0 + 3 * std::pow(d, 4) * ddy0 - 8 * std::pow(t, 3) * dy0 + 12 * std::pow(d, 3) * dy0 + 20 * std::pow(t, 2) * y0 - 20 * std::pow(t, 2) * targetFootstep[1] + 20 * std::pow(d, 2) * y0 - 20 * std::pow(d, 2) * targetFootstep[1] + 80 * t * d * y0 - 80 * t * d * targetFootstep[1] + 4 * std::pow(t, 3) * d * ddy0 + 28 * t * std::pow(d, 2) * dy0 - 32 * std::pow(t, 2) * d * dy0 - 8 * std::pow(t, 2) * std::pow(d, 2) * ddy0) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ay(3, j) = -(std::pow(d, 5) * ddy0 + 4 * t * std::pow(d, 4) * ddy0 + 3 * std::pow(t, 4) * d * ddy0 + 36 * t * std::pow(d, 3) * dy0 - 24 * std::pow(t, 3) * d * dy0 + 60 * t * std::pow(d, 2) * y0 + 60 * std::pow(t, 2) * d * y0 - 60 * t * std::pow(d, 2) * targetFootstep[1] - 60 * std::pow(t, 2) * d * targetFootstep[1] - 8 * std::pow(t, 2) * std::pow(d, 3) * ddy0 - 12 * std::pow(t, 2) * std::pow(d, 2) * dy0) / (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ay(4, j) = -(2 * std::pow(d, 5) * dy0 - 2 * t * std::pow(d, 5) * ddy0 - 10 * t * std::pow(d, 4) * dy0 + std::pow(t, 2) * std::pow(d, 4) * ddy0 + 4 * std::pow(t, 3) * std::pow(d, 3) * ddy0 - 3 * std::pow(t, 4) * std::pow(d, 2) * ddy0 - 16 * std::pow(t, 2) * std::pow(d, 3) * dy0 + 24 * std::pow(t, 3) * std::pow(d, 2) * dy0 - 60 * std::pow(t, 2) * std::pow(d, 2) * y0 + 60 * std::pow(t, 2) * std::pow(d, 2) * targetFootstep[1]) / (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
        Ay(5, j) = (2 * targetFootstep[1] * std::pow(t, 5) - ddy0 * std::pow(t, 4) * std::pow(d, 3) - 10 * targetFootstep[1] * std::pow(t, 4) * d + 2 * ddy0 * std::pow(t, 3) * std::pow(d, 4) + 8 * dy0 * std::pow(t, 3) * std::pow(d, 3) + 20 * targetFootstep[1] * std::pow(t, 3) * std::pow(d, 2) - ddy0 * std::pow(t, 2) * std::pow(d, 5) - 10 * dy0 * std::pow(t, 2) * std::pow(d, 4) - 20 * y0 * std::pow(t, 2) * std::pow(d, 3) + 2 * dy0 * t * std::pow(d, 5) + 10 * y0 * t * std::pow(d, 4) - 2 * y0 * std::pow(d, 5)) / (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));

        targetFootstep_(0, j) = targetFootstep[0];
        targetFootstep_(1, j) = targetFootstep[1];
    }

    // Coefficients for z (deterministic)
    Vector4 Az;
    Az(0, j) = -maxHeight_ / (std::pow((d / 2), 3) * std::pow((d - d / 2), 3));
    Az(1, j) = (3 * d * maxHeight_) / (std::pow((d / 2), 3) * std::pow((d - d / 2), 3));
    Az(2, j) = -(3 * std::pow(d, 2) * maxHeight_) / (std::pow((d / 2), 3) * std::pow((d - d / 2), 3));
    Az(3, j) = (std::pow(d, 3) * maxHeight_) / (std::pow((d / 2), 3) * std::pow((d - d / 2), 3));

    // Get the next point
    double ev = t + dt;

    if (t < 0.0 || t > d)  // Just vertical motion
    {
        position_(0, j) = x0;
        position_(1, j) = y0;
        velocity_(0, j) = 0.0;
        velocity_(1, j) = 0.0;
        acceleration_(0, j) = 0.0;
        acceleration_(1, j) = 0.0;
    }
    else
    {
        position_(0, j) = Ax(5, j) + Ax(4, j) * ev + Ax(3, j) * std::pow(ev, 2) + Ax(2, j) * std::pow(ev, 3) + Ax(1, j) * std::pow(ev, 4) + Ax(0, j) * std::pow(ev, 5);
        position_(1, j) = Ay(5, j) + Ay(4, j) * ev + Ay(3, j) * std::pow(ev, 2) + Ay(2, j) * std::pow(ev, 3) + Ay(1, j) * std::pow(ev, 4) + Ay(0, j) * std::pow(ev, 5);
        velocity_(0, j) = Ax(4, j) + 2 * Ax(3, j) * ev + 3 * Ax(2, j) * std::pow(ev, 2) + 4 * Ax(1, j) * std::pow(ev, 3) + 5 * Ax(0, j) * std::pow(ev, 4);
        velocity_(1, j) = Ay(4, j) + 2 * Ay(3, j) * ev + 3 * Ay(2, j) * std::pow(ev, 2) + 4 * Ay(1, j) * std::pow(ev, 3) + 5 * Ay(0, j) * std::pow(ev, 4);
        acceleration_(0, j) = 2 * Ax(3, j) + 3 * 2 * Ax(2, j) * ev + 4 * 3 * Ax(1, j) * std::pow(ev, 2) + 5 * 4 * Ax(0, j) * std::pow(ev, 3);
        acceleration_(1, j) = 2 * Ay(3, j) + 3 * 2 * Ay(2, j) * ev + 4 * 3 * Ay(1, j) * std::pow(ev, 2) + 5 * 4 * Ay(0, j) * std::pow(ev, 3);
    }
    velocity_(2, j) = 3 * Az(3, j) * std::pow(ev, 2) + 4 * Az(2, j) * std::pow(ev, 3) + 5 * Az(1, j) * std::pow(ev, 4) + 6 * Az(0, j) * std::pow(ev, 5);
    acceleration_(2, j) = 2 * 3 * Az(3, j) * ev + 3 * 4 * Az(2, j) * std::pow(ev, 2) + 4 * 5 * Az(1, j) * std::pow(ev, 3) + 5 * 6 * Az(0, j) * std::pow(ev, 4);
    position_(2, j) = Az(3, j) * std::pow(ev, 3) + Az(2, j) * std::pow(ev, 4) + Az(1, j) * std::pow(ev, 5) + Az(0, j) * std::pow(ev, 6);
}

void FootTrajectoryGenerator::update(int k, MatrixN const& targetFootstep)
{
    if ((k % k_mpc) == 0)
    {
        // Indexes of feet in swing phase
        feet.clear();
        for (int i = 0; i < 4; i++)
        {
            if (gait_->getCurrentGait()(0, 1 + i) == 0)
                feet.push_back(i);
        }
        // If no foot in swing phase
        if (feet.size() == 0)
            return;

        // For each foot in swing phase get remaining duration of the swing phase
        for (int j = 0; j < (int)feet.size(); j++)
        {
            int i = feet[j];
            t_swing[i] = gait_->getPhaseDuration(0, feet[j], 0.0);  // 0.0 for swing phase
            double value = t_swing[i] - (gait_->getRemainingTime() * k_mpc - ((k + 1) % k_mpc)) * dt_tsid - dt_tsid;
            t0s[i] = std::max(0.0, value);
        }
    }
    else
    {
        // If no foot in swing phase
        if (feet.size() == 0)
            return;

        // Increment of one time step for feet in swing phase
        for (int i = 0; i < (int)feet.size(); i++)
        {
            double value = t0s[feet[i]] + dt_tsid;
            t0s[feet[i]] = std::max(0.0, value);
        }
    }

    for (int i = 0; i < (int)feet.size(); i++)
    {
        updateFootPosition(feet[i], targetFootstep.col(feet[i]));
    }
    return;
}
