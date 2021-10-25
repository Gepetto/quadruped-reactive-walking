#include "qrw/FootTrajectoryGeneratorBezier.hpp"
#include <chrono>

using namespace std::chrono;


// Trajectory generator functions (output reference pos, vel and acc of feet in swing phase)

FootTrajectoryGeneratorBezier::FootTrajectoryGeneratorBezier()
    : gait_(NULL),
      dt_wbc(0.0),
      k_mpc(0),
      maxHeight_(0.0),
      lockTime_(0.0),
      feet(),
      t0s(Vector4::Zero()),
      t0_bezier(Vector4::Zero()),
      t_stop(Vector4::Zero()),
      t_swing(Vector4::Zero()),
      targetFootstep_(Matrix34::Zero()),
      Ax(Matrix64::Zero()),
      Ay(Matrix64::Zero()),
      Az(Matrix74::Zero()),
      position_(Matrix34::Zero()),
      velocity_(Matrix34::Zero()),
      acceleration_(Matrix34::Zero()),
      jerk_(Matrix34::Zero()),
      position_base_(Matrix34::Zero()),
      velocity_base_(Matrix34::Zero()),
      acceleration_base_(Matrix34::Zero()),
      intersectionPoint_(Vector2::Zero()),
      ineq_vector_{Vector4::Zero()},
      x_margin_{Vector4::Zero()} {
  // Initialise vector
  for (int i = 0; i < 4; i++) {
    pDefs.push_back(optimization::problem_definition<pointX_t, double>(3));
    pDefs[i].degree = 7;
    pDefs[i].flag = optimization::INIT_POS | optimization::END_POS | optimization::INIT_VEL | optimization::END_VEL |
                    optimization::INIT_ACC | optimization::END_ACC;
    Vector6 vector = Vector6::Zero();
    problem_data_t pbData = optimization::setup_control_points<pointX_t, double, safe>(pDefs[0]);
    bezier_linear_variable_t* bez = pbData.bezier;

    bezier_t bezi = evaluateLinear<bezier_t, bezier_linear_variable_t>(*bez, vector);
    fitBeziers.push_back(bezi);

    ineq_.push_back(Vector3::Zero());
  }
}

void FootTrajectoryGeneratorBezier::initialize(Params& params, Gait& gaitIn, Surface initialSurface_in,
                                               double x_margin_max_in, double t_margin_in, double z_margin_in,
                                               int N_samples_in, int N_samples_ineq_in, int degree_in) {
  N_samples = N_samples_in;
  N_samples_ineq = N_samples_ineq_in;
  degree = degree_in;
  res_size = dim * (degree + 1 - 6);

  P_ = MatrixN::Zero(res_size, res_size);
  q_ = VectorN::Zero(res_size);
  C_ = MatrixN::Zero(res_size, 0);
  d_ = VectorN::Zero(0);
  x = VectorN::Zero(res_size);
  G_ = MatrixN::Zero(N_samples_ineq, res_size);
  h_ = VectorN::Zero(N_samples_ineq);
  dt_wbc = params.dt_wbc;
  k_mpc = (int)std::round(params.dt_mpc / params.dt_wbc);
  maxHeight_ = params.max_height;
  lockTime_ = params.lock_time;
  targetFootstep_ <<
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params.footsteps_init.data(), params.footsteps_init.size());
  position_ <<
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params.footsteps_init.data(), params.footsteps_init.size());
  gait_ = &gaitIn;
  for (int foot = 0; foot < 4; foot++) {
    newSurface_.push_back(initialSurface_in);
    pastSurface_.push_back(initialSurface_in);
  }
  x_margin_max_ = x_margin_max_in;
  t_margin_ = t_margin_in;  // 1 % of the curve after critical point
  z_margin_ = z_margin_in;
}

void FootTrajectoryGeneratorBezier::updatePolyCoeff_XY(int const& i_foot, Vector3 const& x_init, Vector3 const& v_init,
                                                       Vector3 const& a_init, Vector3 const& x_target,
                                                       double const& t0, double const& t1) {
  double x0 = x_init(0);
  double y0 = x_init(1);
  double dx0 = v_init(0);
  double dy0 = v_init(1);
  double ddx0 = a_init(0);
  double ddy0 = a_init(1);
  double x1 = x_target(0);
  double y1 = x_target(1);

  double d = t1;
  double t = t0;

  // Compute polynoms coefficients for x and y
  Ax(5, i_foot) =
      (ddx0 * std::pow(t, 2) - 2 * ddx0 * t * d - 6 * dx0 * t + ddx0 * std::pow(d, 2) + 6 * dx0 * d + 12 * x0 -
       12 * x1) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));

  Ax(4, i_foot) =
      (30 * t * x1 - 30 * t * x0 - 30 * d * x0 + 30 * d * x1 - 2 * std::pow(t, 3) * ddx0 - 3 * std::pow(d, 3) * ddx0 +
       14 * std::pow(t, 2) * dx0 - 16 * std::pow(d, 2) * dx0 + 2 * t * d * dx0 + 4 * t * std::pow(d, 2) * ddx0 +
       std::pow(t, 2) * d * ddx0) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ax(3, i_foot) =
      (std::pow(t, 4) * ddx0 + 3 * std::pow(d, 4) * ddx0 - 8 * std::pow(t, 3) * dx0 + 12 * std::pow(d, 3) * dx0 +
       20 * std::pow(t, 2) * x0 - 20 * std::pow(t, 2) * x1 + 20 * std::pow(d, 2) * x0 - 20 * std::pow(d, 2) * x1 +
       80 * t * d * x0 - 80 * t * d * x1 + 4 * std::pow(t, 3) * d * ddx0 + 28 * t * std::pow(d, 2) * dx0 -
       32 * std::pow(t, 2) * d * dx0 - 8 * std::pow(t, 2) * std::pow(d, 2) * ddx0) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ax(2, i_foot) = -(std::pow(d, 5) * ddx0 + 4 * t * std::pow(d, 4) * ddx0 + 3 * std::pow(t, 4) * d * ddx0 +
                    36 * t * std::pow(d, 3) * dx0 - 24 * std::pow(t, 3) * d * dx0 + 60 * t * std::pow(d, 2) * x0 +
                    60 * std::pow(t, 2) * d * x0 - 60 * t * std::pow(d, 2) * x1 - 60 * std::pow(t, 2) * d * x1 -
                    8 * std::pow(t, 2) * std::pow(d, 3) * ddx0 - 12 * std::pow(t, 2) * std::pow(d, 2) * dx0) /
                  (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                   (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ax(1, i_foot) =
      -(2 * std::pow(d, 5) * dx0 - 2 * t * std::pow(d, 5) * ddx0 - 10 * t * std::pow(d, 4) * dx0 +
        std::pow(t, 2) * std::pow(d, 4) * ddx0 + 4 * std::pow(t, 3) * std::pow(d, 3) * ddx0 -
        3 * std::pow(t, 4) * std::pow(d, 2) * ddx0 - 16 * std::pow(t, 2) * std::pow(d, 3) * dx0 +
        24 * std::pow(t, 3) * std::pow(d, 2) * dx0 - 60 * std::pow(t, 2) * std::pow(d, 2) * x0 +
        60 * std::pow(t, 2) * std::pow(d, 2) * x1) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ax(0, i_foot) = (2 * x1 * std::pow(t, 5) - ddx0 * std::pow(t, 4) * std::pow(d, 3) - 10 * x1 * std::pow(t, 4) * d +
                   2 * ddx0 * std::pow(t, 3) * std::pow(d, 4) + 8 * dx0 * std::pow(t, 3) * std::pow(d, 3) +
                   20 * x1 * std::pow(t, 3) * std::pow(d, 2) - ddx0 * std::pow(t, 2) * std::pow(d, 5) -
                   10 * dx0 * std::pow(t, 2) * std::pow(d, 4) - 20 * x0 * std::pow(t, 2) * std::pow(d, 3) +
                   2 * dx0 * t * std::pow(d, 5) + 10 * x0 * t * std::pow(d, 4) - 2 * x0 * std::pow(d, 5)) /
                  (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                   (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));

  Ay(5, i_foot) =
      (ddy0 * std::pow(t, 2) - 2 * ddy0 * t * d - 6 * dy0 * t + ddy0 * std::pow(d, 2) + 6 * dy0 * d + 12 * y0 -
       12 * y1) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ay(4, i_foot) =
      (30 * t * y1 - 30 * t * y0 - 30 * d * y0 + 30 * d * y1 - 2 * std::pow(t, 3) * ddy0 - 3 * std::pow(d, 3) * ddy0 +
       14 * std::pow(t, 2) * dy0 - 16 * std::pow(d, 2) * dy0 + 2 * t * d * dy0 + 4 * t * std::pow(d, 2) * ddy0 +
       std::pow(t, 2) * d * ddy0) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ay(3, i_foot) =
      (std::pow(t, 4) * ddy0 + 3 * std::pow(d, 4) * ddy0 - 8 * std::pow(t, 3) * dy0 + 12 * std::pow(d, 3) * dy0 +
       20 * std::pow(t, 2) * y0 - 20 * std::pow(t, 2) * y1 + 20 * std::pow(d, 2) * y0 - 20 * std::pow(d, 2) * y1 +
       80 * t * d * y0 - 80 * t * d * y1 + 4 * std::pow(t, 3) * d * ddy0 + 28 * t * std::pow(d, 2) * dy0 -
       32 * std::pow(t, 2) * d * dy0 - 8 * std::pow(t, 2) * std::pow(d, 2) * ddy0) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ay(2, i_foot) = -(std::pow(d, 5) * ddy0 + 4 * t * std::pow(d, 4) * ddy0 + 3 * std::pow(t, 4) * d * ddy0 +
                    36 * t * std::pow(d, 3) * dy0 - 24 * std::pow(t, 3) * d * dy0 + 60 * t * std::pow(d, 2) * y0 +
                    60 * std::pow(t, 2) * d * y0 - 60 * t * std::pow(d, 2) * y1 - 60 * std::pow(t, 2) * d * y1 -
                    8 * std::pow(t, 2) * std::pow(d, 3) * ddy0 - 12 * std::pow(t, 2) * std::pow(d, 2) * dy0) /
                  (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                   (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ay(1, i_foot) =
      -(2 * std::pow(d, 5) * dy0 - 2 * t * std::pow(d, 5) * ddy0 - 10 * t * std::pow(d, 4) * dy0 +
        std::pow(t, 2) * std::pow(d, 4) * ddy0 + 4 * std::pow(t, 3) * std::pow(d, 3) * ddy0 -
        3 * std::pow(t, 4) * std::pow(d, 2) * ddy0 - 16 * std::pow(t, 2) * std::pow(d, 3) * dy0 +
        24 * std::pow(t, 3) * std::pow(d, 2) * dy0 - 60 * std::pow(t, 2) * std::pow(d, 2) * y0 +
        60 * std::pow(t, 2) * std::pow(d, 2) * y1) /
      (2 * std::pow((t - d), 2) * (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
  Ay(0, i_foot) = (2 * y1 * std::pow(t, 5) - ddy0 * std::pow(t, 4) * std::pow(d, 3) - 10 * y1 * std::pow(t, 4) * d +
                   2 * ddy0 * std::pow(t, 3) * std::pow(d, 4) + 8 * dy0 * std::pow(t, 3) * std::pow(d, 3) +
                   20 * y1 * std::pow(t, 3) * std::pow(d, 2) - ddy0 * std::pow(t, 2) * std::pow(d, 5) -
                   10 * dy0 * std::pow(t, 2) * std::pow(d, 4) - 20 * y0 * std::pow(t, 2) * std::pow(d, 3) +
                   2 * dy0 * t * std::pow(d, 5) + 10 * y0 * t * std::pow(d, 4) - 2 * y0 * std::pow(d, 5)) /
                  (2 * (std::pow(t, 2) - 2 * t * d + std::pow(d, 2)) *
                   (std::pow(t, 3) - 3 * std::pow(t, 2) * d + 3 * t * std::pow(d, 2) - std::pow(d, 3)));
}

void FootTrajectoryGeneratorBezier::updatePolyCoeff_Z(int const& i_foot, Vector3 const& x_init,
                                                      Vector3 const& x_target, double const& t1, double const& h) {
  double z0 = x_init(2);
  double z1 = x_target(2);

  //  coefficients for z (deterministic)
  //  Version 2D (z1 = 0)
  //  Az[6,i_foot] = -h/((t1/2)**3*(t1 - t1/2)**3)
  //  Az[5,i_foot]  = (3*t1*h)/((t1/2)**3*(t1 - t1/2)**3)
  //  Az[4,i_foot]  = -(3*t1**2*h)/((t1/2)**3*(t1 - t1/2)**3)
  //  Az[3,i_foot]  = (t1**3*h)/((t1/2)**3*(t1 - t1/2)**3)
  //  Az[:3,i_foot] = 0

  //  Version 3D (z1 != 0)
  Az(6, i_foot) = (32. * z0 + 32. * z1 - 64. * h) / std::pow(t1, 6);
  Az(5, i_foot) = -(102. * z0 + 90. * z1 - 192. * h) / std::pow(t1, 5);
  Az(4, i_foot) = (111. * z0 + 81. * z1 - 192. * h) / std::pow(t1, 4);
  Az(3, i_foot) = -(42. * z0 + 22. * z1 - 64. * h) / std::pow(t1, 3);
  Az(2, i_foot) = 0;
  Az(1, i_foot) = 0;
  Az(0, i_foot) = z0;
}

Vector3 FootTrajectoryGeneratorBezier::evaluateBezier(int const& i_foot, int const& indice, double const& t) {
  double t1 = t_swing(i_foot);
  double delta_t = t1 - t0_bezier(i_foot);
  double t_b = std::min((t - t0_bezier(i_foot)) / (t1 - t0_bezier(i_foot)), 1.);

  if (indice == 0) {
    return fitBeziers[i_foot](t_b);
  } else if (indice == 1) {
    return fitBeziers[i_foot].derivate(t_b, 1) / delta_t;
  } else if (indice == 2) {
    return fitBeziers[i_foot].derivate(t_b, 2) / std::pow(delta_t, 2);
  } else {
    Vector3 vector = Vector3::Zero();
    return vector;
  }
}

Vector3 FootTrajectoryGeneratorBezier::evaluatePoly(int const& i_foot, int const& indice, double const& t) {
  Vector3 vector = Vector3::Zero();
  if (indice == 0) {
    double x = Ax(0, i_foot) + Ax(1, i_foot) * t + Ax(2, i_foot) * std::pow(t, 2) + Ax(3, i_foot) * std::pow(t, 3) +
               Ax(4, i_foot) * std::pow(t, 4) + Ax(5, i_foot) * std::pow(t, 5);
    double y = Ay(0, i_foot) + Ay(1, i_foot) * t + Ay(2, i_foot) * std::pow(t, 2) + Ay(3, i_foot) * std::pow(t, 3) +
               Ay(4, i_foot) * std::pow(t, 4) + Ay(5, i_foot) * std::pow(t, 5);
    double z = Az(0, i_foot) + Az(1, i_foot) * t + Az(2, i_foot) * std::pow(t, 2) + Az(3, i_foot) * std::pow(t, 3) +
               Az(4, i_foot) * std::pow(t, 4) + Az(5, i_foot) * std::pow(t, 5) + Az(6, i_foot) * std::pow(t, 6);
    vector << x, y, z;
  }

  if (indice == 1) {
    double vx = Ax(1, i_foot) + 2 * Ax(2, i_foot) * t + 3 * Ax(3, i_foot) * std::pow(t, 2) +
                4 * Ax(4, i_foot) * std::pow(t, 3) + 5 * Ax(5, i_foot) * std::pow(t, 4);
    double vy = Ay(1, i_foot) + 2 * Ay(2, i_foot) * t + 3 * Ay(3, i_foot) * std::pow(t, 2) +
                4 * Ay(4, i_foot) * std::pow(t, 3) + 5 * Ay(5, i_foot) * std::pow(t, 4);
    double vz = Az(1, i_foot) + 2 * Az(2, i_foot) * t + 3 * Az(3, i_foot) * std::pow(t, 2) +
                4 * Az(4, i_foot) * std::pow(t, 3) + 5 * Az(5, i_foot) * std::pow(t, 4) +
                6 * Az(6, i_foot) * std::pow(t, 5);
    vector << vx, vy, vz;
  }

  if (indice == 2) {
    double ax = 2 * Ax(2, i_foot) + 6 * Ax(3, i_foot) * t + 12 * Ax(4, i_foot) * std::pow(t, 2) +
                20 * Ax(5, i_foot) * std::pow(t, 3);
    double ay = 2 * Ay(2, i_foot) + 6 * Ay(3, i_foot) * t + 12 * Ay(4, i_foot) * std::pow(t, 2) +
                20 * Ay(5, i_foot) * std::pow(t, 3);
    double az = 2 * Az(2, i_foot) + 6 * Az(3, i_foot) * t + 12 * Az(4, i_foot) * std::pow(t, 2) +
                20 * Az(5, i_foot) * std::pow(t, 3) + 30 * Az(6, i_foot) * std::pow(t, 4);
    vector << ax, ay, az;
  }

  return vector;
}

void FootTrajectoryGeneratorBezier::updateFootPosition(int const& k, int const& i_foot, Vector3 const& targetFootstep) {
  double t0 = t0s[i_foot];
  double t1 = t_swing[i_foot];
  double h = maxHeight_;
  double delta_t = t1 - t0;
  double dt = dt_wbc;

  if (t0 < t1 - lockTime_) {
    // compute reference polynoms coefficients
    if (t0s[i_foot] < 10e-4 || k == 0) {
      // Update Z coefficients only at the beginning of the flying phase
      updatePolyCoeff_Z(i_foot, position_.col(i_foot), targetFootstep, t1, h + targetFootstep[2]);
      // Initale velocity and acceleration nulle
      updatePolyCoeff_XY(i_foot, position_.col(i_foot), Vector3::Zero(), Vector3::Zero(), targetFootstep, t0, t1);

      // Update initial conditions of the Problem Definition
      Vector3 vector;
      vector << position_(0, i_foot), position_(1, i_foot), evaluatePoly(i_foot, 0, t0)(2);
      pDefs[i_foot].init_pos = vector;

      vector << 0., 0., delta_t * evaluatePoly(i_foot, 1, t0)(2);
      pDefs[i_foot].init_vel = vector;

      vector << 0., 0., std::pow(delta_t, 2) * evaluatePoly(i_foot, 2, t0)(2);
      pDefs[i_foot].init_acc = vector;

      // New swing phase --> ineq surface
      t_stop[i_foot] = 0.;


      if ((newSurface_[i_foot].getHeight(targetFootstep.head(2)) -
           pastSurface_[i_foot].getHeight(targetFootstep.head(2))) >= 10e-3)
      // Only uphill
      {
        std::cout << "\n\n\n\n\n--------------------" << std::endl;
        std::cout << "DIFF SURFACES" << std::endl;
        std::cout << "newSurface_[i_foot].getb" << std::endl;
        std::cout << newSurface_[i_foot].getb() << std::endl;
        std::cout << "pastSurface_[i_foot].getb" << std::endl;
        std::cout << pastSurface_[i_foot].getb() << std::endl;

        int nb_vert = newSurface_[i_foot].vertices_.rows();
        MatrixN vert = newSurface_[i_foot].vertices_;

        Vector2 P1 = position_.col(i_foot).head(2);
        Vector2 P2 = targetFootstep.head(2);

        Vector2 Q1;
        Vector2 Q2;
        for (int l = 0; l < nb_vert; l++) {
          Q1 << vert(l, 0), vert(l, 1);
          if (l < nb_vert - 1) {
            Q2 << vert(l + 1, 0), vert(l + 1, 1);
          } else {
            Q2 << vert(0, 0), vert(0, 1);
          }
          if (doIntersect_segment(P1, P2, Q1, Q2)) {
            get_intersect_segment(P1, P2, Q1, Q2);
            //  Should be sorted
            //  self.ineq[i_foot] = surface.ineq[k, :]
            //  self.ineq_vect[i_foot] = surface.ineq_vect[k]
            double a = 0.;
            if ((Q1[0] - Q2[0]) != 0.) {
              a = (Q2[1] - Q1[1]) / (Q2[0] - Q1[0]);
              double b = Q1[1] - a * Q1[0];
              ineq_[i_foot] << -a, 1., 0.;  // -ax + y = b
              ineq_vector_[i_foot] = b;
            } else {
              //  Inequality of the surface corresponding to these vertices
              ineq_[i_foot] << -1., 0., 0.;
              ineq_vector_[i_foot] = -Q1[0];
            }
            if (ineq_[i_foot].transpose() * position_.col(i_foot) > ineq_vector_[i_foot]) {
              // Wrong side, the targeted point is inside the surface
              ineq_[i_foot] = -ineq_[i_foot];
              ineq_vector_[i_foot] = -ineq_vector_[i_foot];
            }

            // If foot position already closer than margin
            x_margin_[i_foot] = std::max(std::min(x_margin_max_, std::abs(intersectionPoint_[0] - P1[0]) - 0.001), 0.);
          }
        }
      } else {
        ineq_[i_foot].setZero();
        ineq_vector_[i_foot] = 0.;
      }
    } else {
      updatePolyCoeff_XY(i_foot, position_.col(i_foot), velocity_.col(i_foot), acceleration_.col(i_foot),
                         targetFootstep, t0, t1);
      // Update initial conditions of the Problem Definition
      pDefs[i_foot].init_pos = position_.col(i_foot);
      pDefs[i_foot].init_vel = delta_t * velocity_.col(i_foot);
      pDefs[i_foot].init_acc = std::pow(delta_t, 2) * acceleration_.col(i_foot);
    }
    // REset inequalities to zero
    G_.setZero();
    for (int l = 0; l < h_.size(); l++) {
      h_(l) = 0.;
    }
    // Update final conditions of the Problem Definition
    pDefs[i_foot].end_pos = targetFootstep;
    pDefs[i_foot].end_vel = Vector3::Zero();
    pDefs[i_foot].end_acc = Vector3::Zero();

    pDefs[i_foot].flag = optimization::INIT_POS | optimization::END_POS | optimization::INIT_VEL |
                         optimization::END_VEL | optimization::INIT_ACC | optimization::END_ACC;

    // generates the linear variable of the bezier curve with the parameters of problemDefinition
    problem_data_t pbData = optimization::setup_control_points<pointX_t, double, safe>(pDefs[i_foot]);
    bezier_linear_variable_t* linear_bezier = pbData.bezier;

    // Prepare the inequality matrix :
    Vector3 x_t = evaluatePoly(i_foot, 0, t0);
    double t_margin = t_margin_ * t1;  // 10% around the limit point !inferior to 1/nb point in linspace

    // No surface switch or already overpass the critical point
    if (!(ineq_[i_foot].isZero()) and ((x_t[2] < targetFootstep(2)) or (t0s[i_foot] < t_stop[i_foot] + t_margin)) and
        x_margin_[i_foot] != 0.) {
      int nb_vert = newSurface_[i_foot].vertices_.rows();
      MatrixN vert = newSurface_[i_foot].vertices_;

      Vector2 P1 = position_.col(i_foot).head(2);
      Vector2 P2 = targetFootstep.head(2);

      Vector2 Q1;
      Vector2 Q2;
      for (int l = 0; l < nb_vert; l++) {
        Q1 << vert(l, 0), vert(l, 1);
        if (l < nb_vert - 1) {
          Q2 << vert(l + 1, 0), vert(l + 1, 1);
        } else {
          Q2 << vert(0, 0), vert(0, 1);
        }
        if (doIntersect_segment(P1, P2, Q1, Q2)) {
          get_intersect_segment(P1, P2, Q1, Q2);
          //  Should be sorted
          //  self.ineq[i_foot] = surface.ineq[k, :]
          //  self.ineq_vect[i_foot] = surface.ineq_vect[k]
          double a = 0.;
          if ((Q1[0] - Q2[0]) != 0.) {
            a = (Q2[1] - Q1[1]) / (Q2[0] - Q1[0]);
            double b = Q1[1] - a * Q1[0];
            ineq_[i_foot] << -a, 1., 0.;  // -ax + y = b
            ineq_vector_[i_foot] = b;
          } else {
            //  Inequality of the surface corresponding to these vertices
            ineq_[i_foot] << -1., 0., 0.;
            ineq_vector_[i_foot] = -Q1[0];
          }
          if (ineq_[i_foot].transpose() * position_.col(i_foot) > ineq_vector_[i_foot]) {
            // Wrong side, the targeted point is inside the surface
            ineq_[i_foot] = -ineq_[i_foot];
            ineq_vector_[i_foot] = -ineq_vector_[i_foot];
          }
          // If foot position already closer than margin
          x_margin_[i_foot] = std::max(std::min(x_margin_max_, std::abs(intersectionPoint_[0] - P1[0]) - 0.001), 0.);
        }
      }

      double z_margin = targetFootstep(2) * z_margin_;  // 10% around the limit height
      double t_s;
      double zt;

      linear_variable_t linear_var_;

      for (int its = 0; its < N_samples_ineq; its++) {
        t_s = (its + 1.0) / N_samples_ineq;
        zt = evaluatePoly(i_foot, 0, t0 + (t1 - t0) * t_s)[2];
        if (t0 + (t1 - t0) * t_s < t_stop[i_foot] + t_margin) {
          if (zt < targetFootstep(2) + z_margin) {
            t_stop[i_foot] = t0 + (t1 - t0) * t_s;
          }
          linear_var_ = linear_bezier->operator()(t_s);
          G_.row(its) = -ineq_[i_foot].transpose() * linear_var_.B();
          h_(its) = -ineq_[i_foot].transpose() * linear_var_.c() + ineq_vector_[i_foot] - (x_margin_[i_foot]);
        } else {
          G_.row(its).setZero();
          h_(its) = 0.;
        }
      }

    } else {
      G_.setZero();
      for (int l = 0; l < h_.size(); l++) {
        h_(l) = 0.;
      }
    }

    P_.setZero();
    q_.setZero();
    linear_variable_t linear_var;
    double t_b_;
    for (int j = 0; j < N_samples; j++) {
      t_b_ = (j + 1.0) / N_samples;

      linear_var = linear_bezier->operator()(t_b_);

      P_ += linear_var.B().transpose() * linear_var.B();
      q_ += linear_var.B().transpose() * (linear_var.c() - evaluatePoly(i_foot, 0, t0 + (t1 - t0) * t_b_));
    }

    // Eiquadprog-Fast solves the problem :
    // min. 1/2 * x' C_ x + q_' x
    // s.t. C_ x + d_ = 0
    //      G_ x + h_ >= 0
    status = qp.solve_quadprog(P_, q_, C_, d_, G_, h_, x);

    // Evaluate Bezier Linear with optimsed Points
    fitBeziers[i_foot] = evaluateLinear<bezier_t, bezier_linear_variable_t>(*linear_bezier, x);

    t0_bezier[i_foot] = t0;
  }

  // Get the next point
  double ev = t0 + dt;

  double t_b = std::min((ev - t0_bezier[i_foot]) / (t1 - t0_bezier[i_foot]), 1.);
  delta_t = t1 - t0_bezier[i_foot];

  if (t0 < 0.0 || t0 > t1)  // Just vertical motion
  {
    position_(0, i_foot) = targetFootstep(0);
    position_(1, i_foot) = targetFootstep(1);
    position_(2, i_foot) = evaluatePoly(i_foot, 0, ev)[2];
    velocity_(0, i_foot) = 0.0;
    velocity_(1, i_foot) = 0.0;
    velocity_(2, i_foot) = evaluatePoly(i_foot, 1, ev)[2];
    acceleration_(0, i_foot) = 0.0;
    acceleration_(1, i_foot) = 0.0;
    acceleration_(2, i_foot) = evaluatePoly(i_foot, 2, ev)[2];
  } else {
    position_.col(i_foot) = fitBeziers[i_foot](t_b);
    velocity_.col(i_foot) = fitBeziers[i_foot].derivate(t_b, 1) / delta_t;
    acceleration_.col(i_foot) = fitBeziers[i_foot].derivate(t_b, 2) / std::pow(delta_t, 2);
    // position_.col(i_foot) = evaluatePoly(i_foot, 0, ev);
    // velocity_.col(i_foot) = evaluatePoly(i_foot, 1, ev);
    // acceleration_.col(i_foot) = evaluatePoly(i_foot, 2, ev);
  }
}

void FootTrajectoryGeneratorBezier::update(int k, MatrixN const& targetFootstep, SurfaceVector const& surfacesSelected,
                                           MatrixN const& currentPosition) {
  if ((k % k_mpc) == 0) {
    // Indexes of feet in swing phase
    feet.clear();
    for (int i = 0; i < 4; i++) {
      if (gait_->getCurrentGait()(0, i) == 0) feet.push_back(i);
    }
    // If no foot in swing phase
    if (feet.size() == 0) return;

    // For each foot in swing phase get remaining duration of the swing phase
    for (int j = 0; j < (int)feet.size(); j++) {
      int i = feet[j];
      t_swing[i] = gait_->getPhaseDuration(0, feet[j], 0.0);  // 0.0 for swing phase
      double value = t_swing[i] - (gait_->getRemainingTime() * k_mpc - ((k + 1) % k_mpc)) * dt_wbc - dt_wbc;
      t0s[i] = std::max(0.0, value);
    }
  } else {
    // If no foot in swing phase
    if (feet.size() == 0) return;

    // Increment of one time step for feet in swing phase
    for (int i = 0; i < (int)feet.size(); i++) {
      double value = t0s[feet[i]] + dt_wbc;
      t0s[feet[i]] = std::max(0.0, value);
    }
  }
  // Update new surface and past if t0 == 0 (new swing phase)
  if (((k % k_mpc) == 0) and (surfacesSelected.size() != 0)) {
    for (int i_foot = 0; i_foot < (int)feet.size(); i_foot++) {
      if (t0s[i_foot] <= 10e-5) {
        pastSurface_[i_foot] = newSurface_[i_foot];
        newSurface_[i_foot] = surfacesSelected[i_foot];
      }
    }
  }

  for (int i = 0; i < (int)feet.size(); i++) {
    position_.col(feet[i]) = currentPosition.col(feet[i]);
    updateFootPosition(k, feet[i], targetFootstep.col(feet[i]));
  }
  return;
}

bool FootTrajectoryGeneratorBezier::doIntersect_segment(Vector2 const& p1, Vector2 const& q1, Vector2 const& p2,
                                                        Vector2 const& q2) {
  //  Find the 4 orientations required for
  //  the general and special cases
  int o1 = orientation(p1, q1, p2);
  int o2 = orientation(p1, q1, q2);
  int o3 = orientation(p2, q2, p1);
  int o4 = orientation(p2, q2, q1);

  //  General case
  if ((o1 != o2) and (o3 != o4)) {
    return true;
  }

  //  Special Cases
  //  p1 , q1 and p2 are colinear and p2 lies on segment p1q1
  if ((o1 == 0) and onSegment(p1, p2, q1)) {
    return true;
  }

  //  p1 , q1 and q2 are colinear and q2 lies on segment p1q1
  if ((o2 == 0) and onSegment(p1, q2, q1)) {
    return true;
  }

  //  p2 , q2 and p1 are colinear and p1 lies on segment p2q2
  if ((o3 == 0) and onSegment(p2, p1, q2)) {
    return true;
  }

  //  p2 , q2 and q1 are colinear and q1 lies on segment p2q2
  if ((o4 == 0) and onSegment(p2, q1, q2)) {
    return true;
  }

  //  If none of the cases
  return false;
}

//  Given three colinear points p, q, r, the function checks if
//  point q lies on line segment 'pr'
bool FootTrajectoryGeneratorBezier::onSegment(Vector2 const& p, Vector2 const& q, Vector2 const& r) {
  if ((q[0] <= std::max(p[0], r[0])) and (q[0] >= std::min(p[0], r[0])) and (q[1] <= std::max(p[1], r[1])) and
      (q[1] >= std::min(p[1], r[1]))) {
    return true;
  }
  return false;
}

// to find the orientation of an ordered triplet (p,q,r)
// function returns the following values:
// 0 : Colinear points
// 1 : Clockwise points
// 2 : Counterclockwise
// See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
// for details of below formula.
// Modified, remove class Point, directly handle p = [px,py]
int FootTrajectoryGeneratorBezier::orientation(Vector2 const& p, Vector2 const& q, Vector2 const& r) {
  double val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]));
  if (val > 0) {
    // Clockwise orientation
    return 1;
  } else if (val < 0) {
    // Counterclockwise orientation
    return 2;
  } else {
    // Colinear orientation
    return 0;
  }
}

//  Method to intersect 2 segment --> useful to retrieve which inequality is crossed in a surface (2D)
// Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
// a1: [x, y] a point on the first line
// a2: [x, y] another point on the first line
// b1: [x, y] a point on the second line
// b2: [x, y] another point on the second line

void FootTrajectoryGeneratorBezier::get_intersect_segment(Vector2 a1, Vector2 a2, Vector2 b1, Vector2 b2) {
  Vector3 cross_l1;
  Vector3 cross_l2;
  Vector3 cross_ll;

  cross_l1 << a1[1] - a2[1], a1[0] - a2[0], a2[0] * a1[1] - a2[1] * a1[0];
  cross_l2 << b1[1] - b2[1], b1[0] - b2[0], b2[0] * b1[1] - b2[1] * b1[0];

  cross_ll << cross_l1[1] - cross_l2[1], cross_l1[0] - cross_l2[0],
      cross_l2[0] * cross_l1[1] - cross_l2[1] * cross_l1[0];

  intersectionPoint_(0) = cross_ll[0] / cross_ll[2];
  intersectionPoint_(1) = cross_ll[1] / cross_ll[2];
}

Eigen::MatrixXd FootTrajectoryGeneratorBezier::getFootPositionBaseFrame(const Eigen::Matrix<double, 3, 3>& R,
                                                                  const Eigen::Matrix<double, 3, 1>& T) {
  position_base_ =
      R * (position_ - T.replicate<1, 4>());  // Value saved because it is used to get velocity and acceleration
  return position_base_;
}

Eigen::MatrixXd FootTrajectoryGeneratorBezier::getFootVelocityBaseFrame(const Eigen::Matrix<double, 3, 3>& R,
                                                                  const Eigen::Matrix<double, 3, 1>& v_ref,
                                                                  const Eigen::Matrix<double, 3, 1>& w_ref) {
  velocity_base_ = R * velocity_ - v_ref.replicate<1, 4>() +
                   position_base_.colwise().cross(w_ref);  // Value saved because it is used to get acceleration
  return velocity_base_;
}

Eigen::MatrixXd FootTrajectoryGeneratorBezier::getFootAccelerationBaseFrame(const Eigen::Matrix<double, 3, 3>& R,
                                                                      const Eigen::Matrix<double, 3, 1>& w_ref,
                                                                      const Eigen::Matrix<double, 3, 1>& a_ref) {
  return R * acceleration_ - (position_base_.colwise().cross(w_ref)).colwise().cross(w_ref) +
         2 * velocity_base_.colwise().cross(w_ref) - a_ref.replicate<1, 4>();
}
