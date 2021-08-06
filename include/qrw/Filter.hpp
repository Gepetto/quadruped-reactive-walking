#ifndef FILTER_H_INCLUDED
#define FILTER_H_INCLUDED

#include <iostream>
#include <fstream>
#include <cmath>
#include <deque>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "qrw/Params.hpp"
#include "pinocchio/math/rpy.hpp"

class Filter
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Filter();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Initialize with given data
    ///
    /// \param[in] params Object that stores parameters
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void initialize(Params& params);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~Filter() {}  // Empty destructor

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Run one iteration of the filter
    ///
    /// \param[in] x Quantity to filter
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void filter(VectorN const& x);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Add or remove 2 PI to all elements in the queues to handle +- pi modulo
    ///
    /// \param[in] a Angle that needs change (3, 4, 5 for Roll, Pitch, Yaw respectively)
    /// \param[in] dir Direction of the change (+pi to -pi or -pi to +pi)
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    void handle_modulo(int a, bool dir);

    VectorN getFilt() { return y_; }

private:

    Vector5 b_;  // Denominator coefficients of the filter transfer function
    Vector5 a_;  // Numerator coefficients of the filter transfer function
    Vector6 x_;  // Latest measurement
    VectorN y_;  // Latest result
    Vector6 accum_;  // Used to compute the result (accumulation)

    std::deque<Vector6> x_queue_;  // Store the last measurements for product with denominator coefficients
    std::deque<Vector6> y_queue_;  // Store the last results for product with numerator coefficients

    bool init_;  // Initialisation flag

};

#endif  // FILTER_H_INCLUDED
