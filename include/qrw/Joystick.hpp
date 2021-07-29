#ifndef JOYSTICK_H_INCLUDED
#define JOYSTICK_H_INCLUDED

#include "qrw/Types.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>


class Joystick
{
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Empty constructor
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    Joystick();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Destructor.
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ~Joystick() {}

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Compute the remaining and total duration of a swing phase or a stance phase based
    ///        on the content of the gait matrix
    ///
    /// \param[in] k numero of the current loop
    /// \param[in] k_switch information about the position of key frames
    /// \param[in] v_switch information about the desired velocity for each key frame
    ///
    ////////////////////////////////////////////////////////////////////////////////////////////////
    VectorN handle_v_switch(double k, VectorN const& k_switch, MatrixN const& v_switch);

private:

    Vector6 A3_;
    Vector6 A2_;
    Vector6 v_ref_;
};

#endif  // JOYSTICK_H_INCLUDED
