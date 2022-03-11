///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Estimator and ComplementaryFilter classes
///
/// \details These classes estimate the state of the robot based on sensor measurements
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef COMPLEMENTARY_FILTER_H_INCLUDED
#define COMPLEMENTARY_FILTER_H_INCLUDED

#include "qrw/Types.h"

class ComplementaryFilter {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ComplementaryFilter();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Constructor witht initialization
  ///
  /// \param[in] dt Time step of the complementary filter
  /// \param[in] HighPass Initial value for the high pass filter
  /// \param[in] LowPass Initial value for the low pass filter
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ComplementaryFilter(double dt, Vector3 HighPass, Vector3 LowPass);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~ComplementaryFilter() {}  // Empty destructor

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initialize
  ///
  /// \param[in] dt Time step of the complementary filter
  /// \param[in] HighPass Initial value for the high pass filter
  /// \param[in] LowPass Initial value for the low pass filter
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(double dt, Vector3 HighPass, Vector3 LowPass);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Compute the filtered output of the complementary filter
  ///
  /// \param[in] x Quantity handled by the filter
  /// \param[in] dx Derivative of the quantity
  /// \param[in] alpha Filtering coefficient between x and dx quantities
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector3 compute(Vector3 const& x, Vector3 const& dx, Vector3 const& alpha);

  Vector3 getX() { return x_; }              // Get the input quantity
  Vector3 getDx() { return dx_; }            // Get the derivative of the input quantity
  Vector3 getHighPass() { return HighPass_; }     // Get the high-passed internal quantity
  Vector3 getLowPass() { return LowPass_; }      // Get the low-passed internal quantity
  Vector3 getAlpha() { return alpha_; }      // Get the alpha coefficient of the filter
  Vector3 getFilteredX() { return filteredX_; }  // Get the filtered output

 private:
  double dt_;          // Time step of the complementary filter
  Vector3 HighPass_;   // Initial value for the high pass filter
  Vector3 LowPass_;    // Initial value for the low pass filter
  Vector3 alpha_;      // Filtering coefficient between x and dx quantities
  Vector3 x_;          // Quantity to filter
  Vector3 dx_;         // Quantity to filter derivative's
  Vector3 filteredX_;  // Filtered output
};

#endif  // COMPLEMENTARY_FILTER_H_INCLUDED
