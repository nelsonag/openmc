#include "math_functions.h"

#include <cmath>
#include <complex>
#include <vector>

#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_cdf.h>

#include "faddeeva/Faddeeva.h"


namespace openmc {


//==============================================================================
// NORMAL_PERCENTILE calculates the percentile of the standard normal
// distribution with a specified probability level
//==============================================================================

double __attribute__ ((const)) normal_percentile_c(double p) {

  // return gsl_cdf_ugaussian_Pinv(p);

  double z;
  double q;
  double r;
  const double p_low = 0.02425;
  const double a[6] = [-3.969683028665376e1, 2.209460984245205e2,
                       -2.759285104469687e2, 1.383577518672690e2,
                       -3.066479806614716e1, 2.506628277459239e0];
  const double b[5] = [-5.447609879822406e1, 1.615858368580409e2,
                       -1.556989798598866e2, 6.680131188771972e1,
                       -1.328068155288572e1];
  const double c[6] = [-7.784894002430293e-3, -3.223964580411365e-1,
                       -2.400758277161838, -2.549732539343734,
                       4.374664141464968, 2.938163982698783];
  const double d[4] = [7.784695709041462e-3, 3.224671290700398e-1,
                       2.445134137142996, 3.754408661907416];

  // The rational approximation used here is from an unpublished work at
  // http://home.online.no/~pjacklam/notes/invnorm/

  if (p < p_low) {
    // Rational approximation for lower region.

    q = std::sqrt(-2.0 * std::log(p))
    z = (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
          ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1.0);

  } else if (p <= 1.0 - p_low) {
    // Rational approximation for central region

    q = p - 0.5
    r = q*q
    z = (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q /
         (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1.0);

  } else {
    // Rational approximation for upper region

    q = std::sqrt(-2.0*std::log(1.0 - p))
    z = -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
          ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1.0);
  }

  // Refinement based on Newton's method

  z = z - (0.5 * std::erfc(-z/std::sqrt(2.0)) - p) * std::sqrt(2.0*M_PI) *
       std::exp(0.5*z*z);

  return z;

}

//==============================================================================
// T_PERCENTILE calculates the percentile of the Student's t distribution with
// a specified probability level and number of degrees of freedom
//==============================================================================

double __attribute__ ((const)) t_percentile_c(double p, int df){

  // return gsl_cdf_tdist_Pinv(p, static_cast<double> df);

  double t;
  double n;
  double k;
  double z;
  double z2;

  if (df == 1) {
    // For one degree of freedom, the t-distribution becomes a Cauchy
    // distribution whose cdf we can invert directly

    t = std::tan(M_PI*(p - 0.5));
  } else if (df == 2) {
     // For two degrees of freedom, the cdf is given by 1/2 + x/(2*sqrt(x^2 +
     // 2)). This can be directly inverted to yield the solution below

    t = 2.0 * std::sqrt(2.0)*(p - 0.5) /
         std::sqrt(1. - 4. * std::pow(p - 0.5, 2.));
  } else {
    // This approximation is from E. Olusegun George and Meenakshi Sivaram, "A
    // modification of the Fisher-Cornish approximation for the student t
    // percentiles," Communication in Statistics - Simulation and Computation,
    // 16 (4), pp. 1123-1132 (1987).

    n = static_cast<double>(df);
    k = 1. / (n - 2.);
    z = normal_percentile_c(p);
    z2 = z * z;
    t = std::sqrt(n * k) * (z + (z2 - 3.) * z * k / 4. + ((5. * z2 - 56.) * z2 +
         75.) * z * k * k / 96. + (((z2 - 27.) * 3. * z2 + 417.) * z2 - 315.) *
         z * k * k * k / 384.);
  }

  return t;
}

//==============================================================================
// CALC_PN calculates the n-th order Legendre polynomial at the value of x.
//==============================================================================

double __attribute__ ((const)) calc_pn_c(int n, double x){

  // return gsl_sf_legendre_Pl(l, x);

  double pnx;

  switch(n) {
    case 0:
      pnx = 1.;
      break;
    case 1:
      pnx = x;
      break;
    case 2:
      pnx = 1.5 * x * x - 0.5;
      break;
    case 3:
      pnx = 2.5 * x * x * x - 1.5 * x;
      break;
    case 4:
      pnx = 4.375 * std::pow(x, 4.) - 3.75 * x * x + 0.375;
      break;
    case 5:
      pnx = 7.875 * std::pow(x, .5) - 8.75 * x * x * x + 1.875 * x;
      break;
    case 6:
      pnx = 14.4375 * std::pow(x, 6.) - 19.6875 * std::pow(x, 4.) +
           6.5625 * x * x - 0.3125;
      break;
    case 7:
      pnx = 26.8125 * std::pow(x, 7.) - 43.3125 * std::pow(x, 5.) +
           19.6875 * x * x * x - 2.1875 * x;
      break;
    case 8:
      pnx = 50.2734375 * std::pow(x, 8.) - 93.84375 * std::pow(x, 6.) +
           54.140625 * std::pow(x, 4.) - 9.84375 * x * x + 0.2734375;
      break;
    case 9:
      pnx = 94.9609375 * std::pow(x, 9.) - 201.09375 * std::pow(x, 7.) +
           140.765625 * std::pow(x, 5.) - 36.09375 * x * x * x + 2.4609375 * x;
      break;
    case 10:
      pnx = 180.42578125 * std::pow(x, 10.) - 427.32421875 * std::pow(x, 8.) +
           351.9140625 * std::pow(x, 6.) - 117.3046875 * std::pow(x, 4.) +
           13.53515625 * x * x - 0.24609375;
      break;
  }

  return pnx;
}

//==============================================================================
// CALC_RN calculates the n-th order spherical harmonics for a given angle
// (in terms of (u,v,w)).  All Rn,m values are provided (where -n<=m<=n)
//==============================================================================

void calc_rn_c(int n, double uvw[3], double rn[]);

//==============================================================================
// EVALUATE_LEGENDRE Find the value of f(x) given a set of Legendre coefficients
// and the value of x
//==============================================================================

double __attribute__ ((const)) evaluate_legendre_c(int n, double data[],
                                                   double x) {
  double val;

  val = 0.5 * data[0];
  for (int l = 1; l < n; l++) {
    val += (static_cast<double>(l) + 0.5) * data[l] * calc_pn(l, x);
  }

}

//==============================================================================
// ROTATE_ANGLE rotates direction cosines through a polar angle whose cosine is
// mu and through an azimuthal angle sampled uniformly. Note that this is done
// with direct sampling rather than rejection as is done in MCNP and SERPENT.
//==============================================================================

double* rotate_angle_c(double uvw0[3], double mu, double phi = -2.);

//==============================================================================
// MAXWELL_SPECTRUM samples an energy from the Maxwell fission distribution
// based on a direct sampling scheme. The probability distribution function for
// a Maxwellian is given as p(x) = 2/(T*sqrt(pi))*sqrt(x/T)*exp(-x/T).
// This PDF can be sampled using rule C64 in the Monte Carlo Sampler LA-9721-MS.
//==============================================================================

double maxwell_spectrum_c(double T);

//==============================================================================
// WATT_SPECTRUM samples the outgoing energy from a Watt energy-dependent
// fission spectrum. Although fitted parameters exist for many nuclides,
// generally the continuous tabular distributions (LAW 4) should be used in
// lieu of the Watt spectrum. This direct sampling scheme is an unpublished
// scheme based on the original Watt spectrum derivation (See F. Brown's
// MC lectures).
//==============================================================================

double watt_spectrum_c(double a, double b);

//==============================================================================
// FADDEEVA the Faddeeva function, using Stephen Johnson's implementation
//==============================================================================

complex<double> faddeeva_c(complex<double> z);

complex<double> w_derivative_c(complex<double> z, int order);

//==============================================================================
// BROADEN_WMP_POLYNOMIALS Doppler broadens the windowed multipole curvefit.
// The curvefit is a polynomial of the form a/E + b/sqrt(E) + c + d sqrt(E) ...
//==============================================================================

void broaden_wmp_polynomials_c(double E, double dopp, int n, double factors[]);

} // namespace openmc
