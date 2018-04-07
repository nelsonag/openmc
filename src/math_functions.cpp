#include "math_functions.h"

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
  const double a[6] = {-3.969683028665376e1, 2.209460984245205e2,
                       -2.759285104469687e2, 1.383577518672690e2,
                       -3.066479806614716e1, 2.506628277459239e0};
  const double b[5] = {-5.447609879822406e1, 1.615858368580409e2,
                       -1.556989798598866e2, 6.680131188771972e1,
                       -1.328068155288572e1};
  const double c[6] = {-7.784894002430293e-3, -3.223964580411365e-1,
                       -2.400758277161838, -2.549732539343734,
                       4.374664141464968, 2.938163982698783};
  const double d[4] = {7.784695709041462e-3, 3.224671290700398e-1,
                       2.445134137142996, 3.754408661907416};

  // The rational approximation used here is from an unpublished work at
  // http://home.online.no/~pjacklam/notes/invnorm/

  if (p < p_low) {
    // Rational approximation for lower region.

    q = std::sqrt(-2.0 * std::log(p));
    z = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
          ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);

  } else if (p <= 1.0 - p_low) {
    // Rational approximation for central region

    q = p - 0.5;
    r = q * q;
    z = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
         (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);

  } else {
    // Rational approximation for upper region

    q = std::sqrt(-2.0*std::log(1.0 - p));
    z = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
          ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
  }

  // Refinement based on Newton's method

  z = z - (0.5 * std::erfc(-z / std::sqrt(2.0)) - p) * std::sqrt(2.0 * M_PI) *
       std::exp(0.5 * z * z);

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

double __attribute__ ((const)) calc_pn_c(int n, double x) {

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
  }

  return pnx;
}

//==============================================================================
// CALC_RN calculates the n-th order spherical harmonics for a given angle
// (in terms of (u,v,w)).  All Rn,m values are provided (where -n<=m<=n)
//==============================================================================

void calc_rn_c(int n, double uvw[3], double rn[]){
  double phi;
  double w;
  double w2m1;

  // rn[] is assumed to have already been allocated to the correct size

  // Store the cosine of the polar angle and the azimuthal angle
  w = uvw[2];
  if (uvw[0] == 0.) {
    phi = 0.;
  } else {
    phi = std::atan2(uvw[1], uvw[0]);
  }

  // Store the shorthand of 1-w * w
  w2m1 = 1. - w * w;

  // Now evaluate the spherical harmonics function depending on the order
  // requested
  switch (n) {
    case 0:
      // l = 0, m = 0
      rn[0] = 1.;
      break;
    case 1:
      // l = 1, m = -1
      rn[0] = -(1.*std::sqrt(w2m1) * std::sin(phi));
      // l = 1, m = 0
      rn[1] = w;
      // l = 1, m = 1
      rn[2] = -(1.*std::sqrt(w2m1) * std::cos(phi));
      break;
    case 2:
      // l = 2, m = -2
      rn[0] = 0.288675134594813 * (-3. * w * w + 3.) * std::sin(2. * phi);
      // l = 2, m = -1
      rn[1] = -(1.73205080756888 * w*std::sqrt(w2m1) * std::sin(phi));
      // l = 2, m = 0
      rn[2] = 1.5 * w * w - 0.5;
      // l = 2, m = 1
      rn[3] = -(1.73205080756888 * w*std::sqrt(w2m1) * std::cos(phi));
      // l = 2, m = 2
      rn[4] = 0.288675134594813 * (-3. * w * w + 3.) * std::cos(2. * phi);
      break;
    case 3:
      // l = 3, m = -3
      rn[0] = -(0.790569415042095 * std::pow(w2m1, 1.5) * std::sin(3. * phi));
      // l = 3, m = -2
      rn[1] = 1.93649167310371 * w*(w2m1) * std::sin(2.*phi);
      // l = 3, m = -1
      rn[2] = -(0.408248290463863*std::sqrt(w2m1)*((7.5)*w * w - 3./2.) *
           std::sin(phi));
      // l = 3, m = 0
      rn[3] = 2.5 * std::pow(w, 3) - 1.5 * w;
      // l = 3, m = 1
      rn[4] = -(0.408248290463863*std::sqrt(w2m1)*((7.5)*w * w - 3./2.) *
           std::cos(phi));
      // l = 3, m = 2
      rn[5] = 1.93649167310371 * w*(w2m1) * std::cos(2.*phi);
      // l = 3, m = 3
      rn[6] = -(0.790569415042095 * std::pow(w2m1, 1.5) * std::cos(3.* phi));
      break;
    case 4:
      // l = 4, m = -4
      rn[0] = 0.739509972887452 * (w2m1 * w2m1) * std::sin(4.0*phi);
      // l = 4, m = -3
      rn[1] = -(2.09165006633519 * w * std::pow(w2m1, 1.5) * std::sin(3.* phi));
      // l = 4, m = -2
      rn[2] = 0.074535599249993 * (w2m1)*(52.5 * w * w - 7.5) * std::sin(2. *phi);
      // l = 4, m = -1
      rn[3] = -(0.316227766016838*std::sqrt(w2m1)*(17.5 * std::pow(w, 3) - 7.5 * w) *
           std::sin(phi));
      // l = 4, m = 0
      rn[4] = 4.375 * std::pow(w, 4) - 3.75 * w * w + 0.375;
      // l = 4, m = 1
      rn[5] = -(0.316227766016838*std::sqrt(w2m1)*(17.5 * std::pow(w, 3) - 7.5*w) *
           std::cos(phi));
      // l = 4, m = 2
      rn[6] = 0.074535599249993 * (w2m1)*(52.5*w * w - 7.5) * std::cos(2.*phi);
      // l = 4, m = 3
      rn[7] = -(2.09165006633519 * w * std::pow(w2m1, 1.5) * std::cos(3.* phi));
      // l = 4, m = 4
      rn[8] = 0.739509972887452 * w2m1 * w2m1 * std::cos(4.0*phi);
      break;
    case 5:
      // l = 5, m = -5
      rn[0] = -(0.701560760020114 * std::pow(w2m1, 2.5) * std::sin(5.0 * phi));
      // l = 5, m = -4
      rn[1] = 2.21852991866236 * w * w2m1 * w2m1 * std::sin(4.0 * phi);
      // l = 5, m = -3
      rn[2] = -(0.00996023841111995 * std::pow(w2m1, 1.5) *
           ((945.0 /2.)* w * w - 52.5) * std::sin(3.*phi));
      // l = 5, m = -2
      rn[3] = 0.0487950036474267 * (w2m1)
           * ((315.0/2.)* std::pow(w, 3) - 52.5 * w) * std::sin(2.*phi);
      // l = 5, m = -1
      rn[4] = -(0.258198889747161*std::sqrt(w2m1) *
           (39.375 * std::pow(w, 4) - 105.0/4.0 * w * w + 15.0/8.0) * std::sin(phi));
      // l = 5, m = 0
      rn[5] = 7.875 * std::pow(w, 5) - 8.75 * std::pow(w, 3) + 1.875 * w;
      // l = 5, m = 1
      rn[6] = -(0.258198889747161 * std::sqrt(w2m1)*
           (39.375 * std::pow(w, 4) - 105.0/4.0 * w * w + 15.0/8.0) * std::cos(phi));
      // l = 5, m = 2
      rn[7] = 0.0487950036474267 * (w2m1) *
           ((315.0 / 2.) * std::pow(w, 3) - 52.5*w) * std::cos(2.*phi);
      // l = 5, m = 3
      rn[8] = -(0.00996023841111995 * std::pow(w2m1, 1.5) *
           ((945.0 / 2.) * w * w - 52.5) * std::cos(3.*phi));
      // l = 5, m = 4
      rn[9] = 2.21852991866236 * w * w2m1 * w2m1 * std::cos(4.0*phi);
      // l = 5, m = 5
      rn[10] = -(0.701560760020114 * std::pow(w2m1, 2.5) * std::cos(5.0* phi));
      break;
    case 6:
      // l = 6, m = -6
      rn[0] = 0.671693289381396 * std::pow(w2m1, 3) * std::sin(6.0*phi);
      // l = 6, m = -5
      rn[1] = -(2.32681380862329 * w*std::pow(w2m1, 2.5) * std::sin(5.0*phi));
      // l = 6, m = -4
      rn[2] = 0.00104990131391452 * w2m1 * w2m1 *
           ((10395.0/2.) * w * w - 945.0/2.) * std::sin(4.0 * phi);
      // l = 6, m = -3
      rn[3] = -(0.00575054632785295 * std::pow(w2m1, 1.5) *
           ((3465.0/2.) * std::pow(w, 3) - 945.0/2.*w) * std::sin(3.*phi));
      // l = 6, m = -2
      rn[4] = 0.0345032779671177 * (w2m1) *
           ((3465.0/8.0)* std::pow(w, 4) - 945.0/4.0 * w * w + 105.0/8.0) *
           std::sin(2. * phi);
      // l = 6, m = -1
      rn[5] = -(0.218217890235992*std::sqrt(w2m1) *
           ((693.0/8.0)* std::pow(w, 5)- 315.0/4.0 * std::pow(w, 3) + (105.0/8.0)*w) *
           std::sin(phi));
      // l = 6, m = 0
      rn[6] = 14.4375 * std::pow(w, 6) - 19.6875 * std::pow(w, 4) + 6.5625 * w * w -
           0.3125;
      // l = 6, m = 1
      rn[7] = -(0.218217890235992*std::sqrt(w2m1) *
           ((693.0/8.0)* std::pow(w, 5)- 315.0/4.0 * std::pow(w, 3) + (105.0/8.0)*w) *
           std::cos(phi));
      // l = 6, m = 2
      rn[8] = 0.0345032779671177 * w2m1 *
           ((3465.0/8.0)* std::pow(w, 4) -945.0/4.0 * w * w + 105.0/8.0) *
           std::cos(2.*phi);
      // l = 6, m = 3
      rn[9] = -(0.00575054632785295 * std::pow(w2m1, 1.5) *
           ((3465.0/2.) * std::pow(w, 3) - 945.0/2.*w) * std::cos(3.*phi));
      // l = 6, m = 4
      rn[10] = 0.00104990131391452 * w2m1 * w2m1 *
           ((10395.0/2.)*w * w - 945.0/2.) * std::cos(4.0*phi);
      // l = 6, m = 5
      rn[11] = -(2.32681380862329 * w * std::pow(w2m1, 2.5) * std::cos(5.0*phi));
      // l = 6, m = 6
      rn[12] = 0.671693289381396 * std::pow(w2m1, 3) * std::cos(6.0*phi);
      break;
    case 7:
      // l = 7, m = -7
      rn[0] = -(0.647259849287749 * std::pow(w2m1, 3.5) * std::sin(7.0*phi));
      // l = 7, m = -6
      rn[1] = 2.42182459624969 * w*std::pow(w2m1, 3) * std::sin(6.0*phi);
      // l = 7, m = -5
      rn[2] = -(9.13821798555235e-5*std::pow(w2m1, 2.5) *
           ((135135.0/2.)*w * w - 10395.0/2.) * std::sin(5.0*phi));
      // l = 7, m = -4
      rn[3] = 0.000548293079133141 * w2m1 * w2m1 *
           ((45045.0/2.)*std::pow(w, 3) - 10395.0/2.*w) * std::sin(4.0*phi);
      // l = 7, m = -3
      rn[4] = -(0.00363696483726654 * std::pow(w2m1, 1.5) *
           ((45045.0/8.0)* std::pow(w, 4) - 10395.0/4.0 * w * w + 945.0/8.0) *
           std::sin(3.*phi));
      // l = 7, m = -2
      rn[5] = 0.025717224993682 * (w2m1) *
           ((9009.0/8.0)* std::pow(w, 5) -3465.0/4.0 * std::pow(w, 3) + (945.0/8.0)*w) *
           std::sin(2.*phi);
      // l = 7, m = -1
      rn[6] = -(0.188982236504614*std::sqrt(w2m1) *
           ((3003.0/16.0)* std::pow(w, 6) - 3465.0/16.0 * std::pow(w, 4) +
           (945.0/16.0)*w * w - 35.0/16.0) * std::sin(phi));
      // l = 7, m = 0
      rn[7] = 26.8125 * std::pow(w, 7) - 43.3125 * std::pow(w, 5) + 19.6875 * std::pow(w, 3) -
           2.1875 * w;
      // l = 7, m = 1
      rn[8] = -(0.188982236504614*std::sqrt(w2m1) * ((3003.0/16.0) * std::pow(w, 6) -
           3465.0/16.0 * std::pow(w, 4) + (945.0/16.0)*w * w - 35.0/16.0) * std::cos(phi));
      // l = 7, m = 2
      rn[9] = 0.025717224993682 * (w2m1) * ((9009.0/8.0)* std::pow(w, 5) -
           3465.0/4.0 * std::pow(w, 3) + (945.0/8.0)*w) * std::cos(2.*phi);
      // l = 7, m = 3
      rn[10] = -(0.00363696483726654 * std::pow(w2m1, 1.5) *
           ((45045.0/8.0)* std::pow(w, 4) - 10395.0/4.0 * w * w + 945.0/8.0) *
           std::cos(3.*phi));
      // l = 7, m = 4
      rn[11] = 0.000548293079133141 * w2m1 * w2m1 *
           ((45045.0/2.)*std::pow(w, 3) - 10395.0/2.*w) * std::cos(4.0*phi);
      // l = 7, m = 5
      rn[12] = -(9.13821798555235e-5*std::pow(w2m1, 2.5) *
           ((135135.0/2.)*w * w - 10395.0/2.) * std::cos(5.0*phi));
      // l = 7, m = 6
      rn[13] = 2.42182459624969 * w*std::pow(w2m1, 3) * std::cos(6.0*phi);
      // l = 7, m = 7
      rn[14] = -(0.647259849287749 * std::pow(w2m1, 3.5) * std::cos(7.0*phi));
      break;
    case 8:
      // l = 8, m = -8
      rn[0] = 0.626706654240044 * std::pow(w2m1, 4) * std::sin(8.0*phi);
      // l = 8, m = -7
      rn[1] = -(2.50682661696018 * w*std::pow(w2m1, 3.5) * std::sin(7.0*phi));
      // l = 8, m = -6
      rn[2] = 6.77369783729086e-6*std::pow(w2m1, 3)*
           ((2027025.0/2.)*w * w - 135135.0/2.) * std::sin(6.0*phi);
      // l = 8, m = -5
      rn[3] = -(4.38985792528482e-5*std::pow(w2m1, 2.5) *
                ((675675.0/2.)*std::pow(w, 3) - 135135.0/2.*w) * std::sin(5.0*phi));
      // l = 8, m = -4
      rn[4] = 0.000316557156832328 * w2m1 * w2m1 *
           ((675675.0/8.0)* std::pow(w, 4) - 135135.0/4.0 * w * w + 10395.0/8.0) *
           std::sin(4.0*phi);
      // l = 8, m = -3
      rn[5] = -(0.00245204119306875 * std::pow(w2m1, 1.5) * ((135135.0/8.0) *
           std::pow(w, 5) - 45045.0/4.0 * std::pow(w, 3) + (10395.0/8.0)*w) * std::sin(3.*phi));
      // l = 8, m = -2
      rn[6] = 0.0199204768222399 * (w2m1) *
           ((45045.0/16.0)* std::pow(w, 6)- 45045.0/16.0 * std::pow(w, 4) +
           (10395.0/16.0)*w * w - 315.0/16.0) * std::sin(2.*phi);
      // l = 8, m = -1
      rn[7] = -(0.166666666666667*std::sqrt(w2m1) *
           ((6435.0/16.0)* std::pow(w, 7) - 9009.0/16.0 * std::pow(w, 5) +
           (3465.0/16.0)*std::pow(w, 3) - 315.0/16.0 * w) * std::sin(phi));
      // l = 8, m = 0
      rn[8] = 50.2734375 * std::pow(w, 8) - 93.84375 * std::pow(w, 6) + 54.140625 *
           std::pow(w, 4) - 9.84375 * w * w + 0.2734375;
      // l = 8, m = 1
      rn[9] = -(0.166666666666667*std::sqrt(w2m1) *
                 ((6435.0/16.0)* std::pow(w, 7) - 9009.0/16.0 * std::pow(w, 5) +
                 (3465.0/16.0)*std::pow(w, 3) - 315.0/16.0 * w) * std::cos(phi));
      // l = 8, m = 2
      rn[10] = 0.0199204768222399 * (w2m1)*((45045.0/16.0)* std::pow(w, 6)-
           45045.0/16.0 * std::pow(w, 4) + (10395.0/16.0)*w * w -
           315.0/16.0) * std::cos(2.*phi);
      // l = 8, m = 3
      rn[11] = -(0.00245204119306875 * std::pow(w2m1, 1.5)*
                 ((135135.0/8.0) * std::pow(w, 5) - 45045.0/4.0 * std::pow(w, 3) +
                 (10395.0/8.0)*w) * std::cos(3.*phi));
      // l = 8, m = 4
      rn[12] = 0.000316557156832328 * w2m1 * w2m1*((675675.0/8.0)* std::pow(w, 4) -
           135135.0/4.0 * w * w + 10395.0/8.0) * std::cos(4.0*phi);
      // l = 8, m = 5
      rn[13] = -(4.38985792528482e-5*std::pow(w2m1, 2.5)*((675675.0/2.)*std::pow(w, 3) -
                 135135.0/2.*w) * std::cos(5.0*phi));
      // l = 8, m = 6
      rn[14] = 6.77369783729086e-6*std::pow(w2m1, 3)*((2027025.0/2.)*w * w -
           135135.0/2.) * std::cos(6.0*phi);
      // l = 8, m = 7
      rn[15] = -(2.50682661696018 * w*std::pow(w2m1, 3.5) * std::cos(7.0*phi));
      // l = 8, m = 8
      rn[16] = 0.626706654240044 * std::pow(w2m1, 4) * std::cos(8.0*phi);
      break;
    case 9:
      // l = 9, m = -9
      rn[0] = -(0.609049392175524 * std::pow(w2m1, 4.5) * std::sin(9.0 * phi));
      // l = 9, m = -8
      rn[1] = 2.58397773170915 * w*std::pow(w2m1, 4) * std::sin(8.0 * phi);
      // l = 9, m = -7
      rn[2] = -(4.37240315267812e-7*std::pow(w2m1, 3.5) *
           ((34459425.0/2.)*w * w - 2027025.0/2.) * std::sin(7.0 * phi));
      // l = 9, m = -6
      rn[3] = 3.02928976464514e-6*std::pow(w2m1, 3)*
           ((11486475.0/2.)*std::pow(w, 3) - 2027025.0/2.*w) * std::sin(6.0 * phi);
      // l = 9, m = -5
      rn[4] = -(2.34647776186144e-5*std::pow(w2m1, 2.5) *
           ((11486475.0/8.0)* std::pow(w, 4) - 2027025.0 / 4.0 * w * w +
            135135.0/8.0) * std::sin(5.0 * phi));
      // l = 9, m = -4
      rn[5] = 0.000196320414650061 * w2m1 * w2m1*((2297295.0/8.0)* std::pow(w, 5) -
           675675.0/4.0 * std::pow(w, 3) + (135135.0/8.0)*w) * std::sin(4.0*phi);
      // l = 9, m = -3
      rn[6] = -(0.00173385495536766 * std::pow(w2m1, 1.5) *
                ((765765.0/16.0)* std::pow(w, 6) - 675675.0/16.0 * std::pow(w, 4) +
                (135135.0/16.0)*w * w - 3465.0/16.0) * std::sin(3. * phi));
      // l = 9, m = -2
      rn[7] = 0.0158910431540932 * (w2m1)*((109395.0/16.0)* std::pow(w, 7)-
           135135.0/16.0 * std::pow(w, 5) + (45045.0/16.0)*std::pow(w, 3) -
           3465.0/16.0 * w) * std::sin(2. * phi);
      // l = 9, m = -1
      rn[8] = -(0.149071198499986*std::sqrt(w2m1)*((109395.0/128.0)* std::pow(w, 8) -
                45045.0/32.0 * std::pow(w, 6) + (45045.0/64.0)* std::pow(w, 4) -
                3465.0/32.0 * w * w + 315.0/128.0) * std::sin(phi));
      // l = 9, m = 0
      rn[9] = 94.9609375 * std::pow(w, 9) - 201.09375 * std::pow(w, 7) +
           140.765625 * std::pow(w, 5)- 36.09375 * std::pow(w, 3) + 2.4609375 * w;
      // l = 9, m = 1
      rn[10] = -(0.149071198499986*std::sqrt(w2m1)*((109395.0/128.0)* std::pow(w, 8) -
                 45045.0/32.0 * std::pow(w, 6) + (45045.0/64.0)* std::pow(w, 4) -
                 3465.0/32.0 * w * w + 315.0/128.0) * std::cos(phi));
      // l = 9, m = 2
      rn[11] = 0.0158910431540932 * (w2m1)*((109395.0/16.0)* std::pow(w, 7) -
           135135.0/16.0 * std::pow(w, 5) + (45045.0/16.0)*std::pow(w, 3) -
           3465.0/ 16.0 * w) * std::cos(2. * phi);
      // l = 9, m = 3
      rn[12] = -(0.00173385495536766 * std::pow(w2m1, 1.5)*((765765.0/16.0) *
                 std::pow(w, 6) - 675675.0/16.0 * std::pow(w, 4) +
                 (135135.0/16.0)* w * w - 3465.0/16.0)* std::cos(3. * phi));
      // l = 9, m = 4
      rn[13] = 0.000196320414650061 * w2m1 * w2m1*((2297295.0/8.0) * std::pow(w, 5) -
           675675.0/4.0 * std::pow(w, 3) + (135135.0/8.0)*w) * std::cos(4.0 * phi);
      // l = 9, m = 5
      rn[14] = -(2.34647776186144e-5*std::pow(w2m1, 2.5)*((11486475.0/8.0) *
                 std::pow(w, 4) - 2027025.0/4.0 * w * w + 135135.0/8.0) *
                 std::cos(5.0 * phi));
      // l = 9, m = 6
      rn[15] = 3.02928976464514e-6*std::pow(w2m1, 3)*((11486475.0/2.)*std::pow(w, 3) -
           2027025.0/2. * w) * std::cos(6.0 * phi);
      // l = 9, m = 7
      rn[16] = -(4.37240315267812e-7*std::pow(w2m1, 3.5)*
                 ((34459425.0/2.) * w * w - 2027025.0/2.) * std::cos(7.0 * phi));
      // l = 9, m = 8
      rn[17] = 2.58397773170915 * w*std::pow(w2m1, 4) * std::cos(8.0 * phi);
      // l = 9, m = 9
      rn[18] = -(0.609049392175524 * std::pow(w2m1, 4.5) * std::cos(9.0 * phi));
      break;
    case 10:
      // l = 10, m = -10
      rn[0] = 0.593627917136573 * std::pow(w2m1, 5) * std::sin(10.0 * phi);
      // l = 10, m = -9
      rn[1] = -(2.65478475211798 * w * std::pow(w2m1, 4.5) * std::sin(9.0 * phi));
      // l = 10, m = -8
      rn[2] = 2.49953651452314e-8 * std::pow(w2m1, 4) *
           ((654729075.0/2.) * w * w - 34459425.0/2.) * std::sin(8.0 * phi);
      // l = 10, m = -7
      rn[3] = -(1.83677671621093e-7*std::pow(w2m1, 3.5)*
                ((218243025.0/2.)*std::pow(w, 3) - 34459425.0/2.*w) *
                std::sin(7.0 * phi));
      // l = 10, m = -6
      rn[4] = 1.51464488232257e-6*std::pow(w2m1, 3)*((218243025.0/8.0)* std::pow(w, 4) -
           34459425.0/4.0 * w * w + 2027025.0/8.0) * std::sin(6.0 * phi);
      // l = 10, m = -5
      rn[5] = -(1.35473956745817e-5*std::pow(w2m1, 2.5)*
                ((43648605.0/8.0)* std::pow(w, 5) - 11486475.0/4.0 * std::pow(w, 3) +
                (2027025.0/8.0)*w) * std::sin(5.0 * phi));
      // l = 10, m = -4
      rn[6] = 0.000128521880085575 * w2m1 * w2m1*((14549535.0/16.0)* std::pow(w, 6) -
           11486475.0/16.0 * std::pow(w, 4) + (2027025.0/16.0)*w * w -
           45045.0/16.0) * std::sin(4.0 * phi);
      // l = 10, m = -3
      rn[7] = -(0.00127230170115096 * std::pow(w2m1, 1.5)*
                ((2078505.0/16.0)* std::pow(w, 7) - 2297295.0/16.0 * std::pow(w, 5) +
                (675675.0/16.0)*std::pow(w, 3) - 45045.0/16.0 * w) * std::sin(3. * phi));
      // l = 10, m = -2
      rn[8] = 0.012974982402692 * (w2m1)*((2078505.0/128.0)* std::pow(w, 8) -
           765765.0/32.0 * std::pow(w, 6) + (675675.0/64.0)* std::pow(w, 4) -
           45045.0/32.0 * w * w + 3465.0/128.0) * std::sin(2. * phi);
      // l = 10, m = -1
      rn[9] = -(0.134839972492648*std::sqrt(w2m1)*((230945.0/128.0)* std::pow(w, 9) -
                 109395.0/32.0 * std::pow(w, 7) + (135135.0/64.0)* std::pow(w, 5) -
                 15015.0/32.0 * std::pow(w, 3) + (3465.0/128.0)*w) * std::sin(phi));
      // l = 10, m = 0
      rn[10] = 180.42578125 * std::pow(w, 10) - 427.32421875 * std::pow(w, 8) +351.9140625
           * std::pow(w, 6) - 117.3046875 * std::pow(w, 4) + 13.53515625 * w * w -0.24609375;
      // l = 10, m = 1
      rn[11] = -(0.134839972492648*std::sqrt(w2m1)*((230945.0/128.0)* std::pow(w, 9) -
                 109395.0/32.0 * std::pow(w, 7) + (135135.0/64.0)* std::pow(w, 5) -15015.0/
                 32.0 * std::pow(w, 3) + (3465.0/128.0)*w) * std::cos(phi));
      // l = 10, m = 2
      rn[12] = 0.012974982402692 * (w2m1)*((2078505.0/128.0)* std::pow(w, 8) -
           765765.0/32.0 * std::pow(w, 6) + (675675.0/64.0)* std::pow(w, 4) -
           45045.0/32.0 * w * w + 3465.0/128.0) * std::cos(2. * phi);
      // l = 10, m = 3
      rn[13] = -(0.00127230170115096 * std::pow(w2m1, 1.5)*
                 ((2078505.0/16.0)* std::pow(w, 7) - 2297295.0/16.0 * std::pow(w, 5) +
                 (675675.0/16.0)*std::pow(w, 3) - 45045.0/16.0 * w) * std::cos(3. * phi));
      // l = 10, m = 4
      rn[14] = 0.000128521880085575 * w2m1 * w2m1*((14549535.0/16.0)* std::pow(w, 6) -
           11486475.0/16.0 * std::pow(w, 4) + (2027025.0/16.0) * w * w -
           45045.0/16.0) * std::cos(4.0 * phi);
      // l = 10, m = 5
      rn[15] = -(1.35473956745817e-5*std::pow(w2m1, 2.5)*
                 ((43648605.0/8.0)* std::pow(w, 5) - 11486475.0/4.0 * std::pow(w, 3) +
                 (2027025.0/8.0)*w) * std::cos(5.0 * phi));
      // l = 10, m = 6
      rn[16] = 1.51464488232257e-6*std::pow(w2m1, 3)*((218243025.0/8.0)* std::pow(w, 4) -
           34459425.0/4.0 * w * w + 2027025.0/8.0) * std::cos(6.0 * phi);
      // l = 10, m = 7
      rn[17] = -(1.83677671621093e-7*std::pow(w2m1, 3.5) *
           ((218243025.0/2.)*std::pow(w, 3) - 34459425.0/2.*w) * std::cos(7.0 * phi));
      // l = 10, m = 8
      rn[18] = 2.49953651452314e-8*std::pow(w2m1, 4)*
           ((654729075.0/2.)*w * w - 34459425.0/2.) * std::cos(8.0 * phi);
      // l = 10, m = 9
      rn[19] = -(2.65478475211798 * w*std::pow(w2m1, 4.5) * std::cos(9.0 * phi));
      // l = 10, m = 10
      rn[20] = 0.593627917136573 * std::pow(w2m1, 5) * std::cos(10.0 * phi);
  }
}

//==============================================================================
// EVALUATE_LEGENDRE Find the value of f(x) given a set of Legendre coefficients
// and the value of x
//==============================================================================

double __attribute__ ((const)) evaluate_legendre_c(int n, double data[],
                                                   double x) {
  double val;

  val = 0.5 * data[0];
  for (int l = 1; l < n; l++) {
    val += (static_cast<double>(l) + 0.5) * data[l] * calc_pn_c(l, x);
  }

}

//==============================================================================
// ROTATE_ANGLE rotates direction std::cosines through a polar angle whose std::cosine is
// mu and through an azimuthal angle sampled uniformly. Note that this is done
// with direct sampling rather than rejection as is done in MCNP and SERPENT.
//==============================================================================

void rotate_angle_c(double uvw[3], double mu, double phi) {
  double phi_;   // azimuthal angle
  double sinphi; // std::sine of azimuthal angle
  double cosphi; // cosine of azimuthal angle
  double a;      // sqrt(1 - mu^2)
  double b;      // sqrt(1 - w^2)
  double u0;     // original std::cosine in x direction
  double v0;     // original std::cosine in y direction
  double w0;     // original std::cosine in z direction

  // Copy original directional std::cosines
  u0 = uvw[0];
  v0 = uvw[1];
  w0 = uvw[2];

  // Sample azimuthal angle in [0,2pi) if none provided
  if (phi != -10.) {
    phi_ = phi;
  } else {
    phi_ = 2. * M_PI * prn();
  }

  // Precompute factors to save flops
  sinphi = std::sin(phi_);
  cosphi = std::cos(phi_);
  a = std::sqrt(std::max(0., 1. - mu * mu));
  b = std::sqrt(std::max(0., 1. - w0 * w0));

  // Need to treat special case where sqrt(1 - w**2) is close to zero by
  // expanding about the v component rather than the w component
  if (b > 1e-10) {
    uvw[0] = mu * u0 + a * (u0 * w0 * cosphi - v0 * sinphi) / b;
    uvw[1] = mu * v0 + a * (v0 * w0 * cosphi + u0 * sinphi) / b;
    uvw[2] = mu * w0 - a * b * cosphi;
  } else {
    b = std::sqrt(1. - v0 * v0);
    uvw[0] = mu * u0 + a * (u0 * v0 * cosphi + w0 * sinphi) / b;
    uvw[1] = mu * v0 - a * b * cosphi;
    uvw[2] = mu * w0 + a * (v0 * w0 * cosphi - u0 * sinphi) / b;
  }
}

//==============================================================================
// MAXWELL_SPECTRUM samples an energy from the Maxwell fission distribution
// based on a direct sampling scheme. The probability distribution function for
// a Maxwellian is given as p(x) = 2/(T*sqrt(pi))*sqrt(x/T)*exp(-x/T).
// This PDF can be sampled using rule C64 in the Monte Carlo Sampler LA-9721-MS.
//==============================================================================

double maxwell_spectrum_c(double T) {
  double E_out; // Sampled Energy

  double r1;
  double r2;
  double r3;  // random numbers
  double c;   // cosine of pi/2*r3

  r1 = prn();
  r2 = prn();
  r3 = prn();

  // determine cosine of pi/2*r
  c = std::cos(M_PI / 2. * r3);

  // determine outgoing energy
  E_out = -T * (std::log(r1) + std::log(r2) * c * c);

  return E_out;
}

//==============================================================================
// WATT_SPECTRUM samples the outgoing energy from a Watt energy-dependent
// fission spectrum. Although fitted parameters exist for many nuclides,
// generally the continuous tabular distributions (LAW 4) should be used in
// lieu of the Watt spectrum. This direct sampling scheme is an unpublished
// scheme based on the original Watt spectrum derivation (See F. Brown's
// MC lectures).
//==============================================================================

double watt_spectrum_c(double a, double b) {
  double E_out; // Sampled Energy
  double w;     // sampled from Maxwellian

  w = maxwell_spectrum_c(a);
  E_out = w + 0.25 * a * a * b + (2. * prn() - 1.) * std::sqrt(a * a * b * w);

  return E_out;
}

//==============================================================================
// FADDEEVA the Faddeeva function, using Stephen Johnson's implementation
//==============================================================================

std::complex<double> faddeeva_c(std::complex<double> z) {
  std::complex<double> wv; // The resultant w(z) value
  double relerr;      // Target relative error in the inner loop of MIT Faddeeva

  // Technically, the value we want is given by the equation:
  // w(z) = I/Pi * Integrate[Exp[-t^2]/(z-t), {t, -Infinity, Infinity}]
  // as shown in Equation 63 from Hwang, R. N. "A rigorous pole
  // representation of multilevel cross sections and its practical
  // applications." Nuclear Science and Engineering 96.3 (1987): 192-209.
  //
  // The MIT Faddeeva function evaluates w(z) = exp(-z^2)erfc(-iz). These
  // two forms of the Faddeeva function are related by a transformation.
  //
  // If we call the integral form w_int, and the function form w_fun:
  // For imag(z) > 0, w_int(z) = w_fun(z)
  // For imag(z) < 0, w_int(z) = -conjg(w_fun(conjg(z)))

  // Note that faddeeva_w will interpret zero as machine epsilon

  relerr = 0.;
  if (z.imag() > 0.) {
      wv = Faddeeva::w(z, relerr);
  } else {
    wv = -std::conj(Faddeeva::w(std::conj(z), relerr));
  }

  return wv;
}

std::complex<double> w_derivative_c(std::complex<double> z, int order){
  std::complex<double> wv; // The resultant w(z) value

  const std::complex<double> twoi_sqrtpi(0.0, 2.0 / std::sqrt(M_PI));

  switch(order) {
    case 0:
      wv = faddeeva_c(z);
      break;
    case 1:
      wv = -2. * z * faddeeva_c(z) + twoi_sqrtpi;
      break;
    default:
      wv = -2. * z * w_derivative_c(z, order - 1) - 2. * (order - 1) *
           w_derivative_c(z, order - 2);
  }

  return wv;
}

//==============================================================================
// BROADEN_WMP_POLYNOMIALS Doppler broadens the windowed multipole curvefit.
// The curvefit is a polynomial of the form a/E + b/sqrt(E) + c + d sqrt(E) ...
//==============================================================================

void broaden_wmp_polynomials_c(double E, double dopp, int n, double factors[]) {
  // Factors is already pre-allocated

  double sqrtE;               // sqrt(energy)
  double beta;                // sqrt(atomic weight ratio * E / kT)
  double half_inv_dopp2;      // 0.5 / dopp**2
  double quarter_inv_dopp4;   // 0.25 / dopp**4
  double erf_beta;            // error function of beta
  double exp_m_beta2;         // exp(-beta**2)
  int i;

  sqrtE = std::sqrt(E);
  beta = sqrtE * dopp;
  half_inv_dopp2 = 0.5 / (dopp * dopp);
  quarter_inv_dopp4 = half_inv_dopp2 * half_inv_dopp2;

  if (beta > 6.0) {
    // Save time, ERF(6) is 1 to machine precision.
    // beta/sqrtpi*exp(-beta**2) is also approximately 1 machine epsilon.
    erf_beta = 1.;
    exp_m_beta2 = 0.;
  } else {
    erf_beta = std::erf(beta);
    exp_m_beta2 = std::exp(-beta * beta);
  }

  // Assume that, for sure, we'll use a second order (1/E, 1/V, const)
  // fit, and no less.

  factors[0] = erf_beta / E;
  factors[1] = 1. / sqrtE;
  factors[2] = factors[0] * (half_inv_dopp2 + E) + exp_m_beta2 /
       (beta * std::sqrt(M_PI));

  // Perform recursive broadening of high order components
  for (i = 0; i < n - 3; i++) {
    if (i != 0) {
      factors[i + 3] = -factors[i - 1] * (i - 1.) * i * quarter_inv_dopp4 +
           factors[i + 1] * (E + (1. + 2. * i) * half_inv_dopp2);
    } else {
      // Although it's mathematically identical, factors[0] will contain
      // nothing, and we don't want to have to worry about memory.
      factors[i + 3] = factors[i + 1]*(E + (1. + 2. * i) * half_inv_dopp2);
    }
  }
}

} // namespace openmc
