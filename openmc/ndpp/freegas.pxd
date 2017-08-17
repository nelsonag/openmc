#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False


# Add a type for our calc_Er_integral functions
ctypedef void (*CALC_ER_INTEGRAL)(double[::1] mus, double Eout, double Ein,
                                  double beta, double alpha, double awr,
                                  double kT, double half_beta_2,
                                  double[::1] adist_x, double[::1] adist_p,
                                  int adist_interp, double[::1] xs_x,
                                  double[::1] xs_y, long[::1] xs_bpts,
                                  long[::1] xs_interp, double[::1] results)

cdef double _integrand(double Er, double Eout, double Estar, double E0,
                       double alpha, double most_of_eta, double most_of_mu_cm,
                       double[::1] adist_x, double[::1] adist_p,
                       int adist_interp)


cdef void calc_Er_integral_cxs(double[::1] mus, double Eout, double Ein,
                               double beta, double alpha, double awr, double kT,
                               double half_beta_2, double[::1] adist_x,
                               double[::1] adist_p, int adist_interp,
                               double[::1] xs_x, double[::1] xs_y,
                               long[::1] xs_bpts, long[::1] xs_interp,
                               double[::1] results)


cdef void calc_Er_integral_doppler(double[::1] mus, double Eout, double Ein,
                                   double beta, double alpha, double awr,
                                   double kT, double half_beta_2,
                                   double[::1] adist_x, double[::1] adist_p,
                                   int adist_interp, double[::1] xs_x,
                                   double[::1] xs_y, long[::1] xs_bpts,
                                   long[::1] xs_interp, double[::1] results)
