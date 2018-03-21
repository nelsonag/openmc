module math

  use, intrinsic :: ISO_C_BINDING

  use constants
  use random_lcg, only: prn

  implicit none

!===============================================================================
! FADDEEVA_W evaluates the scaled complementary error function.  This
! interfaces with the MIT C library
!===============================================================================

  interface
    pure function normal_percentile_cc(p) bind(C, name='normal_percentile_c') &
         result(z)
      use ISO_C_BINDING
      implicit none
      real(C_DOUBLE), value, intent(in) :: p
      real(C_DOUBLE) :: z
    end function normal_percentile_cc

    pure function t_percentile_cc(p, df) bind(C, name='t_percentile_c') &
         result(t)
      use ISO_C_BINDING
      implicit none
      real(C_DOUBLE), value, intent(in) :: p
      integer(C_INT), value, intent(in) :: df
      real(C_DOUBLE) :: t
    end function t_percentile_cc

    pure function calc_pn_cc(n, x) bind(C, name='calc_pn_c') result(pnx)
      use ISO_C_BINDING
      implicit none
      integer(C_INT), value, intent(in) :: n
      real(C_DOUBLE), value, intent(in) :: x
      real(C_DOUBLE) :: pnx
    end function calc_pn_cc

    subroutine calc_rn_cc(n, uvw, rn) bind(C, name='calc_rn_c')
      use ISO_C_BINDING
      implicit none
      integer(C_INT), value, intent(in) :: n
      real(C_DOUBLE), intent(in) :: uvw(3)
      real(C_DOUBLE), intent(in) :: rn(2 * n + 1)
    end subroutine calc_rn_cc

    pure function evaluate_legendre_cc(n, data, x) &
         bind(C, name='evaluate_legendre_c') result(val)
      use ISO_C_BINDING
      implicit none
      integer(C_INT), value, intent(in) :: n
      real(C_DOUBLE), intent(in) :: data(n)
      real(C_DOUBLE), value, intent(in) :: x
      real(C_DOUBLE) :: val
    end function evaluate_legendre_cc

    subroutine rotate_angle_cc(uvw, mu, phi) bind(C, name='rotate_angle_c')
      use ISO_C_BINDING
      implicit none
      real(C_DOUBLE), intent(inout) :: uvw(3)
      real(C_DOUBLE), value, intent(in)    :: mu
      real(C_DOUBLE), optional, intent(in) :: phi
    end subroutine rotate_angle_cc

    function maxwell_spectrum_cc(T) bind(C, name='maxwell_spectrum_c') &
         result(E_out)
      use ISO_C_BINDING
      implicit none
      real(C_DOUBLE), value, intent(in) :: T
      real(C_DOUBLE) :: E_out
    end function maxwell_spectrum_cc

    function watt_spectrum_cc(a, b) bind(C, name='watt_spectrum_c') &
         result(E_out)
      use ISO_C_BINDING
      implicit none
      real(C_DOUBLE), value, intent(in) :: a
      real(C_DOUBLE), value, intent(in) :: b
      real(C_DOUBLE) :: E_out
    end function watt_spectrum_cc

    function faddeeva_cc(z) bind(C, name='faddeeva_c') result(wv)
      use ISO_C_BINDING
      implicit none
      complex(C_DOUBLE_COMPLEX), value, intent(in) :: z
      complex(C_DOUBLE_COMPLEX) :: wv
    end function faddeeva_cc


    function w_derivative_cc(z, order) bind(C, name='w_derivative_c') &
           result(wv)
      use ISO_C_BINDING
      implicit none
      complex(C_DOUBLE_COMPLEX), value, intent(in) :: z
      integer(C_INT), value,            intent(in) :: order
      complex(C_DOUBLE_COMPLEX) :: wv
    end function w_derivative_cc

    subroutine broaden_wmp_polynomials_cc(E, dopp, n, factors) &
           bind(C, name='broaden_wmp_polynomials_c')
      use ISO_C_BINDING
      implicit none
      real(C_DOUBLE), value, intent(in) :: E
      real(C_DOUBLE), value, intent(in) :: dopp
      integer(C_INT), value, intent(in) :: n
      real(C_DOUBLE), intent(inout) :: factors(n)
    end subroutine broaden_wmp_polynomials_cc

  end interface

contains

!===============================================================================
! NORMAL_PERCENTILE calculates the percentile of the standard normal
! distribution with a specified probability level
!===============================================================================

  elemental function normal_percentile(p) result(z)

    real(8), intent(in) :: p ! probability level
    real(8)             :: z ! corresponding z-value

    z = normal_percentile_cc(p)

  end function normal_percentile

!===============================================================================
! T_PERCENTILE calculates the percentile of the Student's t distribution with a
! specified probability level and number of degrees of freedom
!===============================================================================

  elemental function t_percentile(p, df) result(t)

    real(8), intent(in) :: p  ! probability level
    integer, intent(in) :: df ! degrees of freedom
    real(8)             :: t  ! corresponding t-value

    t = t_percentile_cc(p, df)

  end function t_percentile

!===============================================================================
! CALC_PN calculates the n-th order Legendre polynomial at the value of x.
! Since this function is called repeatedly during the neutron transport process,
! neither n or x is checked to see if they are in the applicable range.
! This is left to the client developer to use where applicable. x is to be in
! the domain of [-1,1], and 0<=n<=5. If x is outside of the range, the return
! value will be outside the expected range; if n is outside the stated range,
! the return value will be 1.0.
!===============================================================================

  elemental function calc_pn(n,x) result(pnx)

    integer, intent(in) :: n   ! Legendre order requested
    real(8), intent(in) :: x   ! Independent variable the Legendre is to be
                               ! evaluated at; x must be in the domain [-1,1]
    real(8)             :: pnx ! The Legendre poly of order n evaluated at x

    pnx = calc_pn_cc(n, x)

  end function calc_pn

!===============================================================================
! CALC_RN calculates the n-th order spherical harmonics for a given angle
! (in terms of (u,v,w)).  All Rn,m values are provided (where -n<=m<=n)
!===============================================================================

  function calc_rn(n,uvw) result(rn)

    integer, intent(in) :: n      ! Order requested
    real(8), intent(in) :: uvw(3) ! Direction of travel, assumed to be on unit sphere
    real(8)             :: rn(2*n + 1)     ! The resultant R_n(uvw)

    call calc_rn_cc(n, uvw, rn)
  end function calc_rn

!===============================================================================
! EVALUATE_LEGENDRE Find the value of f(x) given a set of Legendre coefficients
! and the value of x
!===============================================================================

  pure function evaluate_legendre(data, x) result(val)
    real(8), intent(in) :: data(:)
    real(8), intent(in) :: x
    real(8)             :: val

    val = evaluate_legendre_cc(size(data), data, x)

  end function evaluate_legendre

!===============================================================================
! ROTATE_ANGLE rotates direction cosines through a polar angle whose cosine is
! mu and through an azimuthal angle sampled uniformly. Note that this is done
! with direct sampling rather than rejection as is done in MCNP and SERPENT.
!===============================================================================

  function rotate_angle(uvw0, mu, phi) result(uvw)
    real(8), intent(in) :: uvw0(3) ! directional cosine
    real(8), intent(in) :: mu      ! cosine of angle in lab or CM
    real(8), optional   :: phi     ! azimuthal angle
    real(8)             :: uvw(3)  ! rotated directional cosine

    uvw = uvw0
    if (present(phi)) then
      call rotate_angle_cc(uvw, mu, phi)
    else
      call rotate_angle_cc(uvw, mu)
    end if

  end function rotate_angle

!===============================================================================
! MAXWELL_SPECTRUM samples an energy from the Maxwell fission distribution based
! on a direct sampling scheme. The probability distribution function for a
! Maxwellian is given as p(x) = 2/(T*sqrt(pi))*sqrt(x/T)*exp(-x/T). This PDF can
! be sampled using rule C64 in the Monte Carlo Sampler LA-9721-MS.
!===============================================================================

  function maxwell_spectrum(T) result(E_out)

    real(8), intent(in)  :: T     ! tabulated function of incoming E
    real(8)              :: E_out ! sampled energy

    E_out = maxwell_spectrum_cc(T)

  end function maxwell_spectrum

!===============================================================================
! WATT_SPECTRUM samples the outgoing energy from a Watt energy-dependent fission
! spectrum. Although fitted parameters exist for many nuclides, generally the
! continuous tabular distributions (LAW 4) should be used in lieu of the Watt
! spectrum. This direct sampling scheme is an unpublished scheme based on the
! original Watt spectrum derivation (See F. Brown's MC lectures).
!===============================================================================

  function watt_spectrum(a, b) result(E_out)

    real(8), intent(in) :: a     ! Watt parameter a
    real(8), intent(in) :: b     ! Watt parameter b
    real(8)             :: E_out ! energy of emitted neutron

    E_out = watt_spectrum_cc(a, b)

  end function watt_spectrum

!===============================================================================
! FADDEEVA the Faddeeva function, using Stephen Johnson's implementation
!===============================================================================

  function faddeeva(z) result(wv)
    complex(C_DOUBLE_COMPLEX), intent(in) :: z  ! The point to evaluate Z at
    complex(C_DOUBLE_COMPLEX)             :: wv ! The resulting w(z) value

    wv = faddeeva_cc(z)

  end function faddeeva

  recursive function w_derivative(z, order) result(wv)
    complex(C_DOUBLE_COMPLEX), intent(in) :: z ! The point to evaluate Z at
    integer,                   intent(in) :: order
    complex(C_DOUBLE_COMPLEX)             :: wv     ! The resulting w(z) value

    wv = w_derivative_cc(z, order)

  end function w_derivative

!===============================================================================
! BROADEN_WMP_POLYNOMIALS Doppler broadens the windowed multipole curvefit.  The
! curvefit is a polynomial of the form
! a/E + b/sqrt(E) + c + d sqrt(E) ...
!===============================================================================

  subroutine broaden_wmp_polynomials(E, dopp, n, factors)
    real(8), intent(in) :: E          ! Energy to evaluate at
    real(8), intent(in) :: dopp       ! sqrt(atomic weight ratio / kT),
                                      !  kT given in eV.
    integer, intent(in) :: n          ! number of components to polynomial
    real(8), intent(out):: factors(n) ! output leading coefficient

    call broaden_wmp_polynomials_cc(E, dopp, n, factors)

  end subroutine broaden_wmp_polynomials

end module math
