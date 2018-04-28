module math

  use, intrinsic :: ISO_C_BINDING

  use constants

  implicit none

  interface

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
      real(C_DOUBLE), value, intent(in) :: phi
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
! the domain of [-1,1], and 0<=n<=10. If x is outside of the range, the return
! value will be outside the expected range.
!===============================================================================

  elemental function calc_pn(n,x) result(pnx)

    integer, intent(in) :: n   ! Legendre order requested
    real(8), intent(in) :: x   ! Independent variable the Legendre is to be
                               ! evaluated at; x must be in the domain [-1,1]
    real(8)             :: pnx ! The Legendre poly of order n evaluated at x

    pnx = calc_pn_cc(n, x)

  end function calc_pn

!===============================================================================
! CALC_RN calculates the n-th order real spherical harmonics for a given angle
! (in terms of (u,v,w)).  All Rn,m values are provided (where -n<=m<=n)
!===============================================================================

  function calc_rn(n,uvw) result(rn)

    integer, intent(in) :: n      ! Order requested
    real(8), intent(in) :: uvw(3) ! Direction of travel, assumed to be on unit sphere
    real(8)             :: rn(2*n + 1)     ! The resultant R_n(uvw)

    call calc_rn_cc(n, uvw, rn)
  end function calc_rn

!===============================================================================
! CALC_ZN calculates the n-th order modified Zernike polynomial moment for a
! given angle (rho, theta) location in the unit disk. The normlization of the
! polynomials is such that the integral of Z_pq*Z_pq over the unit disk is
! exactly pi
!===============================================================================

  subroutine calc_zn(n, rho, phi, zn)
    ! This procedure uses the modified Kintner's method for calculating Zernike
    ! polynomials as outlined in Chong, C. W., Raveendran, P., & Mukundan,
    ! R. (2003). A comparative analysis of algorithms for fast computation of
    ! Zernike moments. Pattern Recognition, 36(3), 731-742.

    integer, intent(in) :: n           ! Maximum order
    real(8), intent(in) :: rho         ! Radial location in the unit disk
    real(8), intent(in) :: phi         ! Theta (radians) location in the unit disk
    real(8), intent(out) :: zn(:)      ! The resulting list of coefficients

    real(8) :: sin_phi, cos_phi        ! Sine and Cosine of phi
    real(8) :: sin_phi_vec(n+1)        ! Contains sin(n*phi)
    real(8) :: cos_phi_vec(n+1)        ! Contains cos(n*phi)
    real(8) :: zn_mat(n+1, n+1)        ! Matrix form of the coefficients which is
                                       ! easier to work with
    real(8) :: k1, k2, k3, k4          ! Variables for R_m_n calculation
    real(8) :: sqrt_norm               ! normalization for radial moments
    integer :: i,p,q                   ! Loop counters

    real(8), parameter :: SQRT_N_1(0:10) = [&
         sqrt(1.0_8), sqrt(2.0_8), sqrt(3.0_8), sqrt(4.0_8), &
         sqrt(5.0_8), sqrt(6.0_8), sqrt(7.0_8), sqrt(8.0_8), &
         sqrt(9.0_8), sqrt(10.0_8), sqrt(11.0_8)]
    real(8), parameter :: SQRT_2N_2(0:10) = SQRT_N_1*sqrt(2.0_8)

    ! n == radial degree
    ! m == azimuthal frequency

    ! ==========================================================================
    ! Determine vector of sin(n*phi) and cos(n*phi). This takes advantage of the
    ! following recurrence relations so that only a single sin/cos have to be
    ! evaluated (http://mathworld.wolfram.com/Multiple-AngleFormulas.html)
    !
    ! sin(nx) = 2 cos(x) sin((n-1)x) - sin((n-2)x)
    ! cos(nx) = 2 cos(x) cos((n-1)x) - cos((n-2)x)

    sin_phi = sin(phi)
    cos_phi = cos(phi)

    sin_phi_vec(1) = 1.0_8
    cos_phi_vec(1) = 1.0_8

    sin_phi_vec(2) = 2.0_8 * cos_phi
    cos_phi_vec(2) = cos_phi

    do i = 3, n+1
      sin_phi_vec(i) = 2.0_8 * cos_phi * sin_phi_vec(i-1) - sin_phi_vec(i-2)
      cos_phi_vec(i) = 2.0_8 * cos_phi * cos_phi_vec(i-1) - cos_phi_vec(i-2)
    end do

    do i = 1, n+1
      sin_phi_vec(i) = sin_phi_vec(i) * sin_phi
    end do

    ! ==========================================================================
    ! Calculate R_pq(rho)

    ! Fill the main diagonal first (Eq. 3.9 in Chong)
    do p = 0, n
      zn_mat(p+1, p+1) = rho**p
    end do

    ! Fill in the second diagonal (Eq. 3.10 in Chong)
    do q = 0, n-2
      zn_mat(q+2+1, q+1) = (q+2) * zn_mat(q+2+1, q+2+1) - (q+1) * zn_mat(q+1, q+1)
    end do

    ! Fill in the rest of the values using the original results (Eq. 3.8 in Chong)
    do p = 4, n
      k2 = 2 * p * (p - 1) * (p - 2)
      do q = p-4, 0, -2
        k1 = (p + q) * (p - q) * (p - 2) / 2
        k3 = -q**2*(p - 1) - p * (p - 1) * (p - 2)
        k4 = -p * (p + q - 2) * (p - q - 2) / 2
        zn_mat(p+1, q+1) = ((k2 * rho**2 + k3) * zn_mat(p-2+1, q+1) + k4 * zn_mat(p-4+1, q+1)) / k1
      end do
    end do

    ! Roll into a single vector for easier computation later
    ! The vector is ordered (0,0), (1,-1), (1,1), (2,-2), (2,0),
    ! (2, 2), ....   in (n,m) indices
    ! Note that the cos and sin vectors are offset by one
    ! sin_phi_vec = [sin(x), sin(2x), sin(3x) ...]
    ! cos_phi_vec = [1.0, cos(x), cos(2x)... ]
    i = 1
    do p = 0, n
      do q = -p, p, 2
        if (q < 0) then
          zn(i) = zn_mat(p+1, abs(q)+1) * sin_phi_vec(abs(q)) * SQRT_2N_2(p)
        else if (q == 0) then
          zn(i) = zn_mat(p+1, q+1) * SQRT_N_1(p)
        else
          zn(i) = zn_mat(p+1, q+1) * cos_phi_vec(abs(q)+1) * SQRT_2N_2(p)
        end if
        i = i + 1
      end do
    end do
  end subroutine calc_zn

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
      call rotate_angle_cc(uvw, mu, -10._8)
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
