module ndpp_material_header

  use algorithm,      only: binary_search, sort, find
  use constants
  use error,          only: fatal_error
  use material_header, only: Material
  use ndpp_header,    only: Ndpp
  use nuclide_header, only: Nuclide, NuclideMicroXS
  use sab_header,     only: SAlphaBeta
  use tally_header,   only: TallyObject

  implicit none

!===============================================================================
! NDPPPOINTER contains an array of pointers to Ndpp objects
!===============================================================================

  type NdppPointer
    type(Ndpp), pointer :: p => null()
  end type NdppPointer

!===============================================================================
! NDPPMATERIAL contains and processes the pre-processed data used to improve
! tallying convergence for a material
!===============================================================================

  type NdppMaterial
    ! The atom densities of the items in the material; this comes from the
    ! Material object
    real(8), allocatable :: atom_density(:)

    ! Store the locations within the global nuclides & sab_tables lists so we
    ! can later access micro_xs
    integer,           allocatable :: i_nuclides(:)   ! index in nuclides
    integer,           allocatable :: i_sab_tables(:) ! index in sab_tables

    ! Pointer to the relevant Ndpp objects, the indices correspond to the
    ! position within the Material object
    type(NdppPointer), allocatable :: nuclides(:)
    type(NdppPointer), allocatable :: sabs(:)

  contains
    procedure :: init => ndppmat_init
    procedure :: tally_scatter => ndppmat_tally_scatter
    ! procedure :: tally_chi => ndppmat_tally_chi
    ! procedure :: tally_delayed_chi => ndppmat_tally_delayed_chi
  end type NdppMaterial

  contains

!===============================================================================
! NDPPMAT_INIT initializes the object given a material object and the nuclides
! array
!===============================================================================

  subroutine ndppmat_init(this, mat, nuclides, sab_tables)
    class(NdppMaterial),                intent(inout) :: this
    type(Material),                     intent(in)    :: mat
    type(Nuclide), allocatable, target, intent(in)    :: nuclides(:)   ! Nuclidic data
    type(SAlphaBeta), allocatable, target, intent(in) :: sab_tables(:) ! S(a,b) tables

    integer i

    this % atom_density = mat % atom_density

    allocate(this % nuclides(mat % n_nuclides))
    do i = 1, mat % n_nuclides
      this % nuclides(i) % p => nuclides(mat % nuclide(i)) % ndpp_data
    end do
    this % i_nuclides = mat % nuclide

    if (mat % n_sab > 0) then
      allocate(this % sabs(mat % n_sab))
      do i = 1, mat % n_sab
        this % sabs(i) % p => sab_tables(i) % ndpp_data
      end do
      this % i_sab_tables = mat % i_sab_tables
    end if

  end subroutine ndppmat_init

!===============================================================================
! NDPPMAT_TALLY_* scores the correct score type to the tally
!===============================================================================

  subroutine ndppmat_tally_scatter(this, Ein, kT, ndpp_outgoing, t, &
                                   i_score, i_filter, score_bin, order, wgt, &
                                   uvw, micro_xs)
    class(NdppMaterial), intent(in)    :: this
    real(8),             intent(in)    :: Ein
    real(8),             intent(in)    :: kT
    real(8),             intent(inout) :: ndpp_outgoing(:, :)
    type(TallyObject),   intent(inout) :: t
    integer,             intent(in)    :: i_score
    integer,             intent(in)    :: i_filter
    integer,             intent(in)    :: score_bin
    integer,             intent(in)    :: order
    real(8),             intent(in)    :: wgt
    real(8),             intent(in)    :: uvw(3)
    type(NuclideMicroXS), intent(in) :: micro_xs(:)

    integer :: mat_nuc, i_nuclide, i_sab
    real(8) :: score

    do mat_nuc = 1, size(this % nuclides)
      ! Get the global nuclides value
      i_nuclide = this % i_nuclides(mat_nuc)
      i_sab = micro_xs(i_nuclide) % index_sab

      ! Set the weighting function
      score = wgt * this % atom_density(mat_nuc) / &
           (micro_xs(i_nuclide) % total - micro_xs(i_nuclide) % absorption)

      ! See if we have already determined if there is S(a,b) scattering or not
      ! for this nuclide
      if (i_sab > 0) then
        call this % sabs(i_sab) % p % tally_scatter( &
             Ein, kT, ndpp_outgoing, t, i_score, i_filter, score_bin, order, &
             score, uvw, micro_xs(i_nuclide) % elastic)
      else
        call this % nuclides(mat_nuc) % p % tally_scatter( &
             Ein, kT, ndpp_outgoing, t, i_score, i_filter, score_bin, order, &
             score, uvw, micro_xs(i_nuclide) % elastic)
      end if
    end do

  end subroutine ndppmat_tally_scatter

end module ndpp_material_header
