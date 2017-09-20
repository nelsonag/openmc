module ndpp_header

  use hdf5,           only: HID_T, HSIZE_T, SIZE_T

  use algorithm,      only: binary_search, sort, find
  use constants
  use error,          only: fatal_error
  use hdf5_interface, only: read_attribute, open_group, close_group, &
       open_dataset, read_dataset, close_dataset, get_shape, get_datasets, &
       object_exists, get_name, get_groups
  use jagged_header
  use math,           only: calc_rn
  use random_lcg,     only: prn
  use stl_vector,     only: VectorInt, VectorReal
  use string
  use tally_header,   only: TallyObject

  implicit none

!===============================================================================
! Module constants
!===============================================================================

  ! Flags to denote which type of tally to process in get_tally_data
  integer :: NDPP_CHI_T = 0
  integer :: NDPP_CHI_P = 1
  integer :: NDPP_CHI_D = 2

  integer :: NDPP_N_LOG_BINS = 100 ! Number of log bins to use

!===============================================================================
! ENERGYGRID stores the energy grid used with a logarithmic grid map.
! This is also used by nuclide_header
!===============================================================================

  type EnergyGrid
    integer, allocatable :: grid_index(:) ! log grid mapping indices
    real(8), allocatable :: energy(:)     ! energy values corresponding to xs
  end type EnergyGrid

!===============================================================================
! MOMENTMATRIX and MATRIX contain the jagged array of data for a given Ein
!===============================================================================

  type MomentMatrix
    type(Jagged2D), allocatable :: at_Ein(:)
  end type MomentMatrix

  type Matrix
    type(Jagged1D), allocatable :: at_Ein(:)
  end type Matrix

!===============================================================================
! NDPP contains and processes the pre-processed data used to improve tallying
! convergence
!===============================================================================

  type Ndpp
    real(8), allocatable :: kTs(:) ! temperature in eV (k*T)

    ! Meta-data information
    integer       :: num_groups         ! Number of groups
    logical       :: fissionable        ! is it fissionable?
    logical       :: has_inelastic      ! If inelastic data is included
    character(10) :: scatter_format     ! either "legendre", or "histogram"
    integer       :: order              ! Legendre order or # of histogram bins
    integer       :: num_delayed_groups ! Number of delayed groups

    ! Temperature interp method [this will store the global data on the object
    ! instead of using the global data to avoid circular dependencies]
    integer :: temperature_method

    ! Elastic data
    type(EnergyGrid),   allocatable :: elastic_grid(:) ! Ein grid for each kT
    type(MomentMatrix), allocatable :: elastic(:)  ! Elastic data for each kT
    real(8), allocatable :: elastic_log_spacing(:) ! spacing on logarithmic grid

    ! Inelastic data
    ! This is temperature independent and so looks like elastic without being
    ! an array
    type(EnergyGrid)   :: inelastic_grid ! Ein grid for inelastic data
    type(MomentMatrix) :: inelastic      ! Inelastic data
    type(MomentMatrix) :: nu_inelastic   ! Inelastic Production data
    real(8) :: inelastic_log_spacing ! spacing on logarithmic grid

    ! Chi spectra data (temperature dependent)
    type(EnergyGrid), allocatable :: chi_grid(:) ! Ein grid for each kT
    type(Matrix),     allocatable :: chi(:)      ! Total chi for each kT
    type(Matrix),     allocatable :: chi_p(:)    ! Prompt chi for each kT
    type(Matrix),     allocatable :: chi_d(:, :) ! Delayed chi for each kT & precursor
    real(8), allocatable :: chi_log_spacing(:)   ! spacing on logarithmic grid

  contains
    procedure :: from_hdf5 => ndpp_from_hdf5
    procedure :: tally_scatter => ndpp_tally_scatter
    procedure :: tally_chi => ndpp_tally_chi
    procedure :: tally_delayed_chi => ndpp_tally_delayed_chi
  end type Ndpp

  contains

!===============================================================================
! NDPP_FROM_HDF5 initializes the data from a given HDF5 file
!===============================================================================

  subroutine ndpp_from_hdf5(this, group_id, temperature, method, tolerance, &
                            master, num_groups, scatter_format, order)
    class(Ndpp),      intent(inout)  :: this
    integer(HID_T),   intent(inout)  :: group_id       ! HDF5 group to gather from
    type(VectorReal), intent(in)     :: temperature    ! requested temperatures
    integer,          intent(inout)  :: method         ! kT interpolation method
    real(8),          intent(in)     :: tolerance      ! kT interp tolerance
    integer,          intent(in)     :: num_groups     ! Number of energy groups
    character(len=*), intent(in)     :: scatter_format ! legendre or histogram
    integer,          intent(in)     :: order          ! Legendre order/ histogram bins
    logical,          intent(in)     :: master         ! if this is the master proc

    integer :: i, c
    integer :: i_closest
    integer :: n_temperature
    integer(HID_T) :: kT_group, temp_group, chi_group, energy_dset
    integer(HSIZE_T) :: j
    integer(HSIZE_T) :: dims(1)
    character(MAX_WORD_LEN) :: temp_str
    character(MAX_WORD_LEN), allocatable :: dset_names(:)
    real(8), allocatable :: temps_available(:) ! temperatures available
    real(8) :: temp_desired
    real(8) :: temp_actual
    type(VectorInt) :: temps_to_read
    integer :: order_dim

    this % order = order
    call read_attribute(this % fissionable, group_id, 'fissionable')
    call read_attribute(this % num_delayed_groups, group_id, &
                        'num_delayed_groups')
    this % scatter_format = scatter_format
    if (trim(this % scatter_format) == 'legendre') then
      order_dim = this % order + 1
    else
      order_dim = this % order
    end if
    this % num_groups = num_groups

    kT_group = open_group(group_id, 'kTs')

    ! Determine temperatures available
    call get_datasets(kT_group, dset_names)
    allocate(temps_available(size(dset_names)))
    do i = 1, size(dset_names)
      ! Read temperature value
      call read_dataset(temps_available(i), kT_group, trim(dset_names(i)))
      temps_available(i) = temps_available(i) / K_BOLTZMANN
    end do
    call sort(temps_available)

    ! If only one temperature is available, revert to nearest temperature
    if (size(temps_available) == 1 .and. method == TEMPERATURE_INTERPOLATION) then
      if (master) then
        call warning("NDPP data only available at one temperature. Reverting &
                     &to nearest temperature method.")
      end if
      method = TEMPERATURE_NEAREST
    end if

    ! Determine actual temperatures to read
    select case (method)
    case (TEMPERATURE_NEAREST)
      ! Find nearest temperatures
      do i = 1, temperature % size()
        temp_desired = temperature % data(i)
        i_closest = minloc(abs(temps_available - temp_desired), dim=1)
        temp_actual = temps_available(i_closest)
        if (abs(temp_actual - temp_desired) < tolerance) then
          if (find(temps_to_read, nint(temp_actual)) == -1) then
            call temps_to_read % push_back(nint(temp_actual))
          end if
        else
          call fatal_error("NDPP library does not contain data &
               &at or near " // trim(to_str(nint(temp_desired))) // " K.")
        end if
      end do

    case (TEMPERATURE_INTERPOLATION)
      ! If temperature interpolation or multipole is selected, get a list of
      ! bounding temperatures for each actual temperature present in the model
      TEMP_LOOP: do i = 1, temperature % size()
        temp_desired = temperature % data(i)

        do j = 1, size(temps_available) - 1
          if (temps_available(j) <= temp_desired .and. &
               temp_desired < temps_available(j + 1)) then
            if (find(temps_to_read, nint(temps_available(j))) == -1) then
              call temps_to_read % push_back(nint(temps_available(j)))
            end if
            if (find(temps_to_read, nint(temps_available(j + 1))) == -1) then
              call temps_to_read % push_back(nint(temps_available(j + 1)))
            end if
            cycle TEMP_LOOP
          end if
        end do

        call fatal_error("NDPP library does not contain data at temperatures &
                         &that bound " // &
                         trim(to_str(nint(temp_desired))) // " K.")
      end do TEMP_LOOP
    end select

    ! Store the temperature options
    this % temperature_method = method

    ! Sort temperatures to read
    call sort(temps_to_read)

    n_temperature = temps_to_read % size()
    allocate(this % kTs(n_temperature))
    allocate(this % elastic_grid(n_temperature))
    allocate(this % elastic(n_temperature))
    allocate(this % elastic_log_spacing(n_temperature))
    if (this % fissionable) then
      allocate(this % chi_grid(n_temperature))
      allocate(this % chi(n_temperature))
      allocate(this % chi_p(n_temperature))
      allocate(this % chi_d(this % num_delayed_groups, n_temperature))
      allocate(this % chi_log_spacing(n_temperature))
    end if

    ! Get kT values
    do i = 1, n_temperature
      ! Get temperature as a string
      temp_str = trim(to_str(temps_to_read % data(i))) // "K"

      ! Read exact temperature value
      call read_dataset(this % kTs(i), kT_group, trim(temp_str))
    end do
    call close_group(kT_group)

    ! Read temperature dependent data
    do i = 1, n_temperature
      temp_str = trim(to_str(temps_to_read % data(i))) // "K"
      temp_group = open_group(group_id, temp_str)
      energy_dset = open_dataset(temp_group, 'elastic_energy')
      call get_shape(energy_dset, dims)
      allocate(this % elastic_grid(i) % energy(int(dims(1))))
      call read_dataset(this % elastic_grid(i) % energy, energy_dset)
      call close_dataset(energy_dset)

      ! Initialize the logarithmic grid for the elastic data
      call initialize_logarithmic_grid(this % elastic_grid(i), &
                                       this % elastic_log_spacing(i))

      call sparse_data_from_hdf5(temp_group, 'elastic', int(dims(1)), &
                                 order_dim, this % elastic(i))

      if (this % fissionable) then
        ! energy_dset = open_dataset(temp_group, 'chi_energy')
        ! call get_shape(energy_dset, dims)
        ! allocate(this % chi_grid(i) % energy(int(dims(1))))
        ! call read_dataset(this % chi_grid(i) % energy, energy_dset)
        ! call close_dataset(energy_dset)

        ! ! Initialize the logarithmic grid for the chi data
        ! call initialize_logarithmic_grid(this % chi_grid(i), &
        !                                  this % chi_log_spacing(i))

        ! call sparse_chi_data_from_hdf5(temp_group, 'total_chi', int(dims(1)), &
        !                                this % chi(i))
        ! call sparse_chi_data_from_hdf5(temp_group, 'prompt_chi', int(dims(1)), &
        !                                this % chi_p(i))

        ! ! Open up the delayed chi group
        ! chi_group = open_group(temp_group, "delayed_chi")

        ! ! Loop through all the groups and get the data
        ! do c = 1, this % num_delayed_groups
        !   call sparse_chi_data_from_hdf5(chi_group, to_str(c), &
        !                                  int(dims(1)), this % chi_d(c, i))
        ! end do
        ! call close_group(chi_group)
      end if

      call close_group(temp_group)
    end do

    ! Read temperature-independent inelastic data
    if (object_exists(group_id, 'inelastic_energy')) then
      this % has_inelastic = .true.
      energy_dset = open_dataset(group_id, 'inelastic_energy')
      call get_shape(energy_dset, dims)
      allocate(this % inelastic_grid % energy(int(dims(1))))
      call read_dataset(this % inelastic_grid % energy, energy_dset)
      call close_dataset(energy_dset)

      ! Initialize the logarithmic grid for the elastic data
      call initialize_logarithmic_grid(this % inelastic_grid, &
                                       this % inelastic_log_spacing)

      call sparse_data_from_hdf5(group_id, 'inelastic', int(dims(1)), &
                                 order_dim, this % inelastic)
      call sparse_data_from_hdf5(group_id, 'nu_inelastic', int(dims(1)), &
                                 order_dim, this % nu_inelastic)
    else
      this % has_inelastic = .false.
    end if

  end subroutine ndpp_from_hdf5

!===============================================================================
! NDPP_GET_* produces the requested data for tallying purposes
!===============================================================================

!===============================================================================
! NDPP_TALLY_SCATTER scores the tally data using the NDPP data
!===============================================================================

  subroutine ndpp_tally_scatter(this, Ein, kT, ndpp_outgoing, t, &
                                i_score, i_filter, score_bin, order, wgt, uvw, &
                                elastic_xs)
    class(Ndpp),       intent(in)    :: this
    real(8),           intent(in)    :: Ein
    real(8),           intent(in)    :: kT
    real(8),           intent(inout) :: ndpp_outgoing(:, :)
    type(TallyObject), intent(inout) :: t
    integer,           intent(in)    :: i_score
    integer,           intent(in)    :: i_filter
    integer,           intent(in)    :: score_bin
    integer,           intent(in)    :: order
    real(8),           intent(in)    :: wgt
    real(8),           intent(in)    :: uvw(3)
    real(8),           intent(in)    :: elastic_xs

    integer :: gmin, gmax, f_lo, f_hi, s_lo, s_hi, num_nm, n, f
    real(8), allocatable :: moments(:)

    if (score_bin == SCORE_NDPP_NU_SCATTER_N .or. &
        score_bin == SCORE_NDPP_NU_SCATTER_PN .or. &
        score_bin == SCORE_NDPP_NU_SCATTER_YN) then
      call get_scatter(this, Ein, kT, order, ndpp_outgoing, elastic_xs, &
                       .true., gmin, gmax)
    else
      call get_scatter(this, Ein, kT, order, ndpp_outgoing, elastic_xs, &
                       .false., gmin, gmax)
    end if

    f_lo = i_filter + gmin - 1
    f_hi = i_filter + gmax - 1

    ! Score the data type
    select case(score_bin)

    case (SCORE_NDPP_SCATTER_N, SCORE_NDPP_NU_SCATTER_N)
!$omp critical(case_nu_scatt_n)
      t % results(RESULT_VALUE, i_score, f_lo: f_hi) = &
           t % results(RESULT_VALUE, i_score, f_lo: f_hi) + &
           ndpp_outgoing(1, gmin: gmax) * wgt
!$omp end critical(case_nu_scatt_n)

    case (SCORE_NDPP_SCATTER_PN, SCORE_NDPP_NU_SCATTER_PN)
      s_lo = i_score
      s_hi = i_score + order

!$omp critical(case_nu_scatt_pn)
    t % results(RESULT_VALUE, s_lo: s_hi, f_lo: f_hi) = &
         t % results(RESULT_VALUE, s_lo: s_hi, f_lo: f_hi) + &
         ndpp_outgoing(1: order + 1, gmin: gmax) * wgt
!$omp end critical(case_nu_scatt_pn)

    case (SCORE_NDPP_SCATTER_YN, SCORE_NDPP_NU_SCATTER_YN)
      num_nm = 1
      s_lo = i_score - 1
      ! Find the order for a collection of requested moments
      ! and store the moment contribution of each
      do n = 0, order
        ! determine scoring bin index
        s_lo = s_lo + num_nm
        s_hi = s_lo + num_nm - 1
        ! Update number of total n,m bins for this n (m = [-n: n])
        num_nm = 2 * n + 1

        moments = calc_rn(n, uvw) * wgt
        do f = f_lo, f_hi
          ! multiply score by the angular flux moments and store
!$omp critical (case_nu_scatt_yn)
          t % results(RESULT_VALUE, s_lo: s_hi, f) = &
               t % results(RESULT_VALUE, s_lo: s_hi, f) + &
               ndpp_outgoing(n, f + 1 - i_filter) * moments
!$omp end critical (case_nu_scatt_yn)
        end do
      end do
    end select

  end subroutine ndpp_tally_scatter

  subroutine ndpp_tally_chi(this, Ein, kT, chi_type, data)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein      ! Incoming energy
    real(8),     intent(in) :: kT       ! Requested temperature
    integer,     intent(in) :: chi_type ! total, prompt, or total delayed
    real(8), intent(inout) :: data(:, :) ! combined data

    type(Jagged1D), allocatable :: delay_data(:)

    integer :: t, c, lo, hi, g

    ! First, find the temperature index
    t = find_temperature_index(this % kTs, this % temperature_method, kT)

    ! if (chi_type == NDPP_CHI_T) then
    !   call interp_chi_data(Ein, this % chi_grid(t), &
    !                        this % chi_log_spacing(t), this % chi(t) % at_Ein, data)
    !   ! subroutine interp_chi_data(Ein, grid, log_spacing, data_source, data)
    ! else if (chi_type == NDPP_CHI_P) then
    !   call interp_chi_data(Ein, this % chi_grid(t), &
    !                        this % chi_log_spacing(t), this % chi_p(t) % at_Ein, &
    !                     data)
    ! else if (chi_type == NDPP_CHI_D) then
    !   allocate(delay_data(this % num_delayed_groups))
    !   lo = 1
    !   hi = this % num_groups
    !   do c = 1, this % num_delayed_groups
    !     call interp_chi_data(Ein, this % chi_grid(t), &
    !                          this % chi_log_spacing(t), &
    !                          this % chi_d(c, t) % at_Ein, delay_data(c) % data)
    !     if (lbound(delay_data(c) % data, dim=1) > lo) then
    !       lo = lbound(delay_data(c) % data, dim=1)
    !     end if
    !     if (ubound(delay_data(c) % data, dim=1) > hi) then
    !       hi = ubound(delay_data(c) % data, dim=1)
    !     end if
    !   end do
    !   allocate(data(lo: hi))
    !   data = ZERO
    !   do c = 1, this % num_delayed_groups
    !     do g = lbound(delay_data(c) % data, dim=1), &
    !            ubound(delay_data(c) % data, dim=1)
    !       data(g) = data(g) + delay_data(c) % data(g)
    !     end do
    !   end do
    ! end if

  end subroutine ndpp_tally_chi

  subroutine ndpp_tally_delayed_chi(this, Ein, kT, chi_data)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein  ! Incoming energy
    real(8),     intent(in) :: kT   ! Requested temperature
    type(Jagged1D), intent(inout), allocatable :: chi_data(:)
    integer :: t, c

    ! First, find the temperature index
    t = find_temperature_index(this % kTs, this % temperature_method, kT)

    allocate(chi_data(this % num_delayed_groups))

    do c = 1, this % num_delayed_groups
      call interp_chi_data(Ein, this % chi_grid(t), this % chi_log_spacing(t), &
                           this % chi_d(c, t) % at_Ein, chi_data(c) % data)
    end do

  end subroutine ndpp_tally_delayed_chi

!==============================================================================
! SUPPORT FUNCTIONS
!==============================================================================

  subroutine sparse_data_from_hdf5(group_id, name, data_length, order_dim, &
                                   output_data)
    integer(HID_T), intent(inout) :: group_id    ! Group ID to get data from
    character(len=*), intent(in)  :: name        ! name of dataset get
    integer,          intent(in)  :: data_length ! Length of vector to obtain
    integer,          intent(in)  :: order_dim   ! Legendre or histogram order
    type(MomentMatrix), intent(inout) :: output_data

    character(MAX_WORD_LEN) :: min_max_name
    integer, allocatable :: gmin(:), gmax(:)
    integer :: length, i, l, index, gout
    real(8), allocatable :: temp_arr(:)

    ! Get scattering data
    if (.not. object_exists(group_id, trim(name))) &
         call fatal_error("Data does not exist within provided group!")

    if (name == 'nu_inelastic') then
      min_max_name = 'inelastic'
    else
      min_max_name = name
    end if

    ! Get the outgoing group boundary indices
    if (object_exists(group_id,  trim(min_max_name) // "_g_min")) then
      allocate(gmin(data_length))
      call read_dataset(gmin, group_id, trim(min_max_name) // "_g_min")
    else
      call fatal_error("'g_min' for the requested data must be provided")
    end if

    if (object_exists(group_id, trim(min_max_name) // "_g_max")) then
      allocate(gmax(data_length))
      call read_dataset(gmax, group_id, trim(min_max_name) // "_g_max")
    else
      call fatal_error("'g_max' for the requested data must be provided")
    end if

    ! Now use this information to find the length of a container array
    ! to hold the flattened data
    length = 0
    do i = 1, data_length
      if (gmin(i) > 0) then
        length = length + order_dim * (gmax(i) - gmin(i) + 1)
      end if
    end do

    ! Allocate flattened array
    allocate(temp_arr(length))
    call read_dataset(temp_arr, group_id, name)

    ! Convert temp_arr to a jagged array
    allocate(output_data % at_Ein(data_length))
    index = 1
    do i = 1, data_length
      if (gmin(i) > 0) then
        allocate(output_data % at_Ein(i) % data(order_dim, gmin(i):gmax(i)))
        do gout = gmin(i), gmax(i)
          do l = 1, order_dim
            output_data % at_Ein(i) % data(l, gout) = temp_arr(index)
            index = index + 1
          end do
        end do
      else
        ! Still allocate something so we dont need lots of if-then downstream
        ! to check
        allocate(output_data % at_Ein(i) % data(order_dim, 1))
        output_data % at_Ein(i) % data(:, 1) = ZERO
      end if
    end do
  end subroutine sparse_data_from_hdf5

  subroutine sparse_chi_data_from_hdf5(group_id, name, data_length, output_data)
    integer(HID_T), intent(inout) :: group_id    ! Group ID to get data from
    character(len=*),  intent(in) :: name        ! name of dataset get
    integer,           intent(in) :: data_length ! Length of vector to obtain
    type(Matrix),   intent(inout) :: output_data

    integer, allocatable :: gmin(:), gmax(:)
    integer :: length, i, index, gout
    real(8), allocatable :: temp_arr(:)

    ! Get scattering data
    if (.not. object_exists(group_id, name)) &
         call fatal_error("Data does not exist within provided group!")

    ! First get the outgoing group boundary indices
    if (object_exists(group_id,  trim(name) // "_g_min")) then
      allocate(gmin(data_length))
      call read_dataset(gmin, group_id, trim(name) // "_g_min")
    else
      call fatal_error("'g_min' for the requested data must be provided")
    end if

    if (object_exists(group_id, trim(name) // "_g_max")) then
      allocate(gmax(data_length))
      call read_dataset(gmax, group_id, trim(name) // "_g_max")
    else
      call fatal_error("'g_max' for the requested data must be provided")
    end if

    ! Now use this information to find the length of a container array
    ! to hold the flattened data
    length = 0
    do i = 1, data_length
      length = length + (gmax(i) - gmin(i) + 1)
    end do

    ! Allocate flattened array
    allocate(temp_arr(length))
    call read_dataset(temp_arr, group_id, name)

    ! Convert temp_arr to a jagged array
    allocate(output_data % at_Ein(data_length))
    index = 1
    do i = 1, data_length
      allocate(output_data % at_Ein(i) % data(gmin(i):gmax(i)))
      do gout = gmin(i), gmax(i)
        output_data % at_Ein(i) % data(gout) = temp_arr(index)
        index = index + 1
      end do
    end do
  end subroutine sparse_chi_data_from_hdf5

  subroutine initialize_logarithmic_grid(grid, log_spacing)
    type(EnergyGrid), intent(inout) :: grid
    real(8),          intent(inout) :: log_spacing ! spacing on logarithmic grid
    integer :: i, j                  ! Loop indices
    integer :: M                     ! Number of equally log-spaced bins
    real(8) :: E_max                 ! Maximum energy in MeV
    real(8) :: E_min                 ! Minimum energy in MeV
    real(8), allocatable :: umesh(:) ! Equally log-spaced energy grid

    ! Set minimum/maximum energies
    E_max = grid % energy(size(grid % energy))
    E_min = grid % energy(1)

    ! Determine equal-logarithmic energy spacing
    M = NDPP_N_LOG_BINS
    log_spacing = log(E_max / E_min) / M

    ! Create equally log-spaced energy grid
    allocate(umesh(0:M))
    umesh(:) = [(i * log_spacing, i=0, M)]

    ! Allocate logarithmic mapping for nuclide
    allocate(grid % grid_index(0:M))

    ! Determine corresponding indices in nuclide grid to energies on
    ! equal-logarithmic grid
    i = 1
    do j = 0, M
      do while (log(grid % energy(i + 1) / E_min) <= umesh(j))
        ! Ensure that for isotopes where maxval(energy) << E_max
        ! that there are no out-of-bounds issues.
        if (i + 1 == size(grid % energy)) exit
        i = i + 1
      end do
      grid % grid_index(j) = i
    end do
    deallocate(umesh)

  end subroutine initialize_logarithmic_grid

  function find_temperature_index(kTs, method, kT) result(i_temp)
    real(8), intent(in) :: kTs(:)
    integer, intent(in) :: method
    real(8), intent(in) :: kT

    integer :: i_temp
    real(8) :: f

    select case (method)
    case (TEMPERATURE_NEAREST)
      i_temp = minloc(abs(kTs - kT), dim=1)

    case (TEMPERATURE_INTERPOLATION)
      ! Find temperatures that bound the actual temperature
      do i_temp = 1, size(kTs) - 1
        if (kTs(i_temp) <= kT .and. kT < kTs(i_temp + 1)) exit
      end do

      ! Randomly sample between temperature i and i+1
      f = (kT - kTs(i_temp)) / (kTs(i_temp + 1) - kTs(i_temp))
      if (f > prn()) i_temp = i_temp + 1
    end select
  end function find_temperature_index

  subroutine interp_data(Ein, grid, log_spacing, at_Ein, data, gmin, gmax)
    real(8),          intent(in) :: Ein
    type(EnergyGrid), intent(in) :: grid
    real(8),          intent(in) :: log_spacing
    type(Jagged2D), allocatable, intent(in) :: at_Ein(:)
    real(8),          intent(inout) :: data(:, :)
    integer,          intent(out)   :: gmin
    integer,          intent(out)   :: gmax

    integer :: e, e_low, e_high, g   ! energy indices
    real(8) :: f                     ! Grid interpolant
    integer :: low_gmin, low_gmax    ! Low data point gmin and gmax
    integer :: high_gmin, high_gmax  ! High data point gmin and gmax

    if (Ein <= grid % energy(1)) then
      ! Then dont continue on, the result would be 0 anyways since we are
      ! below the threshold; instead, set gmin and gmax to values which
      ! signify "no useful data" and leave the subroutine
      gmin = size(data, dim=2) + 1
      gmax = 0
      return
    else if (Ein >= grid % energy(size(grid % energy))) then
      e = size(grid % energy) - 1
      f = ONE
    else
      ! Find energy index on logarithmic energy grid
      e = int(log(Ein / grid % energy(1)) / log_spacing)
      ! Determine bounding indices based on which equal log-spaced
      ! interval the energy is in
      e_low  = grid % grid_index(e)
      e_high = grid % grid_index(e + 1) + 1

      ! Perform binary search over reduced range
      e = binary_search(grid % energy(e_low:e_high), &
                        e_high - e_low + 1, Ein) + e_low - 1

      ! calculate interpolation factor
      f = (Ein - grid % energy(e)) / (grid % energy(e + 1) - grid % energy(e))
    end if

    low_gmin = lbound(at_Ein(e) % data, dim=2)
    low_gmax = ubound(at_Ein(e) % data, dim=2)
    high_gmin = lbound(at_Ein(e + 1) % data, dim=2)
    high_gmax = ubound(at_Ein(e + 1) % data, dim=2)

    ! Interpolate to our solution
    do g = low_gmin, low_gmax
      data(:, g) = data(:, g) + (ONE - f) * at_Ein(e) % data(:, g)
    end do
    do g = high_gmin, high_gmax
      data(:, g) = data(:, g) + f * at_Ein(e + 1) % data(:, g)
    end do

    gmin = min(low_gmin, high_gmin)
    gmax = max(low_gmax, high_gmax)

  end subroutine interp_data

  subroutine interp_chi_data(Ein, grid, log_spacing, at_Ein, data)
    real(8),          intent(in) :: Ein
    type(EnergyGrid), intent(in) :: grid
    real(8),          intent(in) :: log_spacing
    type(Jagged1D), allocatable, intent(in) :: at_Ein(:)
    real(8), allocatable, intent(inout) :: data(:)

    integer :: e, e_low, e_high, g   ! energy indices
    real(8) :: f                     ! Grid interpolant
    integer :: low_gmin, low_gmax    ! Low data point gmin and gmax
    integer :: high_gmin, high_gmax  ! High data point gmin and gmax

    if (Ein <= grid % energy(1)) then
      e = 1
      f = ZERO
    else if (Ein >= grid % energy(size(grid % energy))) then
      e = size(grid % energy) - 1
      f = ONE
    else
      ! Find energy index on logarithmic energy grid
      e = int(log(Ein / grid % energy(1)) / log_spacing)
      ! Determine bounding indices based on which equal log-spaced
      ! interval the energy is in
      e_low  = grid % grid_index(e)
      e_high = grid % grid_index(e + 1) + 1

      ! Perform binary search over reduced range
      e = binary_search(grid % energy(e_low:e_high), &
                        e_high - e_low + 1, Ein) + e_low - 1

      ! calculate interpolation factor
      f = (Ein - grid % energy(e)) / (grid % energy(e + 1) - grid % energy(e))
    end if

    low_gmin = lbound(at_Ein(e) % data, dim=1)
    low_gmax = ubound(at_Ein(e) % data, dim=1)
    high_gmin = lbound(at_Ein(e + 1) % data, dim=1)
    high_gmax = ubound(at_Ein(e + 1) % data, dim=1)

    ! Build storage space
    allocate(data(min(low_gmin, high_gmin):max(low_gmax, high_gmax)))

    ! Dont waste time setting all values equal to zero, we only need to set
    ! those where the high points are outside the low points
    data(high_gmin: low_gmin) = ZERO
    data(low_gmax: high_gmax) = ZERO

    ! Interpolate to our solution
    do g = low_gmin, low_gmax
      data(g) = (ONE - f) * at_Ein(e) % data(g)
    end do
    do g = high_gmin, high_gmax
      data(g) = data(g) + f * at_Ein(e + 1) % data(g)
    end do

  end subroutine interp_chi_data

  subroutine get_scatter(this, Ein, kT, order, data, elastic_xs, nu, gmin, gmax)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein   ! Incoming energy
    real(8),     intent(in) :: kT    ! Requested temperature
    integer,     intent(in) :: order ! Order to include
    real(8),     intent(inout) :: data(:, :) ! combined data
    real(8),     intent(in)  :: elastic_xs
    logical,     intent(in)  :: nu
    integer,     intent(out) :: gmin
    integer,     intent(out) :: gmax


    integer :: t, el_gmin, el_gmax, inel_gmin, inel_gmax

    data = ZERO

    ! First, find the temperature index
    t = find_temperature_index(this % kTs, this % temperature_method, kT)

    call interp_data(Ein, this % elastic_grid(t), &
                     this % elastic_log_spacing(t), &
                     this % elastic(t) % at_Ein, data, el_gmin, el_gmax)

    data(:, el_gmin: el_gmax) = elastic_xs * data(:, el_gmin: el_gmax)

    if (this % has_inelastic) then
      if (nu) then
        call interp_data(Ein, this % inelastic_grid, &
                         this % inelastic_log_spacing, &
                         this % nu_inelastic % at_Ein, data, inel_gmin, &
                         inel_gmax)
      else
        call interp_data(Ein, this % inelastic_grid, &
                         this % inelastic_log_spacing, &
                         this % inelastic % at_Ein, data, inel_gmin, inel_gmax)
      end if
      gmin = min(el_gmin, inel_gmin)
      gmax = max(el_gmax, inel_gmax)
    else
      gmin = el_gmin
      gmax = el_gmax
    end if

  end subroutine get_scatter


end module ndpp_header
