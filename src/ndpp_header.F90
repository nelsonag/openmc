module ndpp_header

  use hdf5,           only: HID_T, HSIZE_T, SIZE_T

  use algorithm,      only: binary_search, sort, find
  use constants
  use error,          only: fatal_error
  use hdf5_interface, only: read_attribute, open_group, close_group, &
       open_dataset, read_dataset, close_dataset, get_shape, get_datasets, &
       object_exists, get_name, get_groups
  use jagged_header
  use nuclide_header, only: EnergyGrid
  use random_lcg,     only: prn
  use stl_vector,     only: VectorInt, VectorReal
  use string

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
! GROUPTRANSFER contains the jagged array of data for a given temperature
!===============================================================================

  type GroupTransfer
    type(Jagged2D), allocatable :: data(:)
  end type GroupTransfer


!===============================================================================
! NDPP contains and processes the pre-processed data used to improve tallying
! convergence
!===============================================================================

  type Ndpp
    character(MAX_WORD_LEN) :: name   ! name of nuclide, e.g. U235
    real(8), allocatable    :: kTs(:) ! temperature in eV (k*T)

    ! Meta-data information
    logical       :: fissionable        ! is it fissionable?
    logical       :: has_inelastic      ! If inelastic data is included
    character(10) :: scatter_format     ! either "legendre", or "histogram"
    integer       :: order              ! Legendre order or # of histogram bins
    integer       :: num_delayed_groups ! Number of delayed groups

    ! Temperature interp method [this will store the global data on the object
    ! instead of using the global data to avoid circular dependencies]
    integer :: temperature_method

    ! Elastic data
    type(EnergyGrid),    allocatable :: elastic_grid(:) ! Ein grid for each kT
    type(GroupTransfer), allocatable :: elastic(:) ! Elastic data for each kT
    real(8), allocatable :: elastic_log_spacing(:) ! spacing on logarithmic grid

    ! Inelastic data
    ! This is temperature independent and so looks like elastic without being
    ! an array
    type(EnergyGrid)            :: inelastic_grid ! Ein grid for inelastic data
    type(Jagged2D), allocatable :: inelastic(:)   ! Inlastic data
    real(8) :: inelastic_log_spacing ! spacing on logarithmic grid

    ! Chi spectra data (temperature dependent)
    type(EnergyGrid),    allocatable :: chi_grid(:) ! Ein grid for each kT
    type(Jagged2D),      allocatable :: chi(:)      ! Total chi for each kT
    type(Jagged2D),      allocatable :: chi_p(:)    ! Prompt chi for each kT
    type(GroupTransfer), allocatable :: chi_d(:)    ! Delayed chi for each kT
    real(8), allocatable :: chi_log_spacing(:) ! spacing on logarithmic grid

  contains
    procedure :: from_hdf5 => ndpp_from_hdf5
    procedure :: get_elastic_data => ndpp_get_elastic_data
    procedure :: get_chi_data => ndpp_get_chi_data
    procedure :: get_delayed_chi_data => ndpp_get_delayed_chi_data
  end type Ndpp

  contains

!===============================================================================
! NDPP_FROM_HDF5 initializes the data from a given HDF5 file
!===============================================================================

  subroutine ndpp_from_hdf5(this, group_id, temperature, method, tolerance, &
                            master, group_edges, scatter_format)
    class(Ndpp),      intent(inout)  :: this
    integer(HID_T),   intent(inout)  :: group_id       ! HDF5 group to gather from
    type(VectorReal), intent(in)     :: temperature    ! requested temperatures
    integer,          intent(inout)  :: method         ! kT interpolation method
    real(8),          intent(in)     :: tolerance      ! kT interp tolerance
    real(8), allocatable, intent(in) :: group_edges(:) ! Expected group structure
    character(len=*), intent(in)     :: scatter_format ! legendre or histogram
    logical,          intent(in)     :: master         ! if this is the master proc

    integer :: i
    integer :: i_closest
    integer :: n_temperature
    integer(HID_T) :: energy_dset, kT_group, temp_group
    integer(SIZE_T) :: name_len
    integer(HSIZE_T) :: j
    integer(HSIZE_T) :: dims(1)
    character(MAX_WORD_LEN) :: temp_str
    character(MAX_WORD_LEN), allocatable :: dset_names(:)
    real(8), allocatable :: temps_available(:) ! temperatures available
    real(8) :: temp_desired
    real(8) :: temp_actual
    type(VectorInt) :: temps_to_read
    integer :: order_dim
    integer :: groups
    real(8), allocatable :: lib_group_edges(:)

    ! Get name of nuclide from group
    name_len = len(this % name)
    this % name = get_name(group_id, name_len)

    ! Get rid of leading '/'
    this % name = trim(this % name(2:))

    ! Get meta-data and check as soon as we get it
    energy_dset = open_dataset(group_id, 'group_structure')
    call get_shape(energy_dset, dims)
    allocate(lib_group_edges(dims(1)))
    call read_dataset(lib_group_edges, energy_dset)
    call close_dataset(energy_dset)

    if ((size(lib_group_edges) /= size(group_edges)) .or. &
        (any(lib_group_edges /= group_edges))) then
      call fatal_error("NDPP Library does not contain matching group structure")
    end if

    call read_attribute(this % scatter_format, group_id, 'scatter_format')
    if (trim(this % scatter_format) /= trim(scatter_format)) then
      call fatal_error("NDPP Library does not include the correct &
                       &scatter_format type")
    end if
    call read_attribute(this % order, group_id, 'order')
    call read_attribute(this % fissionable, group_id, 'fissionable')
    call read_attribute(this % num_delayed_groups, group_id, &
                        'num_delayed_groups')
    if (trim(this % scatter_format) == 'legendre') then
      order_dim = this % order + 1
    else
      order_dim = this % order
    end if
    groups = size(group_edges) - 1

    kT_group = open_group(group_id, 'kTs')

    !!!! WHAT ABOUT 0K DATA??? WILL IT BE IN TEMPERATURE VECTOR?
    !!!! IF SO, SKIP IT HERE, BUT DONT ALLOCATE SPACE FOR IT?

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
        call warning("NDPP data for " // trim(this % name) // " are only &
             &available at one temperature. Reverting to nearest temperature &
             &method.")
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
               &for " // trim(this % name) // " at or near " // &
               trim(to_str(nint(temp_desired))) // " K.")
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

        call fatal_error("NDPP library does not contain data &
             &for " // trim(this % name) // " at temperatures that bound " // &
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
      allocate(this % chi_d(n_temperature))
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
        energy_dset = open_dataset(temp_group, 'chi_energy')
        call get_shape(energy_dset, dims)
        allocate(this % chi_grid(i) % energy(int(dims(1))))
        call read_dataset(this % chi_grid(i) % energy, energy_dset)
        call close_dataset(energy_dset)

        ! Initialize the logarithmic grid for the chi data
        call initialize_logarithmic_grid(this % chi_grid(i), &
                                         this % chi_log_spacing(i))

        allocate(this % chi(i) % data(groups, int(dims(1))))
        call read_dataset(this % chi(i) % data, temp_group, 'total_chi')
        allocate(this % chi_p(i) % data(groups, int(dims(1))))
        call read_dataset(this % chi_p(i) % data, temp_group, 'prompt_chi')
        call sparse_data_from_hdf5(temp_group, 'delayed_chi', int(dims(1)), &
                                   this % num_delayed_groups, &
                                   this % chi_d(i) % data)
      end if

      call close_group(temp_group)
    end do

    ! Read temperature-independent inelastic data
    if (object_exists(group_id, 'inelastic_energy')) then
      this % has_inelastic = .True.
      energy_dset = open_dataset(group_id, 'inelastic_energy')
      call get_shape(energy_dset, dims)
      allocate(this % inelastic_grid % energy(int(dims(1))))
      call read_dataset(this % inelastic_grid % energy, energy_dset)
      call close_dataset(energy_dset)

      ! Initialize the logarithmic grid for the elastic data
      call initialize_logarithmic_grid(this % inelastic_grid, &
                                       this % inelastic_log_spacing)

      call sparse_data_from_hdf5(temp_group, 'inelastic', int(dims(1)), &
                                 order_dim, this % inelastic(i))
    end if

    call close_group(group_id)

  end subroutine ndpp_from_hdf5

!===============================================================================
! NDPP_GET_*_DATA produces the requested data for tallying purposes
!===============================================================================

  function ndpp_get_elastic_data(this, Ein, kT, orders) result(data)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein    ! Incoming energy
    real(8),     intent(in) :: kT     ! Requested temperature
    integer,     intent(in) :: orders ! Orders to include

    real(8), allocatable :: data(:, :) ! combined data
    integer :: t

    ! First, find the temperature index
    t = find_temperature_index(this % kTs, this % temperature_method, kT)

    call interp_data(Ein, orders, this % elastic_grid(t), &
                     this % elastic_log_spacing(t), this % elastic(t) % data, &
                     data)

  end function ndpp_get_elastic_data

  function ndpp_get_inelastic_data(this, Ein, orders) result(data)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein    ! Incoming energy
    integer,     intent(in) :: orders ! Orders to include

    real(8), allocatable :: data(:, :) ! combined data

    call interp_data(Ein, orders, this % inelastic_grid, &
                     this % inelastic_log_spacing, this % inelastic, data)

  end function ndpp_get_inelastic_data

  function ndpp_get_chi_data(this, Ein, kT, chi_type) result(data)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein      ! Incoming energy
    real(8),     intent(in) :: kT       ! Requested temperature
    integer,     intent(in) :: chi_type ! total, prompt, or total delayed

    real(8), allocatable :: data(:) ! combined data
    real(8), allocatable :: data_2d(:, :) ! for delayed data only

    integer :: t

    ! First, find the temperature index
    t = find_temperature_index(this % kTs, this % temperature_method, kT)

    if (chi_type == NDPP_CHI_T) then
      call interp_chi_data(Ein, this % chi_grid(t), &
                           this % chi_log_spacing(t), this % chi(t) % data, &
                           data)
    else if (chi_type == NDPP_CHI_P) then
      call interp_chi_data(Ein, this % chi_grid(t), &
                           this % chi_log_spacing(t), this % chi_p(t) % data, &
                        data)
    else if (chi_type == NDPP_CHI_D) then
      call interp_data(Ein, this % num_delayed_groups, this % chi_grid(t), &
                       this % chi_log_spacing(t), this % chi_d(t) % data, &
                       data_2d)
      allocate(data(size(data_2d, dim=2)))
      data = sum(data_2d, dim=1)
    end if

  end function ndpp_get_chi_data

  function ndpp_get_delayed_chi_data(this, Ein, kT) result(data)
    class(Ndpp), intent(in) :: this
    real(8),     intent(in) :: Ein      ! Incoming energy
    real(8),     intent(in) :: kT       ! Requested temperature

    real(8), allocatable :: data(:, :)
    integer :: t

    ! First, find the temperature index
    t = find_temperature_index(this % kTs, this % temperature_method, kT)

    call interp_data(Ein, this % num_delayed_groups, this % chi_grid(t), &
                     this % chi_log_spacing(t), this % chi_d(t) % data, data)

  end function ndpp_get_delayed_chi_data

!==============================================================================
! SUPPORT FUNCTIONS
!==============================================================================

  subroutine sparse_data_from_hdf5(group_id, name, data_length, order_dim, &
                                   output_data)
    integer(HID_T), intent(inout) :: group_id    ! Group ID to get data from
    character(10),  intent(in)    :: name        ! name of dataset get
    integer,        intent(in)    :: data_length ! Length of vector to obtain
    integer,        intent(in)    :: order_dim   ! Legendre or histogram order
    type(Jagged2D), allocatable, intent(inout) :: output_data(:)

    integer, allocatable :: gmin(:), gmax(:)
    integer :: length, i, l, index, gout
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
      length = length + order_dim * (gmax(i) - gmin(i) + 1)
    end do

    ! Allocate flattened array
    allocate(temp_arr(length))
    call read_dataset(temp_arr, group_id, name)

    ! Convert temp_arr to a jagged array
    allocate(output_data(data_length))
    index = 1
    do i = 1, data_length
      allocate(output_data(i) % data(order_dim, gmin(i):gmax(i)))
      do gout = gmin(i), gmax(i)
        do l = 1, order_dim
          output_data(i) % data(l, gout) = temp_arr(index)
          index = index + 1
        end do
      end do
    end do
  end subroutine sparse_data_from_hdf5

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

  subroutine interp_data(Ein, orders, grid, log_spacing, data_source, data)
    real(8),          intent(in) :: Ein
    integer,          intent(in) :: orders
    type(EnergyGrid), intent(in) :: grid
    real(8),          intent(in) :: log_spacing
    type(Jagged2D), allocatable, intent(in) :: data_source(:)
    real(8), allocatable, intent(inout) :: data(:, :)

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
      f = (Ein - grid % energy(e)) / &
           (grid % energy(e + 1) - grid % energy(e))
    end if

    low_gmin = lbound(data_source(e) % data, dim=2)
    low_gmax = ubound(data_source(e) % data, dim=2)
    high_gmin = lbound(data_source(e + 1) % data, dim=2)
    high_gmax = ubound(data_source(e + 1) % data, dim=2)

    ! Build storage space
    allocate(data(orders, min(low_gmin, high_gmin):max(low_gmax, high_gmax)))

    ! Dont waste time setting all values equal to zero, we only need to set
    ! those where the high points are outside the low points
    data(:, high_gmin: low_gmin) = ZERO
    data(:, low_gmax: high_gmax) = ZERO

    ! Interpolate to our solution
    do g = low_gmin, low_gmax
      data(:, g) = (ONE - f) * data_source(e) % data(:, g)
    end do
    do g = high_gmin, high_gmax
      data(:, g) = data(:, g) + f * data_source(e + 1) % data(:, g)
    end do

  end subroutine interp_data

  subroutine interp_chi_data(Ein, grid, log_spacing, data_source, data)
    real(8),          intent(in) :: Ein
    type(EnergyGrid), intent(in) :: grid
    real(8),          intent(in) :: log_spacing
    real(8), allocatable, intent(in) :: data_source(:, :)
    real(8), allocatable, intent(inout) :: data(:)

    integer :: e, e_low, e_high ! energy indices
    real(8) :: f                ! Grid interpolant

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
      f = (Ein - grid % energy(e)) / &
           (grid % energy(e + 1) - grid % energy(e))
    end if

    ! Build storage space
    allocate(data(size(data_source, dim=2)))

    ! Interpolate to our solution
    data(:) = (ONE - f) * data_source(e, :) + f * data_source(e + 1, :)

  end subroutine interp_chi_data

end module ndpp_header
