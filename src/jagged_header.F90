module jagged_header

  implicit none

!===============================================================================
! JAGGED1D and JAGGED2D is a type which allows for jagged 1-D or 2-D array.
!===============================================================================

  type :: Jagged2D
    real(8), allocatable :: data(:, :)
  end type Jagged2D

  type :: Jagged1D
    real(8), allocatable :: data(:)
  end type Jagged1D

end module jagged_header
