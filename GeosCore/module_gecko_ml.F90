!------------------------------------------------------------------------------
!                  GEOS-Chem Global Chemical Transport Model                  !
!------------------------------------------------------------------------------
!BOP
!     
! !MODULE: module_gecko_ml
!     
! !DESCRIPTION: Module MODULE\_GECKO\_ML contains arrays and routines for
!  calling a neural network model. Original code writeen by David J. Gagne
!  and modified accordingly. (jin, 03/08/2021)
!\\   
!\\   
! !INTERFACE: 
!
module module_gecko_ml
    use module_neural_net
    implicit none
    private ! (jin)
!
! !PUBLIC MEMBER FUNCTIONS:
!
    public :: init_gecko_ml
    public :: run_gecko_neural_net_1d
!
! !REMARKS:
!                                                                             .
!  References:
!  ============================================================================
!  (1 ) 
!
! !REVISION HISTORY:
!  08 Mar 2021 - Jinkyul Choi- Add code 
!                              Add log10 and 10** for precursor in the beginning
!                              and at the end of run_gecko_neural_net_1d
!                              Remove run_gecko_neural_net_3d as it is not needed
!                              Bug-fix in parameters and arrays mostly in dimensions
!                              Bug-fix in run_gecko_neural_net_1d in indexing
!EOP
!------------------------------------------------------------------------------
!BOC
!
! !DEFINED PARAMETERS:
!
!
    type :: neural_net_gecko
        type(Dense), allocatable :: gecko_nn(:)
        real(8), allocatable :: input_scale_values(:,:)
        real(8), allocatable :: output_scale_values(:,:)
        integer :: batch_size
        integer :: in_size
        integer :: out_size
    end type neural_net_gecko

    type(neural_net_gecko), save :: nn_gecko

    contains

    ! init gecko ml will need to be called from chemics_init.F or whatever init
    ! module is used before WRF Chem starts forward integration
    subroutine init_gecko_ml(neural_net_path)

        character(len=255), intent(in) :: neural_net_path
        character(len=255) :: input_scale_file, output_scale_file
        character(len=255) :: neural_net_netcdf

        print *, "INIT_GECKO_ML WAS CALLED"

        nn_gecko%batch_size = 1

        input_scale_file = trim(neural_net_path) // "gecko_input_scale_values.csv"
        output_scale_file = trim(neural_net_path) // "gecko_output_scale_values.csv"
        neural_net_netcdf = trim(neural_net_path) // "gecko_neural_net.nc"

        call load_all_scale_values(input_scale_file, &
                                   nn_gecko%input_scale_values)

        call load_all_scale_values(output_scale_file, &
                                   nn_gecko%output_scale_values)

        call init_neural_net(neural_net_netcdf, &
                             nn_gecko%batch_size, &
                             nn_gecko%gecko_nn)

        nn_gecko%in_size = size(nn_gecko%input_scale_values, 1)
        nn_gecko%out_size = size(nn_gecko%output_scale_values, 1)


        print *, "==============================================================================="
        print *, "INPUT SCALE VALUES"
        print *, nn_gecko%in_size
        print *, nn_gecko%input_scale_values
        print *, "==============================================================================="
        print *, "OUTPUT SCALE VALUES"
        print *, nn_gecko%out_size
        print *, nn_gecko%output_scale_values
        print *, "==============================================================================="

    end subroutine init_gecko_ml

    subroutine run_gecko_neural_net_1d(precursor, gas, aerosol, temperature, &
                                       zenith, pre_exist_aerosols, o3, nox, oh, bins)

        integer, intent(in) :: bins
        real(8), intent(inout) :: precursor
        real(8), intent(inout) :: gas(bins), aerosol(bins)
        real(8), intent(in) :: temperature, zenith, pre_exist_aerosols, o3, nox, oh

        real(8) :: nn_input(nn_gecko%batch_size, nn_gecko%in_size)
        real(8) :: nn_output(nn_gecko%batch_size, nn_gecko%out_size)
        real(8) :: nn_scaled_input(nn_gecko%batch_size, nn_gecko%in_size)
        real(8) :: nn_scaled_output(nn_gecko%batch_size, nn_gecko%out_size)
        integer :: i

        ! fill ANN input array
        ! we may need to modify the code to take the log of the precursor here
        ! done - (jin)
        ! Set NaN values to be -35 (min was -35.6 and 1% percentile is -34.4)
        if (precursor .LE. 0) then
            nn_input(1, 1) = -35.0
        else
            nn_input(1, 1) = log10(precursor)
        endif
        !print *, 'log10', precursor, nn_input(1, 1)
        
        i = 2
        ! I tried to enable support for varying numbers of gas and aerosol bins
            nn_input(1, i: i + bins-1) = gas
            !print *, 'gas', nn_input(1, i: i + size(gas)-1)

        i = i + bins
            nn_input(1, i: i + bins-1) = aerosol
            !print *, 'aerosol', nn_input(1, i: i + size(aerosol)-1)

        i = i + bins
        nn_input(1, i: i + 5) = (/ temperature, zenith, pre_exist_aerosols, &
                                    o3, nox, oh /)
        !print *, 't, sza, ', nn_input(1, i: i + 1)
        !print *, 'a, o3   ', nn_input(1, i + 2: i + 3)
        !print *, 'nox, oh ', nn_input(1, i + 4: i + 5)

        ! scale input values
        call standard_scaler_transform(nn_input, nn_gecko%input_scale_values, &
                                       nn_scaled_input)

        ! call the neural network perdiction function
        call neural_net_predict(nn_scaled_input, nn_gecko%gecko_nn, &
                                nn_scaled_output)

        ! inverse scale the output values
        call standard_scaler_inverse_transform(nn_scaled_output, &
                                               nn_gecko%output_scale_values, &
                                               nn_output)

        i = 1
        ! may need to un-log the precursor
        ! done - (jin)
        precursor = 10.0**nn_output(1, i)
        !print *, 'unlog', nn_output(1, i), precursor

        i = 2
        gas = nn_output(1, i: i + bins-1)

        i = i + bins
        aerosol = nn_output(1, i: i + bins-1)

    end subroutine run_gecko_neural_net_1d
!EOC
end module module_gecko_ml
