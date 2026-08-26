! Dumps sigma(M)/alpha(M) and delta_c(z) tables from Parkinson's reference
! PCH08 FORTRAN implementation's own sigmacdm()/deltcrit() routines, for use
! by ../_fortran_compare.py's FakeCosmoData -- see that file's module
! docstring for why (an external ground truth, used to find a sign error
! in this repo's own _branching_rate_terms that internal self-consistency
! checks could not catch) and full build/regeneration instructions.
program sigma_dump
  use Cosmological_Parameters
  use Power_Spectrum_Parameters
  use Time_Parameters
  use Modified_Merger_Tree
  implicit none
  real :: sigmacdm, deltcrit
  external :: sigmacdm, deltcrit
  real :: m, sig, alpha, z, a
  integer :: i, n
  real :: logmmin, logmmax, dlogm

  ! Same setup block as trees.F90
  G0=0.57
  gamma_1=0.38
  gamma_2=-0.01
  eps1=0.1
  eps2=0.1

  pkinfile='pk_Mill.dat'
  itrans=1
  omega0=0.25
  lambda0=0.75
  h0=0.73
  omegab=0.04
  Gamma=omega0*h0

  nspec=1.0
  dndlnk=0.0
  kref=1.0
  sigma8=0.9

  ! Dump sigma(M), alpha(M) over a wide mass grid
  open(unit=21,file='sigma_table.dat')
  write(21,*) '# logM(Msun/h)  sigma  dlnsigma_dlnM(alpha, i.e. -slope)'
  n=400
  logmmin=6.0
  logmmax=16.0
  dlogm=(logmmax-logmmin)/real(n-1)
  do i=1,n
     m=10.0**(logmmin+dlogm*real(i-1))
     sig=sigmacdm(m,alpha)
     write(21,*) logmmin+dlogm*real(i-1), sig, alpha
  end do
  close(21)

  ! Dump deltac(z) over a grid of z from 0 to 30
  open(unit=22,file='deltac_table.dat')
  write(22,*) '# z  a  deltac'
  n=300
  do i=1,n
     z = 30.0*real(i-1)/real(n-1)
     a = 1.0/(1.0+z)
     write(22,*) z, a, deltcrit(a)
  end do
  close(22)

end program sigma_dump
