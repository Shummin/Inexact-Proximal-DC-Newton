# An implementation of the Inexact Proximal DC Newton-type method

Implementation and numerical experience are described in the paper:

> S. Nakayama, Y. Narushima, and H. Yabe, Inexact proximal DC Newton-type method for nonconvex composite functions. Computational Optimization and Applications, (2023). https://doi.org/10.1007/s10589-023-00525-9

This directory contains the following Matlab source codes:

- Main:
    - exL12.m: main codes of Section 5.1
    - exLSP.m: main codes of Section 5.2
- Methods (See Table 1):
    - mlessBFGS_DC.m: mBFGS(S-Newton)
    - mlessBFGS_DC_VFISTA.m: mBFGS(V-FISTA)
    - mlessSR1_DC_VFISTA.m: mSR1(V-FISTA)
    - LBFGS_DC.m: L-BFGS(TFOCS)
    - pDCAe.m: pDCAe
    - APG.m: mAPG
- Subroutine:
  ProxB_semi.m,
  ProxB_VFISTA.m,
  ProxL12.m,
  ProxLSP.m,
  quad_BFGS_syyy.m,
  quad_SR1.m,
  ell2.m,
  logsum.m,
  soft_thresh.m.

We suggest users install [TFOCS](http://cvxr.com/tfocs/), a MATLAB package that uses accelerated first-order methods to solve conic programs.　


## Authors
  * [Shummin Nakayama](https://orcid.org/0000-0001-7780-8348)

