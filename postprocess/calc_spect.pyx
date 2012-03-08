#!/usr/bin/python
#encoding: utf-8
# cconv.pyx
# cython: profile=True
import cython
import numpy as np
cimport numpy as np
ctypedef np.int32_t itype_t
ctypedef np.float64_t dtype_t
cdef extern from "math.h":
    double floor(double)
    double sqrt(double)

@cython.boundscheck(False)
@cython.wraparound(False)

def calc_k(np.ndarray[dtype_t, ndim=3] k_x,
               np.ndarray[dtype_t, ndim=3] k_y,
               np.ndarray[dtype_t, ndim=3] k_z,
               np.ndarray[dtype_t, ndim=3] k_abs,
               dtype_t dk_val):
    "Calculate the continuous momenta:"

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef dtype_t pi, dk, k_len
    cdef int dimx, dimy, dimz, dimz2, N

    if k_x is None or k_y is None or k_z is None:
        raise ValueError("Input arrays cannot be None")

    dimx = k_x.shape[2]
    dimy = k_x.shape[1]
    dimz2 = k_x.shape[0]

    N = dimx

    pi = np.pi
    dk = dk_val

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k_len = sqrt(float(px*px + py*py + pz*pz))

                if k_len > 0.:
                    k_x[z,y,x] = px/k_len
                    k_y[z,y,x] = py/k_len
                    k_z[z,y,x] = pz/k_len
                    k_abs[z,y,x] = k_len*dk
                else:
                    k_x[z,y,x] = 1./sqrt(3.)#k_x
                    k_y[z,y,x] = 1./sqrt(3.)#k_y
                    k_z[z,y,x] = 1./sqrt(3.)#k_z
                    k_abs[z,y,x] = 0.0


def calc_k_eff_le(np.ndarray[dtype_t, ndim=3] k_eff_abs,
                  dtype_t dx_val):
    "Calculate the effective momenta related to the Defrost stencil:"

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef dtype_t pi, dx, k_abs
    cdef dtype_t ii, jj, kk
    cdef int dimx, dimy, dimz, dimz2, N

    if k_eff_abs is None:
        raise ValueError("Input arrays cannot be None")

    dimx = k_eff_abs.shape[2]
    dimy = k_eff_abs.shape[1]
    dimz2 = k_eff_abs.shape[0]

    N = dimx

    pi = np.pi
    dx = dx_val

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        ii = np.sin(pi*px/N)

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            jj = np.sin(pi*py/N)
            for z from 0 <= z < dimz2:

                kk = np.sin(pi*z/N)

                k2 = 4*(ii*ii + jj*jj + kk*kk)
                k_abs = sqrt(k2)

                if k_abs > 0.:
                    k_eff_abs[z,y,x] = k_abs/dx
                else:
                    k_eff_abs[z,y,x] = 0.0


def calc_k_eff_df(np.ndarray[dtype_t, ndim=3] k_eff_abs,
                  dtype_t dx_val):
    "Calculate the effective momenta related to the Defrost stencil:"

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef dtype_t pi, dx, k_abs
    cdef dtype_t ii, jj, kk
    cdef int dimx, dimy, dimz, dimz2, N

    if k_eff_abs is None:
        raise ValueError("Input arrays cannot be None")

    dimx = k_eff_abs.shape[2]
    dimy = k_eff_abs.shape[1]
    dimz2 = k_eff_abs.shape[0]

    N = dimx

    pi = np.pi
    dx = dx_val

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        ii = np.cos(2*pi*px/N)

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            jj = np.cos(2*pi*py/N)
            for z from 0 <= z < dimz2:

                kk = np.cos(2*pi*z/N)

                k2 = (4.2666666666666666 - 0.93333333333333335*(ii+jj+kk)
                      - 0.4*(ii*jj+ii*kk+jj*kk)
                      - 0.26666666666666666*(ii*jj*kk))
                k_abs = sqrt(k2)

                if k_abs > 0.:
                    k_eff_abs[z,y,x] = k_abs/dx
                else:
                    k_eff_abs[z,y,x] = 0.0

def calc_k_eff_2(np.ndarray[dtype_t, ndim=3] k_eff_x,
               np.ndarray[dtype_t, ndim=3] k_eff_y,
               np.ndarray[dtype_t, ndim=3] k_eff_z,
               np.ndarray[dtype_t, ndim=3] k_eff_abs,
               dtype_t dx_val):
    "Calculate the effective momenta related to the second order stencil:"

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef dtype_t pi, dx, k_abs
    cdef int dimx, dimy, dimz, dimz2, N

    if k_eff_x is None or k_eff_y is None or k_eff_z is None:
        raise ValueError("Input arrays cannot be None")

    dimx = k_eff_x.shape[2]
    dimy = k_eff_x.shape[1]
    dimz2 = k_eff_x.shape[0]

    N = dimx

    pi = np.pi
    dx = dx_val

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx
        k_x = np.sin(2*pi*px/N)

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy
            k_y = np.sin(2*pi*py/N)

            for z from 0 <= z < dimz2:
                k_z = np.sin(2*pi*z/N)

                k_abs = sqrt(k_x*k_x + k_y*k_y + k_z*k_z)

                if k_abs > 0.:
                    k_eff_x[z,y,x] = k_x/k_abs
                    k_eff_y[z,y,x] = k_y/k_abs
                    k_eff_z[z,y,x] = k_z/k_abs
                    k_eff_abs[z,y,x] = k_abs/dx
                else:
                    k_eff_x[z,y,x] = 1./sqrt(3.)#k_x
                    k_eff_y[z,y,x] = 1./sqrt(3.)#k_y
                    k_eff_z[z,y,x] = 1./sqrt(3.)#k_z
                    k_eff_abs[z,y,x] = 0.0
                    

def calc_k_eff_4(np.ndarray[dtype_t, ndim=3] k_eff_x,
               np.ndarray[dtype_t, ndim=3] k_eff_y,
               np.ndarray[dtype_t, ndim=3] k_eff_z,
               np.ndarray[dtype_t, ndim=3] k_eff_abs,
               dtype_t dx_val):
    "Calculate the effective momenta related to fourth order stencil:"

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef dtype_t pi, dx, k_abs
    cdef int dimx, dimy, dimz, dimz2, N

    if k_eff_x is None or k_eff_y is None or k_eff_z is None:
        raise ValueError("Input arrays cannot be None")

    dimx = k_eff_x.shape[2]
    dimy = k_eff_x.shape[1]
    dimz2 = k_eff_x.shape[0]

    N = dimx

    pi = np.pi
    dx = dx_val

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx
        k_x = np.sin(2*pi*px/N)*(4.0-np.cos(2*pi*px/N))

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy
            k_y = np.sin(2*pi*py/N)*(4.0-np.cos(2*pi*py/N))

            for z from 0 <= z < dimz2:

                k_z = np.sin(2*pi*z/N)*(4.0-np.cos(2*pi*z/N))

                k_abs = sqrt(k_x*k_x + k_y*k_y + k_z*k_z)

                if k_abs > 0.:
                    k_eff_x[z,y,x] = k_x/k_abs
                    k_eff_y[z,y,x] = k_y/k_abs
                    k_eff_z[z,y,x] = k_z/k_abs
                    k_eff_abs[z,y,x] = k_abs/dx
                else:
                    k_eff_x[z,y,x] = 1./sqrt(3.)#k_x
                    k_eff_y[z,y,x] = 1./sqrt(3.)#k_y
                    k_eff_z[z,y,x] = 1./sqrt(3.)#k_z
                    k_eff_abs[z,y,x] = 0.0


def calc_le(np.ndarray[dtype_t, ndim=3] F_k_Re,
         np.ndarray[dtype_t, ndim=3] F_k_Im,
         np.ndarray[dtype_t, ndim=3] Pi_k_Re,
         np.ndarray[dtype_t, ndim=3] Pi_k_Im,
         np.ndarray[itype_t, ndim=1] counts,
         np.ndarray[dtype_t, ndim=1] S_k,
         np.ndarray[dtype_t, ndim=1] k2_S_k,
         np.ndarray[dtype_t, ndim=1] n_k,
         np.ndarray[dtype_t, ndim=1] rho_k,
         np.ndarray[dtype_t, ndim=1] k2_rho_k,
         dtype_t dk,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):
    "Calculate LATTICEEASY spectra:"

    """This code is adapted from LatticeEasy.
       Viewer discretion is advised!"""

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef int cnt
    cdef dtype_t k, k_val, k2_val, wk, wk2, Fk_sq, nk
    cdef dtype_t x1, x2, y1, y2
    cdef int dimx, dimy, dimz2

    if F_k_Re is None or counts is None or S_k is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = counts.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        S_k[i] = 0.0
        n_k[i] = 0.0
        rho_k[i] = 0.0

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k = sqrt(float(px*px + py*py + pz*pz))
                k_val = k*dk
                k2_val = k_val*k_val

                i = int(k)

                cnt = 1 if (z == 0 or z == dimz2-1) else 2

                Fk_sq = F_k_Re[z,y,x]*F_k_Re[z,y,x]+F_k_Im[z,y,x]*F_k_Im[z,y,x]

                S_k[i] += cnt*(Fk_sq)
                k2_S_k[i] += cnt*k2_val*(Fk_sq)
                counts[i] += cnt

                w_k2 = k*k*dk*dk + wk2_term

                w_k = sqrt(w_k2)

                x1 = a_term*Pi_k_Re[z,y,x]
                x2 = a_term*Pi_k_Im[z,y,x]

                y1 = p_term*F_k_Re[z,y,x]
                y2 = p_term*F_k_Im[z,y,x]

                if w_k > 0:
                    nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
                else:
                    nk = 0

                #nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
                #nk = w_k*(Fk_sq) + (Pi_k_Re[z,y,x])/w_k

                n_k[i] += cnt*nk
                rho_k[i] += cnt*w_k*nk
                k2_rho_k[i] += cnt*k_val*k_val*w_k*nk

def calc_df(np.ndarray[dtype_t, ndim=3] F_k_Re,
         np.ndarray[dtype_t, ndim=3] F_k_Im,
         np.ndarray[dtype_t, ndim=3] Pi_k_Re,
         np.ndarray[dtype_t, ndim=3] Pi_k_Im,
         np.ndarray[dtype_t, ndim=1] counts,
         np.ndarray[dtype_t, ndim=1] S_k,
         np.ndarray[dtype_t, ndim=1] k2_S_k,
         np.ndarray[dtype_t, ndim=1] n_k,
         np.ndarray[dtype_t, ndim=1] rho_k,
         np.ndarray[dtype_t, ndim=1] k2_rho_k,
         dtype_t dk,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):
    "Calculate DEFROST spectra:"

    """This code is partially adapted from Defrost.
       Viewer discretion is advised!"""

    cdef int i, x, y, z, ns
    cdef int px, py, pz
    cdef int cnt, l_int
    cdef int dimx, dimy, dimz2

    cdef dtype_t k, k_val, k2_val, l, c0, c1
    cdef dtype_t x1, x2, y1, y2
    cdef dtype_t wk, wk2, Fk_sq, nk


    if F_k_Re is None or F_k_Im is None or S_k is None or counts is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = S_k.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        S_k[i] = 0.0
        n_k[i] = 0.0
        rho_k[i] = 0.0


    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k = sqrt(float(px*px + py*py + pz*pz))
                k_val = k*dk
                k2_val = k_val*k_val

                l = floor(k)

                i = int(l)

                c0 = (1.0 - (l-k)*(l-k))*(1.0 - (l-k)*(l-k))
                c1 = (1.0 - (l+1.0-k)*(l+1.0-k))*(1.0 - (l+1.0-k)*(l+1.0-k))

                cnt = 1 if (z == 0 or z == dimz2-1) else 2

                Fk_sq = F_k_Re[z,y,x]*F_k_Re[z,y,x]+F_k_Im[z,y,x]*F_k_Im[z,y,x]

                w_k2 = k*k*dk*dk + wk2_term

                w_k = sqrt(w_k2)

                x1 = a_term*Pi_k_Re[z,y,x]
                x2 = a_term*Pi_k_Im[z,y,x]

                y1 = p_term*F_k_Re[z,y,x]
                y2 = p_term*F_k_Im[z,y,x]

                if w_k > 0:
                    nk = (coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 +
                                                x2*x2 + 2.0*x2*y2 + y2*y2)/w_k))
                else:
                    nk = 0


                S_k[i] += c0*cnt*Fk_sq
                k2_S_k[i] += c0*cnt*k2_val*(Fk_sq)
                counts[i] += c0*cnt
                n_k[i] += c0*cnt*nk
                rho_k[i] += c0*cnt*w_k*nk
                k2_rho_k[i] += c0*cnt*k_val*k_val*w_k*nk

                if i < ns-1:
                    S_k[i+1] += c1*cnt*Fk_sq
                    k2_S_k[i] += c1*cnt*k2_val*(Fk_sq)
                    counts[i+1] += c1*cnt
                    n_k[i] += c1*cnt*nk
                    rho_k[i] += c1*cnt*w_k*nk
                    k2_rho_k[i] += c1*cnt*k_val*k_val*w_k*nk

def calc_spect_k2_eff_le(np.ndarray[dtype_t, ndim=3] F_k_Re,
         np.ndarray[dtype_t, ndim=3] F_k_Im,
         np.ndarray[dtype_t, ndim=3] Pi_k_Re,
         np.ndarray[dtype_t, ndim=3] Pi_k_Im,
         np.ndarray[itype_t, ndim=1] counts,
         np.ndarray[dtype_t, ndim=1] S_k,
         np.ndarray[dtype_t, ndim=1] k2_S_k,
         np.ndarray[dtype_t, ndim=1] n_k,
         np.ndarray[dtype_t, ndim=1] rho_k,
         np.ndarray[dtype_t, ndim=1] k2_rho_k,
         np.ndarray[dtype_t, ndim=3] k_eff,
         dtype_t dk,
         dtype_t dk_inv,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):
    "Calculate LATTTICEEASY spectra with effective momenta:"

    """This code is partially adapted from LatticeEasy.
       Viewer discretion is advised!"""

    cdef int i, x, y, z, ns
    cdef int px, py, pz
    cdef int cnt, l_int
    cdef int dimx, dimy, dimz2

    cdef dtype_t k, k2, l, c0, c1
    cdef dtype_t x1, x2, y1, y2
    cdef dtype_t wk, wk2, Fk_sq, nk


    if F_k_Re is None or F_k_Im is None or S_k is None or counts is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = S_k.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        S_k[i] = 0.0
        n_k[i] = 0.0
        rho_k[i] = 0.0


    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k = k_eff[z,y,x]

                k2 = k*k

                l = floor(k*dk_inv)

                i = int(l)

                cnt = 1 if (z == 0 or z == dimz2-1) else 2

                Fk_sq = F_k_Re[z,y,x]*F_k_Re[z,y,x]+F_k_Im[z,y,x]*F_k_Im[z,y,x]

                S_k[i] += cnt*(Fk_sq)
                k2_S_k[i] += cnt*k2*(Fk_sq)
                counts[i] += cnt

                w_k2 = k2 + wk2_term

                w_k = sqrt(w_k2)

                x1 = a_term*Pi_k_Re[z,y,x]
                x2 = a_term*Pi_k_Im[z,y,x]

                y1 = p_term*F_k_Re[z,y,x]
                y2 = p_term*F_k_Im[z,y,x]

                if w_k > 0:
                    nk = (coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 +
                                                x2*x2 + 2.0*x2*y2 + y2*y2)/w_k))
                else:
                    nk = 0

                #nk = (coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 +
                #      x2*x2 + 2.0*x2*y2 + y2*y2)/w_k))
                #nk = w_k*(Fk_sq) + (Pi_k_Re[z,y,x])/w_k

                n_k[i] += cnt*nk
                rho_k[i] += cnt*w_k*nk
                k2_rho_k[i] += cnt*k2*w_k*nk


def calc_spect_k2_eff_df(np.ndarray[dtype_t, ndim=3] F_k_Re,
         np.ndarray[dtype_t, ndim=3] F_k_Im,
         np.ndarray[dtype_t, ndim=3] Pi_k_Re,
         np.ndarray[dtype_t, ndim=3] Pi_k_Im,
         np.ndarray[dtype_t, ndim=1] counts,
         np.ndarray[dtype_t, ndim=1] S_k,
         np.ndarray[dtype_t, ndim=1] n_k,
         np.ndarray[dtype_t, ndim=1] rho_k,
         np.ndarray[dtype_t, ndim=3] k2_eff,
         dtype_t dk,
         dtype_t dk_inv,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):
    """Calculate Defrost spectra with the effective momenta
       (This does not propably work):"""


    """This code is partially adapted from Defrost.
       Viewer discretion is advised!"""

    cdef int i, x, y, z, ns
    cdef int px, py, pz
    cdef int cnt, l_int
    cdef int dimx, dimy, dimz2

    cdef dtype_t k, k2, l, c0, c1
    cdef dtype_t x1, x2, y1, y2
    cdef dtype_t wk, wk2, Fk_sq, nk


    if F_k_Re is None or F_k_Im is None or S_k is None or counts is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = S_k.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        S_k[i] = 0.0
        n_k[i] = 0.0
        rho_k[i] = 0.0


    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k2 = k2_eff[z,y,x]

                k = sqrt(k2)

                l = floor(k*dk_inv)

                i = int(l)

                c0 = (1.0 - (l-k)*(l-k))*(1.0 - (l-k)*(l-k))
                c1 = (1.0 - (l+1.0-k)*(l+1.0-k))*(1.0 - (l+1.0-k)*(l+1.0-k))

                cnt = 1 if (z == 0 or z == dimz2-1) else 2

                Fk_sq = F_k_Re[z,y,x]*F_k_Re[z,y,x]+F_k_Im[z,y,x]*F_k_Im[z,y,x]

                w_k2 = k2 + wk2_term

                w_k = sqrt(w_k2)

                x1 = a_term*Pi_k_Re[z,y,x]
                x2 = a_term*Pi_k_Im[z,y,x]

                y1 = p_term*F_k_Re[z,y,x]
                y2 = p_term*F_k_Im[z,y,x]

                if w_k > 0:
                    nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
                else:
                    nk = 0

                S_k[i] += c0*cnt*Fk_sq
                counts[i] += c0*cnt
                n_k[i] += c0*cnt*nk
                rho_k[i] += c0*cnt*w_k*nk

                if i < ns-1:
                    S_k[i+1] += c1*cnt*Fk_sq
                    counts[i+1] += c1*cnt
                    n_k[i] += c1*cnt*nk
                    rho_k[i] += c1*cnt*w_k*nk

def calc_spect_pi_h(np.ndarray[dtype_t, ndim=3] F_k_Re,
                    np.ndarray[dtype_t, ndim=3] F_k_Im,
                    np.ndarray[dtype_t, ndim=3] k_abs,
                    np.ndarray[itype_t, ndim=1] counts,
                    np.ndarray[dtype_t, ndim=1] spect_k,
                    dtype_t dk,
                    dtype_t dk_inv):
    "Calculate the gravitational wave spectrum:"

    cdef int i, x, y, z, ns
    cdef int px, py, pz
    cdef int cnt, l_int
    cdef int dimx, dimy, dimz2

    cdef dtype_t k, Fk_sq, c0

    if F_k_Re is None or F_k_Im is None or spect_k is None or counts is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = spect_k.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        spect_k[i] = 0.0


    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k = k_abs[z,y,x]

                l = floor(k*dk_inv)

                i = int(l)

                cnt = 1 if (z == 0 or z == dimz2-1) else 2

                Fk_sq = F_k_Re[z,y,x]*F_k_Re[z,y,x]+F_k_Im[z,y,x]*F_k_Im[z,y,x]

                spect_k[i] += cnt*((k)**3.0*Fk_sq)
                counts[i] += cnt


