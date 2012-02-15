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
                    nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
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
         np.ndarray[dtype_t, ndim=3] k2_eff,
         dtype_t dk,
         dtype_t dk_inv,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):

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

                k2 = k2_eff[z,y,x]

                k = sqrt(k2)

                l = floor(k*dk_inv)

                i = int(l)

                c0 = (1.0 - (l-k)*(l-k))*(1.0 - (l-k)*(l-k))
                c1 = (1.0 - (l+1.0-k)*(l+1.0-k))*(1.0 - (l+1.0-k)*(l+1.0-k))

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
                    nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
                else:
                    nk = 0

                #nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
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




