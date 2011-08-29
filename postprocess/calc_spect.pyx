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
         np.ndarray[dtype_t, ndim=1] f_total,
         np.ndarray[dtype_t, ndim=1] n_k,
         np.ndarray[dtype_t, ndim=1] rho_k,
         dtype_t dk,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):

    """This code is adapted from LatticeEasy."""

    cdef int i, x, y, z
    cdef int px, py, pz
    cdef int cnt
    cdef dtype_t k, wk, wk2, Fk_sq, nk
    cdef dtype_t x1, x2, y1, y2
    cdef int dimx, dimy, dimz2

    if F_k_Re is None or counts is None or f_total is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = counts.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        f_total[i] = 0.0
        n_k[i] = 0.0
        rho_k[i] = 0.0

    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k = sqrt(float(px*px + py*py + pz*pz))

                i = int(k)

                cnt = 1 if (z == 0 or z == dimz2-1) else 2

                Fk_sq = F_k_Re[z,y,x]*F_k_Re[z,y,x]+F_k_Im[z,y,x]*F_k_Im[z,y,x]

                f_total[i] += cnt*(Fk_sq)
                counts[i] += cnt

                w_k2 = k*k*dk*dk + wk2_term

                w_k = sqrt(w_k2)

                x1 = a_term*Pi_k_Re[z,y,x]
                x2 = a_term*Pi_k_Im[z,y,x]

                y1 = p_term*F_k_Re[z,y,x]
                y2 = p_term*F_k_Im[z,y,x]

                nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)
                #nk = w_k*(Fk_sq) + (Pi_k_Re[z,y,x])/w_k

                n_k[i] += cnt*nk
                rho_k[i] += cnt*w_k*nk


def calc_df(np.ndarray[dtype_t, ndim=3] F_k_Re,
         np.ndarray[dtype_t, ndim=3] F_k_Im,
         np.ndarray[dtype_t, ndim=3] Pi_k_Re,
         np.ndarray[dtype_t, ndim=3] Pi_k_Im,
         np.ndarray[dtype_t, ndim=1] counts,
         np.ndarray[dtype_t, ndim=1] f_total,
         np.ndarray[dtype_t, ndim=1] n_k,
         np.ndarray[dtype_t, ndim=1] rho_k,
         dtype_t dk,
         dtype_t wk2_term,
         dtype_t a_term,
         dtype_t p_term,
         dtype_t coeff):

    """This code is partially adapted from Defrost."""

    cdef int i, x, y, z, ns
    cdef int px, py, pz
    cdef int cnt, l_int
    cdef int dimx, dimy, dimz2

    cdef dtype_t k, l, c0, c1
    cdef dtype_t x1, x2, y1, y2
    cdef dtype_t wk, wk2, Fk_sq, nk


    if F_k_Re is None or F_k_Im is None or f_total is None or counts is None:
        raise ValueError("Input arrays cannot be None")

    dimx = F_k_Re.shape[2]
    dimy = F_k_Re.shape[1]
    dimz2 = F_k_Re.shape[0]

    ns = f_total.shape[0]

    for i from 0 <= i < ns:
        counts[i] = 0
        f_total[i] = 0.0
        n_k[i] = 0.0
        rho_k[i] = 0.0


    for x from 0 <= x < dimx:
        px = x if x <= dimx/2 else x - dimx

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k = sqrt(float(px*px + py*py + pz*pz))

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

                nk = coeff*(w_k*(Fk_sq) + (x1*x1 + 2.0*x1*y1 + y1*y1 + x2*x2 + 2.0*x2*y2 + y2*y2)/w_k)


                f_total[i] += c0*cnt*Fk_sq
                counts[i] += c0*cnt
                n_k[i] += c0*nk
                rho_k[i] += c0*w_k*nk

                if i < ns-1:
                    f_total[i+1] += c1*cnt*Fk_sq
                    counts[i+1] += c1*cnt
                    n_k[i] += c1*nk
                    rho_k[i] += c1*w_k*nk









