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


def calc_k_eff(np.ndarray[dtype_t, ndim=3] k_eff_x,
               np.ndarray[dtype_t, ndim=3] k_eff_y,
               np.ndarray[dtype_t, ndim=3] k_eff_z,
               np.ndarray[dtype_t, ndim=3] k_eff_abs,
               dtype_t dx_val):

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

        for y from 0 <= y < dimy:
            py = y if y <= dimy/2 else y - dimy

            for z from 0 <= z < dimz2:
                pz = z

                k_x = np.sin(2*pi*px/N)
                k_y = np.sin(2*pi*py/N)
                k_z = np.sin(2*pi*pz/N)

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
                    

def calc_spect_pi_h(np.ndarray[dtype_t, ndim=3] F_k_Re,
                    np.ndarray[dtype_t, ndim=3] F_k_Im,
                    np.ndarray[dtype_t, ndim=3] k_abs,
                    np.ndarray[itype_t, ndim=1] counts,
                    np.ndarray[dtype_t, ndim=1] spect_k,
                    dtype_t dk,
                    dtype_t dk_inv):

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

