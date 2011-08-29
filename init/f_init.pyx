#!/usr/bin/python
#encoding: utf-8
# cconv.pyx
# cython: profile=True
import cython
import numpy as np
cimport numpy as np
ctypedef np.int32_t itype_t
ctypedef np.float64_t dtype_t

#declaring external GSL functions to be used
cdef extern from "math.h":
    double floor(double)
    double sqrt(double)
    double sin(double)
    double cos(double)

cdef extern from "gsl/gsl_rng.h":
   ctypedef struct gsl_rng_type:
       pass
   ctypedef struct gsl_rng:
       pass
   gsl_rng_type *gsl_rng_mt19937
   gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
  
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef extern from "gsl/gsl_randist.h":
   double gamma "gsl_ran_gamma"(gsl_rng * r,double,double)
   double gaussian "gsl_ran_gaussian"(gsl_rng * r,double)
  
#Cython code
def calc_f(np.ndarray[dtype_t, ndim=3] F_k_Re,
         np.ndarray[dtype_t, ndim=3] F_k_Im,
         double a,
         double H,
         double dx,
         double m,
         np.ndarray[dtype_t, ndim=1] c_tilde):
   """Creates initial perturbations for field with mass m and in a lattice with
      spacing dx and initial scale factor a and Hubble parameter H:"""
   cdef double ii, jj, kk, k2
   cdef int i, j, k, n
   cdef int dimx, dimy, dimz2
   cdef double w, twopi
   cdef double c0, c1, c2, c3

   twopi = 6.2831853071795864769252867665590

   c0 = c_tilde[0]
   c1 = c_tilde[1]
   c2 = c_tilde[2]
   c3 = c_tilde[3]

   dimx = F_k_Re.shape[2]
   dimy = F_k_Re.shape[1]
   dimz2 = F_k_Re.shape[0]

   n = dimx

   Vol = float(n)**3.*dx**3.

   w = twopi/float(n)

   for k from 0 <= k < dimz2:
       kk = cos(w*k)
       for j from 0 <= j < dimy:
           jj = cos(w*j)
           for i from 0 <= i < dimx:
               ii = cos(w*i)

               k2 = -(dx**-2.*(c0 + c1*(ii+jj+kk) + c2*(ii*jj+ii*kk+jj*kk) + c3*ii*jj*kk))

               sigma = sqrt((0.5*Vol*(k2+m**2.*a**2.+H**2.*a**2.)/(k2+m**2.*a**2.)**1.5))

               F_k_Re[k,j,i] = gaussian(r,sigma)
               F_k_Im[k,j,i] = gaussian(r,sigma)



def calc_fp(np.ndarray[dtype_t, ndim=3] Pi_k_Re,
         np.ndarray[dtype_t, ndim=3] Pi_k_Im,
         double a,
         double H,
         double dx,
         double m,
         np.ndarray[dtype_t, ndim=1] c_tilde):
   "Time derivative of field"
   cdef double ii, jj, kk, k2
   cdef int i, j, k, n
   cdef int dimx, dimy, dimz2
   cdef double w, twopi
   cdef double c0, c1, c2, c3

   twopi = 6.2831853071795864769252867665590

   c0 = c_tilde[0]
   c1 = c_tilde[1]
   c2 = c_tilde[2]
   c3 = c_tilde[3]

   dimx = Pi_k_Re.shape[2]
   dimy = Pi_k_Re.shape[1]
   dimz2 = Pi_k_Re.shape[0]

   n = dimx

   Vol = float(n)**3.*dx**3.

   w = twopi/float(n)

   for k from 0 <= k < dimz2:
       kk = cos(w*k)
       for j from 0 <= j < dimy:
           jj = cos(w*j)
           for i from 0 <= i < dimx:
               ii = cos(w*i)

               k2 = -(dx**-2.*(c0 + c1*(ii+jj+kk) + c2*(ii*jj+ii*kk+jj*kk) + c3*ii*jj*kk))

               sigma = sqrt((0.5*Vol*(k2+m**2.*a**2.+H**2.*a**2.)/(k2+m**2.*a**2.)**1.5))

               Pi_k_Re[k,j,i] = gaussian(r,sigma)
               Pi_k_Im[k,j,i] = gaussian(r,sigma)

