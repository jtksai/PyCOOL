from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import math
import codecs
import time

from lattice import *

#Note that there is an error in fftw3 ver.0.2!!!!!!!!!!
#See https://bugs.launchpad.net/pyfftw !!!!!!!!!!!!!!!!

"""
Part of this code adapted from DEFROST
http://www.sfu.ca/physics/cosmology/defrost .
See http://arxiv.org/abs/0809.4904 for more information.
"""

def sample_defrost_cpu(lat, func, gamma, m2_eff):
    """Calculates a sample of random values in the lattice.
    Taken from Defrost-program.

    func = name of Cuda kernel
    n = size of cubic lattice
    gamma = -0.25 or +0.25
    m2_eff = effective mass

    This uses numpy to calculate FFTW.
    """
    import fftw3

    "Various constants:"
    mpl = lat.mpl
    n = lat.n
    nn = lat.nn
    os = 16
    nos = n*pow(os,2)
    dk = lat.dk
    dx = lat.dx
    dkos = dk/(2.*os)
    dxos = dx/os
    kcut = nn*dk/2.0
    norm = 0.5/(math.sqrt(2*pi*dk**3.)*mpl)*(dkos/dxos)

    ker = np.empty(nos, dtype=np.float64)
    fft = fftw3.Plan(ker, ker, direction='forward', flags=['measure'],
                     realtypes = ['realodd 10'])

    for k in xrange(nos):
        kk = (k+0.5)*dkos
        ker[k]=(kk*(kk**2. + m2_eff)**gamma)*math.exp(-(kk/kcut)**2.)
    fft.execute()
    fftw3.destroy_plan(fft)

    for k in xrange(nos):
        ker[k] = norm*ker[k]/(k+1)

    l0 = int(np.floor(np.sqrt(3)*n/2*os))

    tmp = np.zeros((n,n,n),dtype=np.float64)
    Fk = np.zeros((n,n,n/2+1),dtype=np.complex128)

    ker_gpu = gpuarray.to_gpu(ker)
    tmp_gpu = gpuarray.to_gpu(tmp)

    func(tmp_gpu, ker_gpu, np.uint32(nn), np.float64(os),
         np.uint32(lat.dimx), np.uint32(lat.dimy), np.uint32(lat.dimz),
         block = lat.cuda_block_1, grid = lat.cuda_grid)

    tmp += tmp_gpu.get()


    
    Fk = np.fft.rfftn(tmp)
    
    if lat.test==True:
        np.random.seed(1)
        
    rr1 = np.random.normal(size=Fk.shape) + np.random.normal(size=Fk.shape)*1j
    Fk *= rr1
    
    tmp = np.fft.irfftn(Fk)

    return tmp

def sample_defrost_cpu2(lat, func, gamma, m2_eff):
    """Calculates a sample of random values in the lattice

    lat = Lattice
    func = name of Cuda kernel
    n = size of cubic lattice
    gamma = -0.25 or +0.25
    m2_eff = effective mass

    This uses fftw3 to calculate FFTW.
    """
    import fftw3

    "Various constants:"
    mpl = lat.mpl
    n = lat.n
    nn = lat.nn
    os = 16
    nos = n*pow(os,2)
    dk = lat.dk
    dx = lat.dx
    dkos = dk/(2.*os)
    dxos = dx/os
    kcut = nn*dk/2.0
    norm = 0.5/(math.sqrt(2*pi*dk**3.)*mpl)*(dkos/dxos)

    ker = np.empty(nos, dtype= lat.prec_real)
    fft = fftw3.Plan(ker,ker, direction='forward', flags=['measure'],
                     realtypes = ['realodd 10'])

    for k in xrange(nos):
        kk = (k+0.5)*dkos
        ker[k] = kk*(kk**2. + m2_eff)**gamma*math.exp(-(kk/kcut)**2.)
    fft.execute()
    fftw3.destroy_plan(fft)

    for k in xrange(nos):
        ker[k] = norm*ker[k]/(k+1)

    tmp = np.zeros((n,n,n),dtype = lat.prec_real)
    Fk = np.zeros((n,n,n/2+1),dtype = lat.prec_complex)

    ker_gpu = gpuarray.to_gpu(ker)
    tmp_gpu = gpuarray.to_gpu(tmp)

    fft2 = fftw3.Plan(tmp, Fk, direction='forward', flags=['measure'])
    fft3 = fftw3.Plan(Fk, tmp, direction='forward', flags=['measure'])
    
    func(tmp_gpu, ker_gpu, np.uint32(nn), np.float64(os),
         np.uint32(lat.dimx), np.uint32(lat.dimy), np.uint32(lat.dimz),
         block = lat.cuda_block_1, grid = lat.cuda_grid)
    
    tmp += tmp_gpu.get()

    fft2.execute()
    fftw3.destroy_plan(fft2)
    
    if lat.test==True:
        np.random.seed(1)

    rr1 = np.random.normal(size=Fk.shape) + np.random.normal(size=Fk.shape)*1j
    Fk *= rr1
    
    fft3.execute()

    fftw3.destroy_plan(fft3)

    tmp *= 1./lat.VL

    return tmp

def sample_defrost_gpu(lat, func, gamma, m2_eff):
    """Calculates a sample of random values in the lattice

    lat = Lattice
    func = name of Cuda kernel
    n = size of cubic lattice
    gamma = -0.25 or +0.25
    m2_eff = effective mass

    This uses CuFFT to calculate FFTW.
    """
    import scikits.cuda.fft as fft
    import fftw3

    "Various constants:"
    mpl = lat.mpl
    n = lat.n
    nn = lat.nn
    os = 16
    nos = n*pow(os,2)
    dk = lat.dk
    dx = lat.dx
    dkos = dk/(2.*os)
    dxos = dx/os
    kcut = nn*dk/2.0
    norm = 0.5/(math.sqrt(2*pi*dk**3.)*mpl)*(dkos/dxos)

    ker = np.empty(nos,dtype = lat.prec_real)
    fft1 = fftw3.Plan(ker,ker, direction='forward', flags=['measure'],
                     realtypes = ['realodd 10'])

    for k in xrange(nos):
        kk = (k+0.5)*dkos
        ker[k]=kk*(kk**2. + m2_eff)**gamma*math.exp(-(kk/kcut)**2.)
    fft1.execute()
    fftw3.destroy_plan(fft1)

    for k in xrange(nos):
        ker[k] = norm*ker[k]/(k+1)

    Fk_gpu = gpuarray.zeros((n/2+1,n,n), dtype = lat.prec_complex)

    ker_gpu = gpuarray.to_gpu(ker)
    tmp_gpu = gpuarray.zeros((n,n,n),dtype = lat.prec_real)

    plan = fft.Plan(tmp_gpu.shape, lat.prec_real, lat.prec_complex)
    plan2 = fft.Plan(tmp_gpu.shape, lat.prec_complex, lat.prec_real)
    
    func(tmp_gpu, ker_gpu, np.uint32(nn), np.float64(os),
         np.uint32(lat.dimx), np.uint32(lat.dimy), np.uint32(lat.dimz),
         block = lat.cuda_block_1, grid = lat.cuda_grid)
    
    fft.fft(tmp_gpu, Fk_gpu, plan)
    
    if lat.test==True:
        np.random.seed(1)

    rr1 = (np.random.normal(size=Fk_gpu.shape)+
           np.random.normal(size=Fk_gpu.shape)*1j)

    Fk = Fk_gpu.get()
    Fk*= rr1
    Fk_gpu = gpuarray.to_gpu(Fk)

    fft.ifft(Fk_gpu, tmp_gpu, plan2)
    res = (tmp_gpu.get()).astype(lat.prec_real)

    res *= 1./lat.VL

    return res

def f_init(lat, field0, field_i, m2_eff, flag_method='defrost_cpu', homogQ=True):
    """
    Initialize the necessary fields

    lat = Lattice
    field_n = number of the field
    falg_gpu = 'cpu' or 'gpu'. If 'gpu' then Fast Fourier Transforms calculated
               on the gpu.
    homoQ = if True add the homogeneous value to the perturbation
    """

    "Open and compile Cuda kernels"
    f = codecs.open('init/gpu_3dconv.cu','r',encoding='utf-8')
    gpu_3dconv = f.read()
    f.close()

    mod = SourceModule(gpu_3dconv)
    gpu_conv = mod.get_function("gpu_3dconv")

    """Subtract one from the given field number in order to keep
        the notation consistent."""

    i = field_i-1
    fields = lat.fields

    if homogQ==True:
        c=1.
    else:
        c=0.

    if flag_method=='defrost_gpu':
        f = sample_defrost_gpu(lat, gpu_conv,-0.25, m2_eff) + c*field0
        print "\nField " + repr(field_i)+ " init on gpu done"
    elif flag_method=='defrost_cpu':
        f = sample_defrost_cpu(lat, gpu_conv,-0.25, m2_eff) + c*field0
        print "\nField " + repr(field_i)+ " init on cpu done"
    elif flag_method=='defrost_cpu2':
        f = sample_defrost_cpu2(lat, gpu_conv,-0.25, m2_eff) + c*field0
        print "\nField " + repr(field_i)+ " init on cpu done"

    return np.array(f, dtype = lat.prec_real)

def fp_init(lat, pi0, field_i, m2_eff, a_in,
            flag_method='defrost_cpu', homogQ=True):
    """
    Initialize the necessary fields
    
    lat = Lattice
    field_n = number of the field
    falg_gpu = 'defrost_cpu' or 'defrost_gpu'. If 'gpu' then Fast Fourier Transforms calculated
               on the gpu.
    homoQ = if True add the homogeneous value to the perturbation
    """

    "Open and compile Cuda kernels"
    f = codecs.open('init/gpu_3dconv.cu','r',encoding='utf-8')
    gpu_3dconv = f.read()
    f.close()

    mod = SourceModule(gpu_3dconv)
    gpu_conv = mod.get_function("gpu_3dconv")

    """Subtract one from the given field number in order to keep
        the notation consistent."""
    i = field_i-1
    fields = lat.fields

    if homogQ==True:
        c=1.
    else:
        c=0.

    if flag_method=='defrost_gpu':
        fp = sample_defrost_gpu(lat, gpu_conv, 0.25, m2_eff) + c*pi0
        print "Field " + repr(field_i)+ " time derivative init on gpu done"
    elif flag_method=='defrost_cpu':
        fp = sample_defrost_cpu(lat, gpu_conv, 0.25, m2_eff) + c*pi0
        print "Field " + repr(field_i)+ " time derivative init on cpu done"
    elif flag_method=='defrost_cpu2':
        fp = sample_defrost_cpu2(lat, gpu_conv, 0.25, m2_eff) + c*pi0
        print "Field " + repr(field_i)+ " time derivative init on cpu done"
    else:
        import sys
        sys.exit(("Init method ill defined!"))

    return np.array(fp, dtype = lat.prec_real)*a_in**2.


def rj_f_init(lat, Fk, m_field, a, H, f0):
    """This calculates initial values with a method used in
       'Non-Gaussianity from resonant curvaton decay'
       http://arxiv.org/abs/arXiv:0909.4535 .
       This has not been tested properly.
       Depends on init.f_init Cython code.
    """

    import init.f_init as fi

    Vol = lat.n**3.*lat.dx**3.
    dx = lat.dx

    ctilde = ([1,2,4,8]*lat.cc).astype(np.float64)

    fi.calc_f(Fk.real, Fk.imag, a, H, lat.dx, m_field, ctilde)

    f_pert = np.fft.irfftn(Fk)/(np.sqrt(Vol)*dx**(1.5))

    f = f0 + f_pert

    return f

def rj_fp_init(lat, Pik, m_field, a, H, pi0):
    """This calculates initial values with a method used in
       'Non-Gaussianity from resonant curvaton decay'
       http://arxiv.org/abs/arXiv:0909.4535 .
       This has not been tested properly.
       Depends on init.f_init Cython code.
    """

    import init.f_init as fi

    Vol = lat.n**3.*lat.dx**3.
    dx = lat.dx

    ctilde = ([1,2,4,8]*lat.cc).astype(np.float64)

    fi.calc_fp(Pik.real, Pik.imag, a, H, lat.dx, m_field, ctilde)

    pi_pert = np.fft.irfftn(Pik)/(np.sqrt(Vol)*dx**(1.5))

    pi = pi0 + pi_pert

    return pi

