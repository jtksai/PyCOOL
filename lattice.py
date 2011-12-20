import numpy as np
import sys
from misc_functions import *


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def checker(some_list,n):
    if len(some_list)!=n:
        print 'length of '+namestr(some_list,
                                   globals())[0] + ' is not equal to '+str(n)

def dim_test(a,b,c):
    if a*b!=c:
        print 'Init dimensions do not match with the evolution dimensions!'
        sys.exit()

"Various constants"
pi = np.pi


class Lattice:
    """
    Lattice Class Object.

    Defines lattice dimensions and CUDA grid dimensions.

    Lattice size = dimx*dimy*dimz = dimx**3.

    n = size of cubic lattice i.e. dimx = dimy = dimz = n

    L = lenght of the side of the lattice in the used units

    fields = number of fields

    mpl = reduced Planck mass in terms of the units used in
          potential function coefficients.

    m = mass of that field which can be used also as a unit,
        i.e. time in units of 1/m etc.

    precision = used precision.

    order = Order of the linearized integrator.
    
    alpha = dx/dtau

    dtau = Set dtau explicitly. If None use alpha to determine dtau.

    init_m = Determine if the initialization fft:s are calculated on gpu
             or cpu. Alternatives 'defrost_cpu', 'defrost_cpu2' and
             'defrost_gpu'. Default value = 'defrost_cpu'.

    hom_mode = if True add the homogeneous value to the perturbation field.

    If test = True numpy random sets seed = 1 and initialization values
              stay the same for different simulation runs.
              Used for debugging.

    field_rho = If True include energy density arrays for individual
                fields.

    field_lp = If True calculates correlation length of field energy
               densities i.e. field_rho has to be True.

    m2_eff = If True write sqrt(m2_eff) into SILO file. Might not work
             if for some reason m2_eff<0. Set then m2_eff = False.

    stat = If True calculate skew and kurtosis of field values.

    dist = If True calculate empirical CDF and PDF.

    unit_m = If 'm' spectrums in 1/m units.

    max_reg = Maximum number of registers per CUDA thread. For complicated
              functions number larger than 32 should be used. If None
              uses default value 32 for double precision.

    scale = If True multiplies energy densities with 1/m^2.
            VisIt might not plot properly very small densities.

    Note that the calculations in the H3_kernel (with cuda_block_2) are done
    by the innermost ( e.g 16x16 ) threads and the outer threads only read
    data into the shared memory.
    For optimal performance block_x should be a multiple of 16
    (i.e. block_x2 is not optimal).
    Some tested values:
        block_x = 16
        block_y = 16  or 8 (for 128**3 lattice)
        block_z = 1

        block_x2 = 16+2
        block_y2 = 16+2 or 8+2
        block_z2 = 1
    """

    def __init__(self, model, precision='double', order = 2,
                 alpha=40.0, init_m = 'defrost_cpu',
                 hom_mode = True, unit_m = 'm', scale = True):

        """def __init__(self, n, L, fields, mpl, m, precision='double', order = 2,
                 alpha=40.0, dtau = None, init_m = 'defrost_cpu',
                 hom_mode = True, test = False, field_rho = False,
                 field_lp = False, m2_eff = True, stats = False,
                 spects = False, dists = False,
                 unit_m = 'm', scale = True, max_reg = None):"""

        print '\nLattice size = ' + str(model.n) + '**3'

        self.init_mode = init_m
        self.hom_mode = hom_mode
        self.test = model.testQ
        
        "If unit == 'm' then uses [1/m] units in plots:"
        self.unit = unit_m

        self.scale = scale

        self.fieldsQ = model.fieldsQ

        self.m2_eff = True if model.m2_effQ and model.spectQ else False

        self.field_lp = model.field_lpQ

        self.stats = model.statsQ

        self.spect = model.spectQ

        self.dist = model.distQ

        self.postQ = True if self.spect or self.stats or self.dist else False

        "Lattice dimensions in x and k-space:"
        self.dimx = model.n
        self.dimy = model.n
        self.dimz = model.n
        self.dimz2 = int(self.dimz/2+1)

        self.dims_xy = (self.dimx, self.dimy)
        self.dims_xyz = (self.dimx, self.dimy, self.dimz)
        self.dims_k = (self.dimz2, self.dimy, self.dimx)
        self.dims_k_alt = (self.dimx, self.dimy, self.dimz2)

        self.stride = model.n*model.n
        self.VL = self.dimx*self.dimy*self.dimz

        "Reduced Planck mass in terms of m:"
        self.mpl = model.mpl
        "The mass that is used in hte units:"
        self.m = model.m

        "This is used frequently in the evolution equation of a:"
        self.VL_reduced = self.VL*model.mpl**2.0

        "Integrator order:"
        self.order = order
        if order not in [2, 4, 6, 8]:
            import sys
            sys.exit("Integrator order has to be  2, 4, 6 or 8!")

        "alpha = dx/dtau"
        self.alpha = alpha

        self.L = float(model.L)
        self.dx = float(model.L)/model.n

        if model.dtau != None:
            self.dtau = model.dtau
        else:
            self.dtau = self.dx/self.alpha

        if model.dtau_hom != None:
            self.dtau_hom = model.dtau_hom
        else:
            self.dtau_hom = self.dx/self.alpha

        #"Test Courant Condition:"
        #if self.dx/self.dtau < self.alpha:
        #    import sys
        #    sys.exit("Courant Condition violated!")

        self.dk = 2*np.pi/(model.n*self.dx)

        self.n = model.n
        "Nyquist frequency:"
        self.nn = model.n/2+1
        "Highest wavenumber on 3D grid:"
        self.ns = int(np.sqrt(3)*(model.n/2) + 1)

        """
        ########################################
        #Different CUDA grid and block choises:
        ########################################
        """

        if model.n == 32 or model.max_reg > 32:
            self.block_x = 16
            self.block_y = 8
            self.block_z = 1

            self.block_x2 = 16+2
            self.block_y2 = 8+2
            self.block_z2 = 1
        else:
            self.block_x = 16
            self.block_y = 16
            self.block_z = 1

            self.block_x2 = 16+2
            self.block_y2 = 16+2
            self.block_z2 = 1

        """The following grid and block sizes are used in
           the linearized evolution:"""

        self.maxk = 3*(model.n/2)**2
        self.k_g = model.n/16

        self.dim_lH = (self.dimx, 12*self.k_g)

        self.block_lin_Hx = 16
        self.block_lin_Hy = 12
        self.block_lin_Hz = 1

        self.grid_x = self.dimx/self.block_x
        self.grid_y = self.dimy/self.block_y

        self.gridx_lin_H = self.k_g
        self.gridy_lin_H = self.k_g
        self.gridz_lin_H = 1

        "Tuples used when calling kernel functions:"
        self.cuda_grid = (self.grid_x, self.grid_y)
        self.cuda_block_1 = (self.block_x, self.block_y, self.block_z)
        self.cuda_block_2 = (self.block_x2, self.block_y2, self.block_z2)
        self.cuda_lin_block = (self.block_x,1,1)

        self.cuda_g_lin_H = (self.gridx_lin_H, self.gridy_lin_H)
        self.cuda_b_lin_H = (self.block_lin_Hx, self.block_lin_Hy,
                             self.block_lin_Hz)


        """Laplacian and nabla squared discretization coefficients
        Taken from "Michael Patra, Mikko Karttunen: Stencils with isotropic
        discretization error for differential operators"."""

        if precision == "float":
            self.prec_real = np.float32
            self.prec_complex = np.complex64
            self.prec_string = "float"
            self.complex_string = "float2"
            self.f_term='f'
            "Set the upper limit for the number of registers per thread:"
            if model.max_reg == None:
                self.reglimit = '24'
            else:
                self.reglimit = str(model.max_reg)
            "Laplacian and nabla squared discretization coefficients:"
            self.cc = np.array([-64./15., 7./15., 1.0/10.0, 1.0/30.0],
                               dtype=np.float32)
            self.cc_a = self.cc

        elif precision == "double":
            self.prec_real = np.float64
            self.prec_complex = np.complex128
            self.prec_string = "double"
            self.complex_string = "double2"
            self.f_term=''
            "Set the upper limit for the number of registers per thread:"
            if model.max_reg == None:
                self.reglimit = '32'
            else:
                self.reglimit = str(model.max_reg)
            "Laplacian and nabla squared discretization coefficients:"
            self.cc = np.array([-64./15., 7./15., 1.0/10.0, 1.0/30.0],
                               dtype=np.float64)
            self.cc_a = self.cc

        "Define the field variables:"

        self.fields = len(model.fields0)

        self.field_list = ['f'+str(i) for i in xrange(1,self.fields+1)]
        self.field_back_list = ['f0'+str(i) for i in xrange(1,self.fields+1)]
        self.dfield_list = ['df'+str(i) for i in xrange(1,self.fields+1)]

        self.fields = self.fields
        self.field_rho = model.field_rho

class Potential:
    """Potential function class:
       model = Model object
       lat = lattice class object
       n = degree of V"""
    def __init__(self,  lat, model, dV_list = None, d2V_list = None,
                 automatic = True, n=20):

        self.model_name = model.model_name

        "List of variables:"
        self.f_list = lat.field_list

        "Replacement list used in format_to_cuda function:"
        self.power_list = model.power_list
        self.power_list.extend(self.f_list)

        "List of potential functions and interaction terms:"
        self.v_l = model.V_list
        self.v_int = model.V_int

        

        "Form the total potential and interaction functions:"
        if len(self.v_l)>0:
            self.V = self.v_l[0]
            for i in xrange(1,len(self.v_l)):
                self.V += ' + ' + self.v_l[i]
            if len(self.v_int)>0 and self.v_int != ['']:
                for x in self.v_int:
                    self.V += ' + ' + x
        else:
            if len(self.v_int)>0 and self.v_int != ['']:
                self.V = self.v_int[0]
                for i in xrange(1,len(self.v_int)):
                    self.V += ' + ' + self.v_int[i]
            else:
                self.V = ''

        if len(self.v_int)>0 and self.v_int != ['']:
            self.V_int = self.v_int[0]
            for i in xrange(1,len(self.v_int)):
                self.V_int += ' + ' + self.v_int[i]
        else:
            self.V_int = '0.0'

        if self.V == '' or None:
            import sys
            sys.exit(("\nAll derivation and no potential function " +
                      "makes PyCOOL a dull boy!\n"))

        "List of different coefficients in potential function:"
        self.C_list = ['C'+str(i) for i in xrange(1,len(model.C_coeff)+1)]
        self.D_list = ['D'+str(i) for i in xrange(1,len(model.D_coeff)+1)]

        self.C_coeff = model.C_coeff
        self.C_coeffs_np = np.array(model.C_coeff,dtype = lat.prec_real)
        self.D_coeff = model.D_coeff
        if len(model.D_coeff) >0:
            self.D_coeffs_np = np.array(model.D_coeff,dtype = lat.prec_real)
        else:
            self.D_coeffs_np = np.array([0],dtype = lat.prec_real)
        self.n_coeffs = len(model.C_coeff)

        """The following calculations will try to calculate the necessary
           forms needed in the different CUDA kernels:"""

        "Potential function V_{i} of field i in CUDA form used in H3 part:"
        if self.v_l != None and automatic:
            self.V_i_H3 = [V_calc(self.v_l[i], n, self.f_list, i+1,
                                  self.power_list, self.C_list, self.D_list,
                                  'H3', deriv_n=0,multiplier='4')
                           for i in xrange(lat.fields)]
        else:
            self.V_i_H3 = [None for f in xrange(lat.fields)]

        """Interaction term V_{int} of the fields in CUDA form used in
           the H3 kernel:"""
        if len(self.v_int)>0 and self.v_int != ['']:
            self.V_int_H3 = V_calc(self.V_int, n, self.f_list, lat.fields,
                                   self.power_list, self.C_list, self.D_list,
                                   'H3', deriv_n=0, multiplier='4')
        else:
            "Use zero to avoid CUDA error messages:"
            self.V_int_H3 = '0.0'

        "Derivative dV/df_i for all field variables f_i in CUDA form:"
        self.dV_H3 = [V_calc(self.V, n, self.f_list, i+1, self.power_list,
                             self.C_list, self.D_list, 'H3', deriv_n=1)
                         for i in xrange(lat.fields)]

        """Potential function V_{i} of field i in CUDA form used in rho and
           pressure kernels:"""
        if self.v_l != None and automatic:
            self.V_i_rp = [V_calc(self.v_l[i], n, self.f_list, i+1,
                                  self.power_list, self.C_list, self.D_list,
                                  'rp', deriv_n=0)
                           for i in xrange(lat.fields)]
        else:
            self.V_i_rp = [None for f in xrange(lat.fields)]


        """Interaction term V_{int} of the fields in CUDA form used in
           the rho and pressure kernels:"""
        if len(self.v_int)>0 and self.v_int != ['']:
            self.V_int_rp = V_calc(self.V_int, n, self.f_list, lat.fields,
                                   self.power_list, self.C_list, self.D_list,
                                   'rp', deriv_n=0)
        else:
            "Use zero to avoid CUDA error messages:"
            self.V_int_rp = '0.0'


        "Derivative d2V/df_i^2 for all field variables f_i in CUDA form:"
        self.d2V_Cuda = [V_calc(self.V , n, self.f_list, i+1,
                                self.power_list, self.C_list,
                                self.D_list, 'H3', deriv_n=2)
                         for i in xrange(lat.fields)]

        if self.v_l != None and automatic:
            self.V_pd_i = [V_calc(self.v_l[i] , n, self.f_list, i+1,
                                  self.power_list,  self.C_list, self.D_list,
                                  'pd', multiplier = '4')
                           for i in xrange(lat.fields)]
        else:
            self.V_pd_i = [None for f in xrange(lat.fields)]

        if len(self.v_int)>0 and self.v_int != ['']:
            self.V_pd_int = V_calc(self.V_int, n, self.f_list, lat.fields,
                                   self.power_list, self.C_list, self.D_list,
                                   'pd', multiplier = '4')
        else:
            "Use zero to avoid CUDA error messages:"
            self.V_pd_int = '0.0'

        "Derivative d2V/df_i^2 for all field variables f_i in CUDA form:"
        self.d2V_pd = [V_calc(self.V , n, self.f_list, i+1,
                              self.power_list,  self.C_list, self.D_list,
                              'pd', deriv_n=2)
                       for i in xrange(lat.fields)]

        """Read the different numerical coefficients in front of the
           C_i terms in dV/df and d^2V/df^2 functions:"""
        #self.dV_mult = dV_coeffs(lat, self.V, self.f_list, self.C_list,
        #                         deriv_n = 1)
        #self.d2V_mult = dV_coeffs(lat, self.V, self.f_list, self.C_list,
        #                         deriv_n = 2)

        """Functions used in the linearized evolution:"""

        """Calculate potential only for the last field since it is the same for
            all fields."""

        if model.lin_evo:

            self.V_back = V_calc_lin(self.V, n, self.f_list, lat.fields,
                                     self.power_list, self.C_list, self.D_list,
                                     self.C_coeff, self.D_coeff,
                                     deriv_n=0)

            self.dV_back = [V_calc_lin(self.V, n, self.f_list, i+1,
                                       self.power_list,
                                       self.C_list, self.D_list,
                                       self.C_coeff, self.D_coeff,
                                       deriv_n=1)
                            for i in xrange(lat.fields)]

            self.d2V_back = [V_calc_lin(self.V, n, self.f_list, i+1,
                                       self.power_list,
                                       self.C_list, self.D_list,
                                        self.C_coeff, self.D_coeff,
                                       deriv_n=2)
                            for i in xrange(lat.fields)]

        "Lenght of the constant memory arrays:"

        if len(self.D_coeff) >0:
            self.d_coeff_l = len(self.D_coeff)
        else:
            self.d_coeff_l = 1
        self.f_coeff_l = 2 + self.n_coeffs
        self.g_coeff_l = 3 + self.n_coeffs
        self.h_coeff_l = 2
        self.p_coeff_l = 2 + self.n_coeffs
        self.lin_coeff_l = 3 + lat.fields


        """Functions needed to update Constant memory arrays used
        in the kernel functions:"""
        #Note that these could have to be edited manually for
        #some nasty potential functions!


    def h_array(self, lat, a, dt):
        "Constant memory array h_coeff coefficients used in H2 kernel"
        return np.array([dt/(a**2.),dt/(a**3.)],dtype = lat.prec_real)

    def f_array(self, lat, a, dt):
        "Constant memory array f_coeff coefficients"
        return np.array([a, a*dt*lat.dx**(-2.)], dtype = lat.prec_real)

    def H3_V_coeff(self, sim, dt):
        a = sim.a
        res = a**3.*dt*self.C_coeffs_np
        return res

    def H3_dV_coeff(self, sim, dt):
        "The coefficients of dV/df_i:"
        import numpy as np
        a = sim.a
        res = a**4.*dt*self.C_coeffs_np*self.dV_mult
        return res

    def H3_d2V_coeff(self, sim, dt):
        "The coefficients of d2V/df_i^2:"
        import numpy as np
        a = sim.a
        res = a**4.*dt*self.C_coeffs_np*self.d2V_mult
        return res

    def g_array(self, lat, a):
        """Constant memory array g_coeff coefficients used in the rho and
        pressure kernels"""
        arr = np.array([1./(2.*a**6.), 1./(2.*a**2.)*lat.dx**(-2.),
                        -1./(6.*a**2.)*lat.dx**(-2.)], dtype = lat.prec_real)
        return arr

    def lin_array(self, lat, sim, dt):
        "Constant memory array used in the linearized mode equations"
        
        a = sim.a
        f0 = [field.f0 for field in sim.fields]

        arr = [lat.dk**2.0, 0.5*dt*a**(-2.0), dt*a**2.0]

        for field in sim.fields:
            arr.append(a**2*field.d2V(*f0))

        res = np.array(arr, dtype = lat.prec_real)

        return res

    def p_coeff(self, lat, sim):
        "Constant memory array used in the pa kernel:"
        
        a = sim.a
        
        arr = [1./(a**4.), lat.dx**(-2.)]
        for x in self.C_coeff:
            arr.append(a**2.*x)

        res = np.array(arr, dtype = lat.prec_real)

        return res

    def k_array(self, lat, sim, w_k):
        "Constant memory array k_coeff coefficients used in spectrum kernels"

        a = sim.a
        p = sim.p

        return np.array([lat.dk, w_k, 1./a**3.,-p/(6.*lat.VL*a)],
                        dtype = lat.prec_real)


#####################################################################################
