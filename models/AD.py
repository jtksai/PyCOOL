import numpy as np
pi = np.pi

"""
###############################################################################
# Define a scalar field model and a lattice
###############################################################################
"""


class Model:
    """Model class that defines the scalar field. Change these values
       for different models:"""

    def __init__(self):

        self.model_name = 'Affleck-Dine mechanism'

        "Model parameters and values:"

        "Reduced Planck mass (mpl) and regular Planck mass (MPl):"
        self.mpl = 1.0
        self.MPl = np.sqrt(8*np.pi)*self.mpl

        "Mass unit that is used to define other variables:"
        self.m = 1e3/(2.4353431507105459e+18)

        "Scalar field masses:"
        self.m2f1 = self.m**2.
        self.m2f2 = self.m**2.
        self.m2_fields = [self.m2f1,self.m2f2]

        "Coupling strength:"
        self.K = -0.1
        self.M = 1.0#1e14/(2.4353431507105459e+18)

        "Initial values for the fields and the field time derivatives:"

        self.f10 =  3*1e14/(2.4353431507105459e+18)
        self.f20 = 3*1e14/(2.4353431507105459e+18)#0.

        self.df1_dt0 = 3*1e14/(2.4353431507105459e+18)*self.m
        self.df2_dt0 = 0.

        self.fields0 = [self.f10, self.f20]
        self.pis0 = [self.df1_dt0, self.df2_dt0]

        "List of the potential functions:"

        "Potentials functions of the fields including self-interactions:"

        #self.V_list = None
        self.V_list = ['C1*f1**2', 'C2*f2**2']

        "Interaction terms of the fields:"
        #self.V_int = ["C1*(log(1+D1*(f1**2+f2**2)))+C2*(f1**2+f2**2)**3"]
        self.V_int = [("C3*(f1**2+f2**2)*(log(D1*(f1**2+f2**2)))"+
                       "+C4*(f1**2+f2**2)**5" +
                       "+C5*(f1**6-f2**6)"+
                       "+C6*(f1**2*f2**4-f2**2*f1**4)"
                       )]

        """Numerical values for C1, C2, ... These will be multiplied by
           a**3*dtau:"""
        self.C_coeff = [self.m2f1, self.m2f2, self.K*self.m**2,
                        1./(self.M**6),
                        np.sqrt(40)*self.m**2*2.0/(6.0*self.M**3),
                        np.sqrt(40)*self.m**2*30.0/(6.0*self.M**3)]

        "Numerical values for bare coefficients D1, D2, ..."
        self.D_coeff = [1./self.M**2]
        #self.D_coeff = []

        """List of functions which are in 'powerform' in potential. For
           example for potential V = 1 + sin(f1)**2 power_list = ['sin(f1)'].
           Field variables are included automatically."""
        self.power_list = []

        "Initial and final times:"
        self.t_in = 0./self.m
        self.t_fin = 100./self.m
        self.t_fin_hom = 10000./self.m

        "Initial values for homogeneous radiation and matter components:"
        self.rho_r0 = 0.#3.*(2./(3.*self.t_in))**2.
        self.rho_m0 = 0.#3.*(self.m)**2

        "Time step:"
        self.dtau = 0.001/self.m
        #self.dtau = 1./(1000*m)

        "Time step for homogeneous system:"
        self.dtau_hom = 1./(2000*self.m)

        "Lattice side length:"
        self.L = 8./self.m

        "Lattice size, where n should be a power of two:"
        self.n = 64

        "Initial scale parameter:"
        self.a_in = 1.0#0.1*(self.t_in*self.m)**(2./3.)

        "Limit for scale factor in linearized evolution:"
        self.a_limit = 2

        "Set if to use linearized evolution:"
        self.lin_evo = True#False

        "Solve homogeneous field evolution if True:"
        self.homogenQ = False

        "Set True to solve non-linearized evolution:"
        self.evoQ = False#True#
        """Whether to do curvature perturbation (zeta) calculations
           (this disables post-processing). Also disables evoQ:"""
        self.zetaQ = False#True#

        """Whether to solve tensor perturbations:"""
        self.gwsQ = False#True#

        "Number of different simulations to run with identical intial values:"
        self.sim_num = 1

        "How frequently to save data:"
        self.flush_freq = 4*256
        self.flush_freq_hom = 128*8

        "If True write to file:"
        self.saveQ = True#False#

        "If True make a superfolder that has all the different simulations:"
        self.superfolderQ = False#True#

        "Name of the superfolder:"
        self.superfolder = 'zeta_run_1'

        """If True multiplies energy densities with 1/m^2.
            VisIt might not plot properly very small densities."""
        self.scale = True

        """If fieldsQ = True save the field data (fields, rho etc.) in
           the Silo files:"""
        self.fieldsQ = True#False#

        "The used discretization. Options 'defrost' or 'hlattice'."
        self.discQ = 'defrost'#'latticeeasy'#'hlattice'#

        "If spectQ = True calculate spectrums at the end:"
        self.spectQ = True

        """The used method to calculate gravitaional spectrums.
           Options 'std' which uses a continuum based wave numbers
           and 'k_eff' which uses k^_eff related to the discretized
           Laplacian to calculate the spectra."""
        self.spect_gw_m = 'std'#'k_eff'#

        #This has been depracated:"
        """The used method to calculate spectrums. Options 'latticeeasy' and
           'defrost'. Defrost uses aliasing polynomial to smooth
           the spectrums."""
        #self.spect_m = 'defrost'#'latticeeasy'

        "If distQ = True calculate empirical CDF and CDF at the end:"
        self.distQ = False#True

        """If statQ = True calculate skewness and kurtosis of the fields:"""
        self.statsQ = True

        """If field_r = True calculate also energy densities of fields
           without interaction terms:"""
        self.field_rho = False

        """If field_lpQ = True calculate correlation lengths of
           the energy densities of the fields without interaction terms:"""
        self.field_lpQ = False

        "If deSitter = True include -9H^2/(4m^2) terms in \omega_k^2 term:"
        self.deSitterQ = True#False#

        """If testQ = True use a constant seed. Can be used for debugging and
           testing:"""
        self.testQ = False

        """If m2_effQ = True writes a*m_eff/m to SILO file. This includes
           also comoving number density."""
        self.m2_effQ = True#False

        "If csvQ = True writes curves from Silo files to csv files:"
        self.csvQ = True#False#

        """Maximum number of registers useb per thread. If set to None uses
           default values 24 for single and 32 for double precision.
           Note that this will also affect the used block size"""
        self.max_reg = 45

        
        """For curvature perturbation studies disable post-processing
           by default:"""
        if self.zetaQ == True:
            self.evoQ = False
            self.spectQ = False
            self.distQ = False
            self.statsQ = False
            self.fieldsQ = False
            self.field_rho = False
            self.field_lpQ = False
            self.testQ = False
            self.m2_effQ = False
            self.flush_freq = 256*120*100000
            self.superfolderQ = True
            self.saveQ = False
