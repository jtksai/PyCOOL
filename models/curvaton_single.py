import numpy as np

"""
###############################################################################
# Define a scalar field model and a lattice
###############################################################################
"""

class Model:
    """Model class that defines the scalar field. Change these values
       for different models:"""

    def __init__(self):

        self.model_name = 'curvaton single field model self-interacting'

        "Model parameters and values:"

        "Reduced Planck mass (mpl) and regular Planck mass (MPl):"
        self.mpl = 1.0
        self.MPl = np.sqrt(8*np.pi)*self.mpl

        "Mass unit that is used to define other variables:"
        self.m = 1e-9

        "Scalar field masses:"
        #self.m2f1 = (3e-10)**2#0.0#
        self.m2f1 = (1e-9)**2
        self.m2f2 = 0.0
        self.m2_fields = [self.m2f1]

        "Coupling strengths:"
        self.g2 = 1e-8
        self.lamb2 = 1e-7

        "Radiation field:"
        self.lamb = 1.0e-16
        self.psi = np.sqrt(self.mpl)

        "Variance of f10 within our current Hubble volume:"
        self.delta_f10 = np.sqrt(4/(9*np.pi**2)*self.lamb*60**3.)

        "Initial values for the fields and the field time derivatives:"
        self.f10 = 2e-4*self.mpl
        #self.f10 = 5e-6*self.mpl

        self.df1_dt0 = 10.**-19.

        self.fields0 = [self.f10]
        self.pis0 = [self.df1_dt0]

        self.k_test = np.sqrt(1.6*self.lamb2*self.f10**2)/self.m
        self.dom_test = np.sqrt(self.lamb2/self.m2f1)*self.f10
        #self.q = self.g2*self.f10**2/(4*self.m2f1)

        "List of the potential functions:"

        "Potentials functions of the fields including self-interactions:"
        self.V_list = ["0.5*C1*f1**2 + 0.25*C2*f1**4"]

        "Interaction terms of the fields:"
        self.V_int = ['']

        """Numerical values for C1, C2, ... These will be multiplied by
           a**3*dtau:"""
        self.C_coeff = [self.m2f1, self.lamb2]

        "Numerical values for bare coefficients D1, D2, ..."
        self.D_coeff = []

        """List of functions which are in 'powerform' in potential. For
           example for potential V = 1 + sin(f1)**2 power_list = ['sin(f1)'].
           Field variables are included automatically."""
        self.power_list = []

        "Initial values for homogeneous radiation and matter components:"
        self.rho_r0 = 0.25*self.lamb*self.psi**4.
        self.rho_m0 = 0.

        "Time step:"
        #self.dtau = 1./(5000*self.m)
        self.dtau = 1./(1000*self.m)

        "Time step for homogeneous system:"
        self.dtau_hom = 1./(10000*self.m)
        #self.dtau_hom = 1./(1000*self.m)

        "Lattice side length:"
        self.L = 5./12./self.m

        "Lattice size, where n should be a power of two:"
        self.n = 64

        "Initial scale parameter:"
        self.a_in = 1.

        "Limit for scale factor in linearized evolution:"
        self.a_limit = 5

        "Initial and final times:"
        self.t_in = 0.
        self.t_fin = 1000./self.m
        self.t_fin_hom = 50./self.m

        "Use linearized evolution if True:"
        self.lin_evo = False#True#

        "Solve homogeneous field evolution if True:"
        self.homogenQ = False#True#

        "Set True to solve non-linearized evolution once:"
        self.evoQ = True#False#

        """Whether to do curvature perturbation (zeta) calculations
           (this disables post-processing). Also disables evoQ:"""
        self.zetaQ = False#True#

        """Whether to solve tensor perturbations:"""
        self.gwsQ = True#False#

        "The reference value at which curvature perturbation is calculated:"
        self.H_ref = 4e-13

        "Number of different simulations to run with identical initial values:"
        self.sim_num = 1

        "How frequently to calculate rho and error:"
        self.flush_freq = 8*64#256#*120
        self.flush_freq_hom = 128*8

        "If True write to file:"
        self.saveQ = True#False#

        "If True make a superfolder that has all the different simulations:"
        self.superfolderQ = False#True#

        "Name of the superfolder:"
        self.superfolder = 'zeta_run_6'

        """If True multiplies energy densities with 1/m^2.
            VisIt might not plot properly very small densities."""
        self.scale = False#True#

        """If fieldsQ = True save the field data (fields, rho etc.) in
           the Silo files:"""
        self.fieldsQ = False#True#

        "If spectQ = True calculate spectrums:"
        self.spectQ = True#False#

        "If distQ = True calculate empirical CDF and CDF:"
        self.distQ = False#True#

        """The used method to calculate spectrums. Options 'latticeeasy' and
           'defrost'. Defrost uses aliasing polynomial to smooth
           the spectrums. 'k2_eff' uses k^2_eff related to the discretized
           Laplacian to calculate the spectra."""
        self.spect_m = 'k2_eff'#'defrost'#'latticeeasy'#

        """If statQ = True calculate skewness and kurtosis of the fields:"""
        self.statsQ = True#False#

        """If field_r = True calculate also energy densities of fields
           without interaction terms:"""
        self.field_rho = True#False#

        """If field_lpQ = True calculate correlation lengths of
           the energy densities of the fields without interaction terms:"""
        self.field_lpQ = False#True#

        "If deSitter = True include -9H^2/(4m^2) term in \omega_k^2 term:"
        self.deSitterQ = True#False#

        """If testQ = True use a constant seed. Can be used for debugging and
           testing:"""
        self.testQ = False#True#

        """If m2_effQ = True writes a*m_eff/m to SILO file. This includes
           also comoving number density."""
        self.m2_effQ = True#False#

        "If csvQ = True writes curves from Silo files to csv files:"
        self.csvQ = True#False#

        """Maximum number of registers useb per thread. If set to None uses
           default values 24 for single and 32 for double precision.
           Note that this will also affect the used block size"""
        self.max_reg = None

        
        """For curvature perturbation studies disable post-processing
           by default:"""
        if self.zetaQ == True:
            self.evoQ = False
            self.spectQ = False
            self.gwsQ = False
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
