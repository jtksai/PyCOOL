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

        self.model_name = 'curvaton'

        "Model parameters and values:"

        "Reduced Planck mass (mpl) and regular Planck mass (MPl):"
        self.mpl = 1.0
        self.MPl = np.sqrt(8*np.pi)*self.mpl

        "Mass unit that is used to define other variables:"
        self.m = 1e-8

        "Scalar field masses:"
        self.m2f1 = self.m**2.
        self.m2f2 = 0.0

        "Coupling strength:"
        self.g2 = 1e-4

        "Initial values for the fields and the field time derivatives:"

        self.f10 =  2e-3*self.mpl
        #self.f10 =  2e-4*self.mpl
        self.f20 = 1e-16*self.mpl

        self.df1_dt0 = 10.**-20.
        self.df2_dt0 = 10.**-20.

        "Radiation field:"
        self.lamb = 1.0e-16
        self.psi = np.sqrt(self.mpl)

        self.fields0 = [self.f10, self.f20]
        self.pis0 = [self.df1_dt0, self.df2_dt0]

        self.q = self.g2*self.f10**2/(4*self.m2f1)

        "List of the potential functions:"

        "Potentials functions of the fields including self-interactions:"
        self.V_list = ["0.5*C1*f1**2","0.5*C2*f2**2"]

        "Interaction terms of the fields:"
        self.V_int = ["0.5*C3*f1**2*f2**2"]

        """Numerical values for C1, C2, ... These will be multiplied by
           a**3*dtau:"""
        self.C_coeff = [self.m2f1, self.m2f2, self.g2]

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
        self.dtau = 1./(5000*self.m)
        #self.dtau = 1./(1000*m)

        self.dtau_hom = 1./(10000*self.m)
        #self.dtau_hom = 1./(1000*m)

        "Lattice side length:"
        self.L = 5./3./self.m

        "Lattice size, where n should be a power of two:"
        self.n = 64

        "Initial scale parameter:"
        self.a_in = 1.

        "Initial and final times:"
        self.t_in = 0.
        self.t_fin = 200./self.m
        #self.t_fin = 10./m

        "How frequently to save data:"
        self.flush_freq = 256*2
        self.flush_freq_hom = 128*8

        "Set if to use linearized evolution:"
        self.lin_evo = True


        "If spectQ = True calculate spectrums at the end:"
        self.spectQ = True

        "If distQ = True calculate empirical CDF and CDF at the end:"
        self.distQ = True

        """The used method to calculate spectrums. Options 'latticeeasy' and
           'defrost'. Defrost uses aliasing polynomial to smooth
           the spectrums."""
        self.spect_m = 'defrost'#'latticeeasy'

        """If statQ = True calculate skewness and kurtosis of the fields:"""
        self.statsQ = True

        """If field_r = True calculate also energy densities of fields
           without interaction terms:"""
        self.field_rho = True

        """If field_lpQ = True calculate correlation lengths of
           the energy densities of the fields without interaction terms:"""
        self.field_lpQ = False

        "If deSitter = True include -9H^2/(4m^2) terms in \omega_k^2 term:"
        self.deSitterQ = True

        """If testQ = True use a constant seed. Can be used for debugging and
           testing:"""
        self.testQ = False

        "If m2_effQ = True writes a*m_eff/m to SILO file."
        self.m2_effQ = True

        """Maximum number of registers useb per thread. If set to None uses
           default values 24 for single and 32 for double precision.
           Note that this will also affect the used block size"""
        self.max_reg = None

        "Non-gaussianity related variables:"
        
        "Whether to do non-Gaussianity calculations:"
        self.nonGaussianityQ = False
        "Number of different simulations to run with identical intial values:"
        self.sim_num = 2
        
        "For non-Gaussianty studies disable post-processing by default:"
        if self.nonGaussianityQ == True:
            self.lin_evo = False
            self.spectQ = False
            self.distQ = False
            self.statsQ = False
            self.field_rho = False
            self.field_lpQ = False
            self.testQ = False
            self.m2_effQ = False
            self.flush_freq = 256*120
            self.flush_freq_hom = 128*8


