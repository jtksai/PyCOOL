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

        self.model_name = 'oscillon'

        "Model parameters and values:"

        "Reduced Planck mass (mpl) and regular Planck mass (MPl):"
        self.mpl = 1.0
        self.MPl = np.sqrt(8*np.pi)*self.mpl

        "Mass unit that is used to define other variables:"
        self.m = 5e-6*self.mpl

        "Scalar field masses:"
        self.m2f1 = self.m**2.
        self.m2f2 = 0.0

        "Coupling strength:"
        self.lamb = 2.8125e-6
        self.g2 = self.lamb**2/0.1

        "Initial values for the fields and the field time derivatives:"

        self.f10 = np.sqrt((3*self.lamb)/(5*self.g2))*self.m
        self.df1_dt0 = 1e-16*self.m
        self.fields0 = [self.f10]
        self.pis0 = [self.df1_dt0]

        "List of the potential functions:"

        "Potentials functions of the fields including self-interactions:"
        self.V_list = ["0.5*C1*f1**2"]

        "Interaction terms of the fields:"
        self.V_int = ["(-0.25*C2*f1**4)","0.16666666666666666*C3*f1**6"]

        """Numerical values for C1, C2, ... These will be multiplied by
           a**3*dtau:"""
        self.C_coeff = [self.m2f1, self.lamb, self.g2/self.m**2]

        "Numerical values for bare coefficients D1, D2, ..."
        self.D_coeff = []

        """List of functions which are in 'powerform' in potential. For
           example for potential V = 1 + sin(f1)**2 power_list = ['sin(f1)'].
           Field variables are included automatically."""
        self.power_list = []

        "Initial values for homogeneous radiation and matter components:"
        self.rho_r0 = 0.
        self.rho_m0 = 0.

        "Time step:"
        self.dtau = 0.005/(self.m)

        "Lattice side length:"
        self.L = 400./self.m

        "Lattice size, where n should be a power of two:"
        self.n = 128

        "Initial scale parameter:"
        self.a_in = 1.

        "Initial and final times:"
        self.t_in = 0.
        self.t_fin = 5000./self.m

        "How frequently to save data:"
        self.flush_freq = 4*1024

        "Set if to use linearized evolution:"
        self.lin_evo = False

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
        self.deSitterQ = False

        """If testQ = True use a constant seed. Can be used for debugging and
           testing:"""
        self.testQ = False

        "If m2_effQ = True writes a*m_eff/m to SILO file."
        self.m2_effQ = False

        """Maximum number of registers useb per thread. If set to None uses
           default values 24 for single and 32 for double precision.
           Note that this will also affect the used block size"""
        self.max_reg = None

        "Non-gaussianity related variables:"
        
        "Whether to do non-Gaussianity calculations:"
        self.nonGaussianityQ = True
        "Number of different simulations to run with identical intial values:"
        self.sim_num = 2
        

