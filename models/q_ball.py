import numpy as np
pi = np.pi

"""
###############################################################################
# Define a scalar field model and a lattice
###############################################################################
"""

model_name = 'q-ball'

"Model parameters and values:"

"Reduced Planck mass (mpl) and regular Planck mass (MPl):"
"mpl = 2.4353431507105459e+18 GeV"

mpl = 1.0
MPl = np.sqrt(8*pi)*mpl


"Mass unit that is used to define other variables:"
m = 1e2/(2.4353431507105459e+18)

"Scalar field masses:"
m2f1 = 2*m**2.
m2f2 = 2*m**2.

"Coupling strength:"
K = -0.1
M_term = 1e14/(2.4353431507105459e+18)
lamb = 0.5


"Initial values for the fields and the field time derivatives:"

f10 =  2.5e7*m
f20 = 0.

df1_dt0 = 0.
df2_dt0 = 2.5e7*m**2

fields0 = [f10, f20]
pis0 = [df1_dt0, df2_dt0]

"""List of the potential functions:
   (Please use f1, f2, ... for the different fields)"""

"Potentials functions of the fields:"
#V_list = None
V_list = ['0.5*C1*f1**2', '0.5*C2*f2**2']

"Interaction terms of the fields:"
#V_int = ["C1*(log(1+D1*(f1**2+f2**2)))+C2*(f1**2+f2**2)**3"]
V_int = ["C3*(f1**2+f2**2)*(log(D1*(f1**2+f2**2)))+C4*(f1**2+f2**2)**3"]

"Numerical values for C1, C2, ... These will be multiplied by a**3*dtau:"
C_coeff = [m2f1, m2f2, K*m**2, 1.]

"Numerical values for bare coefficients D1, D2, ..."
D_coeff = [1./M_term**2]
#D_coeff = []

"""List of functions which are in 'powerform' in potential. For
   example for potential V = 1 + sin(f1)**2 power_list = ['sin(f1)'].
   Field variables are included automatically."""
power_list = []

"Initial canonical variables:"
t_in = 100./m
t_fin = 10000./m

"Initial scale parameter:"
a_in = 0.1*(t_in*m)**(2./3.)

"Initial values for homogeneous radiation and matter components:"
rho_r0 = 3.*(2./(3.*t_in))**2.
rho_m0 = 0.

"Time step:"
dtau_m = 0.005/m

"Lattice side length:"
L_m = 16./(m)

"Lattice size, where n should be a power of two:"
n = 128

"How frequently to save data:"
flush_freq = 4*1024

"Set if to use linearized evolution:"
lin_evo = False

"If calc_spect = True calculate spectrums at the end:"
calc_spect = True

"""The used method to calculate spectrums. Options 'latticeeasy' and 'defrost'.
   Defrost uses aliasing polynomial to smooth the spectrums."""
spectQ = 'defrost'#'latticeeasy'

"""If field_r = True calculate also energy densities of fields
without interaction terms:"""
field_r = False

"""If field_lpQ = True calculate correlation lengths of the energy densities
   of the fields without interaction terms:"""
field_lpQ = False

"If deSitter = True include -9H^2/(4m^2) terms in \omega_k^2 term:"
deSitterQ = True

"If testQ = True use a constant seed. Can be used for debugging and testing:"
testQ = False

"If m2_effQ = True writes a*m_eff/m to SILO file."
m2_effQ = False

"""Maximum number of registers useb per thread. If set to None uses
default values 24 for single and 32 for double precision.
Note that this will also affect the used block size"""
max_reg = 45

