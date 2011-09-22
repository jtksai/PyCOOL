import numpy as np
pi = np.pi

"""
###############################################################################
# Define a scalar field model and a lattice
###############################################################################
"""

model_name = 'curvaton'

"Model parameters and values:"

"Reduced Planck mass (mpl) and regular Planck mass (MPl):"
mpl = 1.0
MPl = np.sqrt(8*pi)*mpl

"Mass unit that is used to define other variables:"
m = 1e-10

"Scalar field masses:"
m2f1 = m**2.
m2f2 = 0.0

"Coupling strength:"
g2 = 1e-8

"Initial values for the fields and the field time derivatives:"

f10 =  2e-4*mpl
f20 = 1e-16*mpl

df1_dt0 = 10.**-20.
df2_dt0 = 10.**-20.

lamb = 1.0e-20
psi = np.sqrt(mpl)

fields0 = [f10, f20]
pis0 = [df1_dt0, df2_dt0]

q = g2*f10**2/(4*m2f1)

"List of the potential functions:"

"Potentials functions of the fields:"
V_list = ["0.5*C1*f1**2","0.5*C2*f2**2"]

"Interaction terms of the fields:"
V_int = ["0.5*C3*f1**2*f2**2"]

"Numerical values for C1, C2, ... These will be multiplied by a**3*dtau:"
C_coeff = [m2f1, m2f2, g2]

"Numerical values for bare coefficients D1, D2, ..."
D_coeff = []

"""List of functions which are in 'powerform' in potential. For
   example for potential V = 1 + sin(f1)**2 power_list = ['sin(f1)'].
   Field variables are included automatically."""
power_list = []

"Initial values for homogeneous radiation and matter components:"
rho_r0 = 0.25*lamb*psi**4.
rho_m0 = 0.

"Time step:"
dtau_m = 1./(2500*m)
#dtau_m = 1./(1000*m)

"Lattice side length:"
L_m = 5./3./m

"Lattice size, where n should be a power of two:"
n = 32

"Initial scale parameter:"
a_in = 1.

"Initial and final times:"
t_in = 0.
t_fin = 500./m

"How frequently to save data:"
flush_freq = 256

"Set if to use linearized evolution:"
lin_evo = False

"If spectQ = True calculate spectrums at the end:"
spectQ = True

"If distQ = True calculate empirical CDF and CDF at the end:"
distQ = True

"""The used method to calculate spectrums. Options 'latticeeasy' and 'defrost'.
   Defrost uses aliasing polynomial to smooth the spectrums."""
spect_m = 'defrost'#'latticeeasy'

"""If statQ = True calculate skewness and kurtosis of the fields:"""
statsQ = True

"""If field_r = True calculate also energy densities of fields
without interaction terms:"""
field_r = True

"""If field_lpQ = True calculate correlation lengths of the energy densities
   of the fields without interaction terms:"""
field_lpQ = False

"If deSitter = True include -9H^2/(4m^2) terms in \omega_k^2 term:"
deSitterQ = True

"If testQ = True use a constant seed. Can be used for debugging and testing:"
testQ = False

"If m2_effQ = True writes a*m_eff/m to SILO file."
m2_effQ = True

"""Maximum number of registers useb per thread. If set to None uses
default values 24 for single and 32 for double precision.
Note that this will also affect the used block size"""
max_reg = None

