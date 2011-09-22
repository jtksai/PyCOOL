import sys, os
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import numpy.linalg as la

import integrator.symp_integrator as si
import postprocess.spectrum as sp

from lattice import *

"""
###############################################################################
# Import a scalar field model and define a lattice
###############################################################################
"""
"Necessary constants defined in the model file:"

#from models.chaotic import *
#from models.curvaton import *
from models.curvaton_si import *
#from models.oscillon import *
#from models.q_ball import *


"Create a lattice:"

lat = Lattice(n, L = L_m, fields = len(fields0), mpl = mpl, m = m, order = 4,
              dtau = dtau_m, field_rho = field_r, field_lp = field_lpQ,
              m2_eff = m2_effQ, test = testQ, stats = statsQ,
              spects = spectQ, dists = distQ, scale = False,
              init_m = 'defrost_cpu', max_reg = max_reg)

" Create a potential function object:"
V = Potential(lat, model_name, power_list, C_coeff, D_coeff,
              v_list=V_list, v_int=V_int, lin_evo = lin_evo)

rho_fields = si.rho_field(lat, V, a_in, pis0, fields0)

"Create simulation, evolution and spectrum instances:"
sim = si.Simulation(lat, V, t_in, a_in, rho_r0, rho_m0, fields0, pis0,
                    deSitter = deSitterQ, steps = 10000, lin_evo = lin_evo)

evo = si.Evolution(lat, V, sim)
postp = sp.Postprocess(lat, V)

"Create a new folder for the data:"
data_path = make_dir(lat, V, sim)

"""Set the average values of the fields equal to
their homogeneous values:"""

sim.adjust_fields(lat)

"""Canonical momentum p calculated with homogeneous fields.
   Field fluctuations will lead to a small perturbation into p.
   Adjust_p compensates this."""
evo.calc_rho_pres(lat, V, sim, print_Q = False, print_w=False, flush=False)
sim.adjust_p(lat)

"GPU memory status:"
show_GPU_mem()

"Time simulation:"
start = cuda.Event()
end = cuda.Event()

start.record()

"""
################
Start Simulation
################
"""

"""
####################################################################
# Linearized evolution (This has not been tested thoroughly!)
####################################################################
"""

if lin_evo:
    "Save initial data:"
    evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
    sim.flush(lat, path = data_path)

    "Go to Fourier space:"
    evo.x_to_k_space(lat, sim, perturb=True)
    evo.update(lat, sim)


    while sim.a<2.0:
        evo.lin_evo_step(lat, V, sim)
        evo.transform(lat, sim)
        evo.calc_rho_pres_back(lat, V, sim, print_Q = True)

    "Go back to position space:"
    evo.k_to_x_space(lat, sim, unperturb=True)
    sim.adjust_fields(lat)

"""
####################################################################
# Non-linear evolution
####################################################################
"""

print '\nNon-linear simulation:\n'

evo_homQ = True
#evo_homQ = False

if evo_homQ:
    while (sim.t_hom<t_fin):
        #if (sim.i0_hom%(flush_freq)==0):
            #evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
            #sim.flush(lat, path = data_path)

        if (sim.i0_hom%(1024)==0):
            print 't: ', sim.t_hom*m

        sim.i0_hom += 1
        evo.evo_step_bg_2(lat, V, sim, lat.dtau)


#evoQ = True
evoQ = False

if evoQ:
    while (sim.t<t_fin):
        if (sim.i0%(flush_freq)==0):
            evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
            sim.flush(lat, path = data_path)

        #if (sim.i0%(1024)==0):
        #    print 'H-hor:', (1./sim.H)/(sim.a*lat.L), 't: ', sim.t*m

        sim.i0 += 1
        evo.evo_step_2(lat, V, sim, lat.dtau)

end.record()
end.synchronize()

time_sim = end.time_since(start)*1e-3
per_stp = time_sim/sim.i0

evo.calc_rho_pres(lat, V, sim, print_Q = True)
sim.flush(lat, path = data_path, save_evo = False)

"Print simulation time info:"
sim_time(time_sim, per_stp, sim.i0, data_path)

"""
####################
Simulation finished
####################
"""

"""
####################################################################
# Calculate spectrums and statistics
####################################################################
"""

if lat.postQ:
    postp.calc_post(lat, V, sim, data_path, spect_m)

write_csv(data_path)

print 'Done.'
