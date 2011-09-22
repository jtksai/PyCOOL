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

"Create a model:"
model = Model()

"Create a lattice:"
lat = Lattice(model, order = 4, scale = False, init_m = 'defrost_cpu')

" Create a potential function object:"
V = Potential(lat, model)

rho_fields = si.rho_field(lat, V, model.a_in, model.pis0, model.fields0)

"Create simulation, evolution and spectrum instances:"
sim = si.Simulation(model, lat, V, steps = 10000)

evo = si.Evolution(lat, V, sim)
postp = sp.Postprocess(lat, V)

"Create a new folder for the data:"
data_path = make_dir(model, lat, V, sim)

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

#start.record()

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

if model.lin_evo:
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


#evo_homQ = True
evo_homQ = False

if evo_homQ:
    print '\nSolve homogeneous equations:\n'

    while (sim.t_hom<model.t_fin):
        if (sim.i0_hom%(model.flush_freq_hom)==0):
            evo.calc_rho_pres_back(lat, V, sim, print_Q = True)
            #sim.flush(lat, path = data_path)

        #if (sim.i0_hom%(1024)==0):
        #    print 't: ', sim.t_hom*model.m

        sim.i0_hom += 1
        evo.evo_step_bg_4(lat, V, sim, lat.dtau)


if model.nonGaussianityQ:
    print "Running non-Gaussianity simulations:"

    start.record()

    i0_sum = 0

    for i in xrange(model.sim_num):
        print 'Simulation run: ', i

        data_folder = make_subdir(i, path = data_path)

        while (sim.t<model.t_fin):
            if (sim.i0%(model.flush_freq)==0):
                evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
                sim.flush(lat, path = data_folder)

            sim.i0 += 1
            evo.evo_step_2(lat, V, sim, lat.dtau)
            
        evo.calc_rho_pres(lat, V, sim, print_Q = True)
        sim.flush(lat, path = data_folder, save_evo = False)

        i0_sum += sim.i0

        "Calculate spectrums and statistics:"
        if lat.postQ:
            postp.calc_post(lat, V, sim, data_folder, model.spect_m)

        "Re-initialize system:"
        if i < model.sim_num-1:
            sim.reinit(model, lat, model.a_in)
            "Adjust p:"
            evo.calc_rho_pres(lat, V, sim, print_Q = False, print_w=False,
                              flush=False)
            sim.adjust_p(lat)



    "Synchronize:"
    end.record()
    end.synchronize()

    time_sim = end.time_since(start)*1e-3
    per_stp = time_sim/i0_sum
            



#evoQ = True
evoQ = False


if evoQ:

    start.record()

    while (sim.t<model.t_fin):
        if (sim.i0%(model.flush_freq)==0):
            evo.calc_rho_pres(lat, V, sim, print_Q = True, print_w=False)
            sim.flush(lat, path = data_path)

        #if (sim.i0%(1024)==0):
        #    print 'H-hor:', (1./sim.H)/(sim.a*lat.L), 't: ', sim.t*m

        sim.i0 += 1
        evo.evo_step_2(lat, V, sim, lat.dtau)

    end.record()
    end.synchronize()

    evo.calc_rho_pres(lat, V, sim, print_Q = True)
    sim.flush(lat, path = data_path, save_evo = False)

    time_sim = end.time_since(start)*1e-3
    per_stp = time_sim/sim.i0

    "Calculate spectrums and statistics:"
    
    if lat.postQ:
        postp.calc_post(lat, V, sim, data_path, model.spect_m)

    write_csv(data_path)


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

print 'Done.'
